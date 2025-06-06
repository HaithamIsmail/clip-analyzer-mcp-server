# pipeline_pytorch_optimized.py

import cv2
import os
import base64
import subprocess
import json
import logging
from PIL import Image
import numpy as np
import pandas as pd
from queue import Queue
from threading import Thread
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

# --- Configuration ---
# ## OPTIMIZATION: Centralized configuration. These could be loaded from a file or env variables.
class Config:
    IMAGE_SIZE = (384, 384)
    SIMILARITY_THRESHOLD = 0.8
    OPENAI_MODEL = "gpt-4o"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    WHISPER_MODEL_NAME = "whisper-1"
    
    # ## OPTIMIZATION: Avoid magic numbers. This is skip_frames + 1.
    FRAME_SAMPLING_RATE = 5  # Process every 5th frame for change detection.
    
    # ## OPTIMIZATION: Use os.path.join for portability and defined output dirs.
    OUTPUT_DIR = "output"
    CSV_DIR = os.path.join(OUTPUT_DIR, "csvs")
    
    # Set up OpenAI client from environment variable for better security.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_default_sk_key_here")
    
    # ## OPTIMIZATION: Use ProcessPoolExecutor for CPU-bound tasks, but be mindful of worker count.
    MAX_WORKERS_PROCESSES = os.cpu_count() or 1 # Default to 1 if cpu_count is not available
    MAX_WORKERS_THREADS = 10 # For I/O bound tasks

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = Config()

# ## OPTIMIZATION: Lazily initialize models in functions where they are used, 
def get_clip_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    model.eval()
    return model, processor, device

def get_openai_client():
    if not config.OPENAI_API_KEY or "your_default_sk_key_here" in config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set or is a placeholder.")
    return OpenAI(api_key=config.OPENAI_API_KEY)

# --- Utility Functions ---
def create_directory(path):
    os.makedirs(path, exist_ok=True)

# --- Core Pipeline Functions ---

def _producer_read_frames(video_path, frame_queue, batch_size, sampling_rate):
    """
    (Producer Thread) Reads frames from the video file and puts them into a queue in batches.
    This is an I/O-bound task.
    """
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logging.error(f"[Producer] Could not open video file: {video_path}")
            frame_queue.put(None) # Signal error/end
            return

        frame_index = 0
        while True:
            batch_cv2_frames = []
            while len(batch_cv2_frames) < batch_size:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = video.read()
                if not ret:
                    # End of video
                    if batch_cv2_frames: # Put the last partial batch if it exists
                        frame_queue.put(batch_cv2_frames)
                    frame_queue.put(None) # Sentinel to signal the end
                    logging.info("[Producer] Reached end of video stream.")
                    video.release()
                    return
                
                batch_cv2_frames.append(frame)
                frame_index += sampling_rate

            frame_queue.put(batch_cv2_frames)
    except Exception as e:
        logging.error(f"[Producer] Error: {e}")
        frame_queue.put(None) # Ensure consumer doesn't block forever

def compare_frames_threaded_batched(video_path, batch_size=32, queue_size=2):
    """
    (Consumer) Computes frame similarities using a producer-consumer pattern.
    The main thread consumes batches for GPU processing while a background
    thread produces the next batch by reading from the disk.
    """
    model, preprocess, device = get_clip_model_and_processor()
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

    frame_queue = Queue(maxsize=queue_size)
    producer_thread = Thread(
        target=_producer_read_frames,
        args=(video_path, frame_queue, batch_size, config.FRAME_SAMPLING_RATE),
        daemon=True
    )
    producer_thread.start()
    logging.info("[Consumer] Producer thread started.")

    similarities = []
    last_frame_features = None
    
    with torch.no_grad():
        while True:
            # 1. Get a pre-fetched batch from the queue (blocks until a batch is ready)
            batch_cv2_frames = frame_queue.get()
            
            # 2. Check for the sentinel value (end of stream)
            if batch_cv2_frames is None:
                logging.info("[Consumer] Received sentinel. Finishing processing.")
                break
            
            logging.info(f"[Consumer] Processing a batch of {len(batch_cv2_frames)} frames.")
            
            # 3. Preprocess the batch (CPU work, but fast) and send to GPU
            batch_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_cv2_frames]
            inputs = preprocess(images=batch_images, return_tensors="pt").to(device)
            
            # 4. Get features for the entire batch (heavy GPU work)
            # While this is running, the producer thread is already reading the *next* batch from disk.
            batch_features = model.get_image_features(**inputs)

            # 5. Compare with the last frame of the *previous* batch
            if last_frame_features is not None:
                sim = torch.nn.functional.cosine_similarity(last_frame_features, batch_features[0].unsqueeze(0))
                similarities.append(sim.item())

            # 6. Calculate similarities *within* the current batch
            if len(batch_features) > 1:
                sims_within_batch = torch.nn.functional.cosine_similarity(batch_features[:-1], batch_features[1:])
                similarities.extend(sims_within_batch.cpu().numpy().tolist())

            # 7. Store the features of the last frame for the next iteration
            last_frame_features = batch_features[-1].unsqueeze(0)

    producer_thread.join() # Wait for the producer to finish cleanly
    logging.info(f"Calculated {len(similarities)} similarities using threaded batched processing.")
    return similarities, fps

def chunk_video(video_path):
    """
    Identifies clip boundaries based on frame similarity.
    (This function now calls the new batched version)
    """
    logging.info("Starting video chunking with batched GPU processing...")
    start_time = time.time()
    
    # ## CALL THE NEW BATCHED FUNCTION ##
    similarities, fps = compare_frames_threaded_batched(video_path)
    
    if not similarities:
        return [], [], [], []

    start_frames = [0]
    end_frames = []
    
    for i, score in enumerate(similarities):
        if score < config.SIMILARITY_THRESHOLD:
            # Frame index is based on the sampling rate. The i-th similarity
            # is between frame (i * rate) and ((i+1) * rate).
            # The end of the scene is at frame ((i+1) * rate).
            frame_idx = (i + 1) * config.FRAME_SAMPLING_RATE
            
            # Avoid creating tiny, meaningless clips
            if frame_idx > start_frames[-1] + fps: # Clip must be at least 1 second long
                end_frames.append(frame_idx)
                start_frames.append(frame_idx) # The next clip starts where this one ended

    # The last clip goes to the end of the video
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    end_frames.append(total_frames - 1)

    start_times = [int(f / fps) for f in start_frames]
    # Ensure end time for the last clip is calculated properly or marked as -1
    # For simplicity, we'll calculate it from total_frames
    end_times = [int(f / fps) for f in end_frames]

    logging.info(f"Video chunking finished in {time.time() - start_time:.2f} seconds.")
    logging.info(f"Identified {len(start_frames)} potential clips.")
    
    return start_frames, end_frames, start_times, end_times

def process_clip(args):
    """
    ## OPTIMIZATION: This function is designed to be run in a separate process.
    It handles a single clip: extracts the first frame as a thumbnail and extracts the audio.
    It no longer re-reads video frames for VLM, only the first frame for the thumbnail.
    """
    video_path, video_name, clip_idx, start_frame, end_frame = args
    
    # --- Audio Extraction using ffmpeg ---
    # ## OPTIMIZATION: Use ffmpeg directly for much faster audio extraction.
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        start_time = start_frame / fps
        end_time = (end_frame if end_frame != -1 else total_frames) / fps
        
        # Ensure we don't go past the end of the video
        duration = end_time - start_time
        if duration <= 0:
            logging.warning(f"Clip {clip_idx} has zero or negative duration. Skipping.")
            return None

        output_dir = os.path.join(config.OUTPUT_DIR, video_name, f"clip_{clip_idx}")
        create_directory(output_dir)
        audio_path = os.path.join(output_dir, "audio.mp3")

        # ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-q:a', '0', # High quality audio
            '-map', 'a',
            '-loglevel', 'error', # Suppress verbose output
            audio_path
        ]
        subprocess.run(command, check=True)
        
        # --- Thumbnail and Frame Extraction for VLM ---
        # ## OPTIMIZATION: Only extract frames needed for the summary, not the whole clip.
        base64_frames = []
        frames_to_sample = 5 # Let's take 5 frames spread across the clip for the summary
        frame_indices = np.linspace(start_frame, end_frame if end_frame != -1 else total_frames -1, frames_to_sample, dtype=int)

        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if success:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        video.release()
        
        if not base64_frames:
             logging.warning(f"Could not extract any frames for clip {clip_idx}. Skipping.")
             return None
             
        return {
            "clip_idx": clip_idx,
            "base64_frames": base64_frames,
            "audio_path": audio_path,
            "thumbnail": base64_frames[0]
        }
    except Exception as e:
        logging.error(f"Failed to process clip {clip_idx} for {video_name}: {e}")
        return None


def summarize_clip(clip_data):
    """
    ## OPTIMIZATION: This function is designed to be run in a thread.
    Takes clip data, transcribes audio, and generates a summary with a VLM.
    """
    client = get_openai_client()
    try:
        # Transcribe audio
        with open(clip_data["audio_path"], "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=config.WHISPER_MODEL_NAME,
                file=audio_file,
            ).text
        
        # Generate summary
        prompt = [
            {"role": "system", "content": "You are a video summarization assistant. Create a concise, single-paragraph summary of the provided video clip, integrating visual actions and background elements with the spoken content."},
            {"role": "user", "content": [
                "These are frames from the video clip.",
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpeg;base64,{x}'}}, clip_data["base64_frames"]),
                {"type": "text", "text": f"The audio transcription is: {transcription}"}
            ]},
        ]
        
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=prompt,
            temperature=0.2,
        )
        summary = response.choices[0].message.content
        full_text = summary + "\n\n---Transcription---\n" + transcription
        
        clip_data["summary"] = full_text
        logging.info(f"Successfully summarized clip {clip_data['clip_idx']}.")
        return clip_data

    except Exception as e:
        logging.error(f"Failed to summarize clip {clip_data['clip_idx']}: {e}")
        clip_data["summary"] = "Error during summarization."
        return clip_data


def run_pipeline(video_path, video_uri):
    """Main pipeline execution logic."""
    video_name = os.path.basename(video_path).split(".")[0]
    create_directory(config.OUTPUT_DIR)
    create_directory(config.CSV_DIR)

    # 1. Chunk video based on visual similarity
    start_frames, end_frames, start_times, end_times = chunk_video(video_path)
    if not start_frames:
        logging.error("No clips were generated. Exiting.")
        return

    # 2. Process clips in parallel (CPU-bound tasks: audio/frame extraction)
    logging.info(f"Processing {len(start_frames)} clips using up to {config.MAX_WORKERS_PROCESSES} processes...")
    processed_clip_data = []
    process_args = [(video_path, video_name, i, start, end) for i, (start, end) in enumerate(zip(start_frames, end_frames))]
    
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS_PROCESSES) as executor:
        futures = [executor.submit(process_clip, arg) for arg in process_args]
        for future in as_completed(futures):
            result = future.result()
            if result:
                processed_clip_data.append(result)
    
    # Sort results by clip index to maintain order
    processed_clip_data.sort(key=lambda x: x['clip_idx'])
    
    # 3. Summarize clips in parallel (I/O-bound tasks: API calls)
    logging.info(f"Summarizing {len(processed_clip_data)} clips using up to {config.MAX_WORKERS_THREADS} threads...")
    final_results = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_THREADS) as executor:
        futures = [executor.submit(summarize_clip, data) for data in processed_clip_data]
        for future in as_completed(futures):
            final_results.append(future.result())

    # 4. Format and save results
    final_results.sort(key=lambda x: x['clip_idx'])
    
    # Re-align with original timing info
    df_data = {
        'clip_id': [d['clip_idx'] for d in final_results],
        'video_name': [video_name] * len(final_results),
        'video_uri': [video_uri] * len(final_results),
        'start_time': [start_times[d['clip_idx']] for d in final_results],
        'end_time': [end_times[d['clip_idx']] for d in final_results],
        'summary': [d['summary'] for d in final_results],
        "thumbnail": [d['thumbnail'] for d in final_results]
    }

    df = pd.DataFrame(df_data)
    df_path = os.path.join(config.CSV_DIR, f"{video_name}.csv")
    df.to_csv(df_path, index=False)
    logging.info(f"Results saved to {df_path}")

    # 5. Insert into Milvus (assuming Milvus service is available)
    try:
        # from milvus_service import Milvus
        # Milvus.insert_df(df_path)
        logging.info("Data insertion into Milvus step is complete (mocked).")
    except ImportError:
        logging.warning("`milvus_service` not found. Skipping Milvus insertion.")
    except Exception as e:
        logging.error(f"Failed to insert data into Milvus: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Video Processing Pipeline.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--video_uri", type=str, required=True, help="URI of the video")
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    run_pipeline(args.video_path, args.video_uri)
    logging.info(f"Total pipeline execution time: {time.time() - total_start_time:.2f} seconds.")