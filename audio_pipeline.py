import os
import json
import logging
import argparse
import time
import subprocess
import tempfile

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from tqdm import tqdm

# Download necessary NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Configuration ---
class Config:
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o"
    WHISPER_MODEL = "whisper-1"
    BOUNDARY_SENSITIVITY = 1.5
    OUTPUT_DIR = "final_clips_output"

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = Config()

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)

# --- STAGE 1: Transcription ---

def extract_audio_from_video(video_path):
    """Extracts audio from a video file into a temporary mp3 file."""
    logging.info(f"Extracting audio from {video_path}...")
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"{os.path.basename(video_path)}.mp3")

    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-q:a', '0',  # High quality audio
        '-map', 'a',
        '-loglevel', 'error',
        temp_audio_path
    ]
    try:
        subprocess.run(command, check=True)
        logging.info(f"Audio extracted successfully to temporary file: {temp_audio_path}")
        return temp_audio_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Failed to extract audio with ffmpeg. Ensure ffmpeg is installed and in your PATH. Error: {e}")
        return None

def transcribe_with_vad(audio_path):
    """Transcribes audio using Whisper to get VAD-based segments."""
    logging.info(f"Starting VAD transcription for {audio_path}...")
    client = get_openai_client()

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=config.WHISPER_MODEL,
                file=audio_file,
                response_format='verbose_json',
                timestamp_granularities=['segment']
            )
        segments = response.segments
        logging.info(f"Transcription complete. Found {len(segments)} speech segments.")
        return segments
    except Exception as e:
        logging.error(f"Could not transcribe audio file. Error: {e}")
        return []

# --- STAGE 2: Semantic Chunking ---

def preprocess_segments_into_sentences(segments):
    """Breaks VAD segments into sentences with approximated timestamps."""
    logging.info("Processing segments into sentences...")
    sentences = []
    for segment in segments:
        segment_text = segment['text'].strip()
        if not segment_text: continue

        segment_duration = segment['end'] - segment['start']
        sents_in_segment = nltk.sent_tokenize(segment_text)
        
        char_offset = 0
        total_chars = len(segment_text)
        for sent in sents_in_segment:
            if not sent.strip(): continue
            
            start_ratio = char_offset / total_chars if total_chars > 0 else 0
            sent_char_len = len(sent)
            end_ratio = (char_offset + sent_char_len) / total_chars if total_chars > 0 else 1
            
            start_time = segment['start'] + start_ratio * segment_duration
            end_time = segment['start'] + end_ratio * segment_duration
            
            sentences.append({'text': sent, 'start': start_time, 'end': end_time})
            char_offset += sent_char_len
            
    logging.info(f"Processed into {len(sentences)} sentences.")
    return sentences

def get_embeddings(sentences):
    """Generates embeddings for a list of sentences."""
    if not sentences: return np.array([])
    logging.info(f"Generating embeddings for {len(sentences)} sentences...")
    client = get_openai_client()
    sentence_texts = [s['text'] for s in sentences]
    response = client.embeddings.create(input=sentence_texts, model=config.EMBEDDING_MODEL)
    return np.array([item.embedding for item in response.data])

def find_topic_boundaries(embeddings):
    """Finds topic boundaries by detecting sharp drops in similarity."""
    if len(embeddings) < 2: return []
    logging.info("Calculating semantic similarity and finding boundaries...")
    similarities = [cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i+1].reshape(1, -1))[0, 0] for i in range(len(embeddings) - 1)]
    mean_sim, std_sim = np.mean(similarities), np.std(similarities)
    threshold = mean_sim - config.BOUNDARY_SENSITIVITY * std_sim
    logging.info(f"Similarity stats: Mean={mean_sim:.3f}, StdDev={std_sim:.3f}, Threshold={threshold:.3f}")
    return [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

def group_sentences_into_clips(sentences, boundaries):
    """Groups sentences into clips based on boundaries."""
    if not sentences: return []
    clips, last_boundary = [], 0
    for boundary in boundaries:
        clips.append(sentences[last_boundary:boundary])
        last_boundary = boundary
    clips.append(sentences[last_boundary:])
    logging.info(f"Grouped sentences into {len(clips)} topical clips.")
    return clips

# --- STAGE 3: LLM Enrichment ---

def enrich_clip_with_llm(clip_sentences):
    """Uses an LLM to generate a title and summary for a clip."""
    full_text = " ".join([s['text'] for s in clip_sentences])
    start_time, end_time = clip_sentences[0]['start'], clip_sentences[-1]['end']

    prompt = f"""You are a video content producer. The following is a transcript of a short video clip. Your task is to create a compelling title and a concise summary.

    Transcript: "{full_text}"

    Provide your response in a structured JSON format with two keys: "title" (max 10 words) and "summary" (one paragraph)."""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        analysis = json.loads(response.choices[0].message.content)
        return {"title": analysis.get("title"), "summary": analysis.get("summary"), "start_time": start_time, "end_time": end_time, "full_transcript": full_text}
    except Exception as e:
        logging.error(f"LLM enrichment failed: {e}")
        return {"title": "Processing Error", "summary": "Could not generate summary.", "start_time": start_time, "end_time": end_time, "full_transcript": full_text}

# --- Main Pipeline ---

def run_full_pipeline(media_path):
    """Runs the full pipeline from media file to enriched topical clips."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(media_path))[0]

    # STAGE 1: Transcription
    is_video = media_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    if is_video:
        audio_path = extract_audio_from_video(media_path)
        if not audio_path: return
    else:
        audio_path = media_path

    segments = transcribe_with_vad(audio_path)
    if is_video:
        os.remove(audio_path) # Clean up temporary audio file
    if not segments:
        logging.error("Pipeline aborted: transcription failed or produced no segments.")
        return

    # STAGE 2: Semantic Chunking
    sentences = preprocess_segments_into_sentences(segments)
    embeddings = get_embeddings(sentences)
    boundaries = find_topic_boundaries(embeddings)
    clips_as_sentences = group_sentences_into_clips(sentences, boundaries)

    # STAGE 3: LLM Enrichment
    enriched_clips = []
    logging.info(f"Enriching {len(clips_as_sentences)} clips with LLM...")
    for clip_sents in tqdm(clips_as_sentences, desc="Enriching Clips"):
        if clip_sents:
            enriched_clips.append(enrich_clip_with_llm(clip_sents))
    
    # Final Output
    output_filename = os.path.join(config.OUTPUT_DIR, f"{base_name}_clips.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(enriched_clips, f, indent=2, ensure_ascii=False)
    logging.info(f"Successfully saved {len(enriched_clips)} clips to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect topical clips from a video or audio file.")
    parser.add_argument("--media_path", type=str, required=True, help="Path to the video or audio file.")
    args = parser.parse_args()

    total_start_time = time.time()
    run_full_pipeline(args.media_path)
    logging.info(f"Total pipeline execution time: {time.time() - total_start_time:.2f} seconds.")