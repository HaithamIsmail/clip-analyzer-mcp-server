import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
from PIL import Image

import modal

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# --- 1. Define the Modal Environment ---
# This section defines the container image for our app.
# It includes all necessary Python packages and pins versions for reproducibility.

app = modal.App("clip-feature-extractor")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi",
        "torch",
        "transformers",
        "Pillow",
        "opencv-python-headless",
        "numpy",
    )
)

# --- 2. Define the Model Class ---

@app.cls(gpu="T4:1",
         scaledown_window=240,
         image=image,
         max_containers=5,
        volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        },
)
class CLIPModel:

    @modal.enter()
    def download_and_load(self):
        """
        This method is called once when the container starts.
        It's the perfect place to download and load the model into memory.
        """
        from transformers import CLIPProcessor, CLIPModel

        print("--- Loading model and processor ---")
        self.device = "cuda"
        MODEL_NAME = "openai/clip-vit-base-patch32"
        
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("--- Model and processor loaded successfully ---")

    @modal.method()
    def get_features(self, inputs: Union[List[Image.Image], List[str]]):
        """
        This method performs the core ML inference. It's called by our API endpoint.
        """
        import torch
        
        if isinstance(inputs[0], Image.Image):
            processed_input = self.processor(
                images=inputs, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                batch_features = self.model.get_image_features(**processed_input)
        else:
            processed_input = self.processor(
                text=inputs, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model.get_text_features(**processed_input)
            
        # 3. Return features as a standard Python list
        return batch_features.cpu().numpy().tolist()

# --- 3. Define the Web API using FastAPI ---

# Pydantic models for API validation
class ImageBatchRequest(BaseModel):
    images: List[str] = Field(..., description="A list of Base64-encoded image strings.")

class TextRequest(BaseModel):
    text: List[str] = Field(..., description="list of texts to embed")

class FeatureResponse(BaseModel):
    features: List[List[float]]

# Create a FastAPI app object
web_app = FastAPI(
    title="Image Feature Extractor API (Modal)",
    description="An API that uses a CLIP model to extract features from a batch of images.",
    version="1.0.0"
)

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """
    This function defines the API routes and serves the FastAPI app.
    It runs in a separate, lightweight container without a GPU.
    """
    @web_app.post("/extract-text-features", response_model=FeatureResponse)
    def extract_text_features(request: TextRequest):
        model = CLIPModel()
        features_list = model.get_features.remote(request.text)
        return {"features": features_list}

    @web_app.post("/extract-image-features", response_model=FeatureResponse)
    def extract_image_features(request: ImageBatchRequest):
        """
        This is the API endpoint. It handles the web request, calls the
        GPU-powered model class for inference, and returns the result.
        """
        # 1. Decode Base64 and convert to CV2 images
        try:
            batch_cv2_frames = []
            for b64_string in request.images:
                img_bytes = base64.b64decode(b64_string)
                img_np_array = np.frombuffer(img_bytes, np.uint8)
                img_cv2 = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
                if img_cv2 is None:
                    raise ValueError("Could not decode one of the images.")
                batch_cv2_frames.append(img_cv2)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        if not batch_cv2_frames:
            raise HTTPException(status_code=400, detail="No images provided.")

        # 2. Convert from CV2 (BGR) to PIL (RGB)
        batch_pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in batch_cv2_frames
        ]

        # 3. Call the remote GPU class to get features
        # This is the magic of Modal: it calls the `get_features` method on the
        # `CLIPModel` class running in a separate, GPU-enabled container.
        model = CLIPModel()
        features_list = model.get_features.remote(batch_pil_images)
        
        return {"features": features_list}

    @web_app.get("/health")
    def health_check():
        return {"status": "ok"}
        
    return web_app