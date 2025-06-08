import os
import re
import requests
import io
import base64
from PIL import Image
from typing import List, Union

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(text) if text.isdigit() else text for text in parts]

def get_text_embedding(text: Union[str, List[str]], embedding_url: str) -> List[List[float]]:
    """
    Send text to the CLIP model and get feature embeddings.
    
    Args:
        text (str): The text to embed
        api_url (str): Base URL of your Modal API
        
    Returns:
        List[List[float]]: Feature embeddings for the text
    """

    if isinstance(text, str):
        text = [text]
        
    endpoint = f"{embedding_url}/extract-text-features"
    
    print(text)

    payload = {
        "text": text
    }
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["features"]
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling text features API: {e}")
        raise
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        raise


def get_image_embedding(images: List[str], embedding_url: str) -> List[List[float]]:
    """
    Send images to the CLIP model and get feature embeddings.
    
    Args:
        images (List[str]): List of image file paths or PIL Images
        api_url (str): Base URL of your Modal API
        
    Returns:
        List[List[float]]: Feature embeddings for each image
    """
    endpoint = f"{embedding_url}/extract-image-features"
    
    # Convert images to base64 strings
    base64_images = []
    
    for img in images:
        if isinstance(img, str):
            # Assume it's a file path
            # with open(img, "rb") as image_file:
                # img_bytes = image_file.read()
            img_bytes = img
            base64_images.append(img_bytes)
        elif isinstance(img, Image.Image):
            # PIL Image object
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            # Encode to base64
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(base64_string)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    payload = {
        "images": base64_images
    }
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["features"]
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling image features API: {e}")
        raise
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        raise