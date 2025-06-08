import modal
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI

load_dotenv(find_dotenv(), override=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.0.1",
        "huggingface_hub[hf_transfer]>=0.32.0",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

MODELS_DIR = "/vlm"
MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"

# Although vLLM will download weights on-demand, we want to cache them if possible. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes),
# which act as a "shared disk" that all Modal Functions can access, for our cache.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen_2.5_32b_vllm")

N_GPU = 2  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY = os.getenv("MODAL_API_KEY")

if modal.is_local():
    local_secret = modal.Secret.from_dict({"MODAL_API_KEY": os.getenv("MODAL_API_KEY")})

MINUTES = 60  # seconds

VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    scaledown_window=5 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(
    max_inputs=100
)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # "--max-model-len",
        # "64000",
        "--tensor-parallel-size",
        "2",
        "--api-key",
        "ak-8jNa93JkuvB6Js9Jh7snpS",
    ]
    print(cmd)
    print(" ".join(cmd))

    subprocess.Popen(" ".join(cmd), shell=True)

@app.local_entrypoint()
def test(test_timeout=10 * MINUTES):
    import json
    import time
    import urllib

    print(f"Running health check for server at {serve.get_web_url()}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(serve.get_web_url() + "/health") as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {serve.get_web_url()}"

    print(f"Successful health check for server at {serve.get_web_url()}")

    messages = [{"role": "user", "content": "Testing! Is this thing on?"}]
    print(f"Sending a sample message to {serve.get_web_url()}", *messages, sep="\n")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"messages": messages, "model": MODEL_NAME})
    req = urllib.request.Request(
        serve.get_web_url() + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))

    client = OpenAI(
        base_url=serve.get_web_url() + "/v1",
        api_key=API_KEY
    )
        ## Use image url in the payload
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    chat_completion_from_url = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
            ],
        }],
        model=MODEL_NAME,
        max_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:", result)

    import urllib.request
    import base64

    # Download the image from the URL
    with urllib.request.urlopen(image_url) as img_response:
        image_data = img_response.read()

    # Encode the image as base64
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    prompt = [
        {"role": "system", "content": \
            """You are a video summarization assistant. Your job is to create a consice, short summary of what happens in the video, integrating visual actions and background elements with the spoken content.
            You should mention any relevant details that appear int the frames are not mentioned in the transcription.
            If the transcription is not provide just explain what you see in the frames.
            Never mention that "I'm unable to provide a summary without the audio transcription". Just give me a summary of what is happening.
            """},
        {"role": "user", "content": [
            {"type": "text", "text": f"These are frames from the video clip."},
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpeg;base64,{x}'}}, [image_base64][0:1]),
            {"type": "text", "text": f"The audio transcription is: "}
        ]},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.2,
    )
    summary = response.choices[0].message.content
    print(response)
