import os

# Load .env file manually if it exists
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

# Set defaults if not already loaded from .env
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "D:\\.cache\\huggingface"
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = "D:\\.cache"

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

def test_pipeline():
    print("--- AI Manga Generator Test Script ---")
    
    # 1. Check GPU
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        dtype = torch.float16
    else:
        print("GPU NOT FOUND! Running on CPU (Warning: This will be very slow).")
        device = "cpu"
        dtype = torch.float32
        
    # 2. Load Model (This will show the progress bar in the terminal!)
    print("\n[INFO] Loading model 'segmind/SSD-1B'...")
    print("If this is the first run, it will download 2.5 GB of weights. Progress will show below:\n")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    
    # Apply fast scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_attention_slicing()
        
    print("\n[SUCCESS] Model loaded successfully!")
    
    # 3. Generate test image
    prompt = "anime style, a cute white cat sitting on a roof under a blue sky, detailed, 16:9 aspect ratio"
    print(f"\n[INFO] Generating test image with prompt: {prompt}")
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=12,
        guidance_scale=7.0,
        width=1024,
        height=576
    ).images[0]
    
    output_filename = "test_anime.png"
    image.save(output_filename)
    print(f"\n[SUCCESS] Test image saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    test_pipeline()
