import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import streamlit as st
import os

# Helper to check if running inside Streamlit
def is_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

@st.cache_resource
def get_pipeline():
    """
    Loads and caches the Stable Diffusion XL pipeline.
    Optimizes for 8GB VRAM (RTX 4060) using float16 and attention slicing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    msg = "⏳ Loading Stable Diffusion model (First run will download 2.5 GB of weights. Please wait, this may take a few minutes)..."
    
    if is_in_streamlit():
        with st.spinner(msg):
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                "segmind/SSD-1B",
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            pipeline = pipeline.to(device)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            if device == "cuda":
                pipeline.enable_attention_slicing()
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        pipeline = pipeline.to(device)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        if device == "cuda":
            pipeline.enable_attention_slicing()
            
    return pipeline

from PIL import Image, ImageDraw

def generate_mock_image(prompt):
    """
    Generates a placeholder image with the prompt text.
    Bypasses all model loading/downloads for instant testing.
    """
    # Create a nice dark gradient/themed card (1024x576)
    img = Image.new("RGB", (1024, 576), color="#1e1b4b")
    draw = ImageDraw.Draw(img)
    
    # Draw simple design shapes in the background
    draw.ellipse([80, 80, 280, 280], fill="#312e81")
    draw.ellipse([750, 220, 980, 450], fill="#4338ca")
    
    # Draw title text
    draw.text((40, 40), "[DRAFT MODE - NO GPU MODEL DOWNLOAD REQUIRED]", fill="#818cf8")
    
    # Wrap text manually for prompt display
    words = prompt.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(" ".join(current_line)) > 70:
            lines.append(" ".join(current_line[:-1]))
            current_line = [word]
    lines.append(" ".join(current_line))
    
    wrapped_text = "\n".join(lines)
    
    # Draw the scene prompt text
    draw.text((40, 200), f"Scene Description:\n\n{wrapped_text}", fill="#f4f4f5")
    
    return img

def generate_scene_image(prompt, style_preset="anime style, highly detailed digital painting, vibrant color scheme, 16:9 aspect ratio", mock_mode=False):
    """
    Generates a 16:9 image using the cached SDXL pipeline, or uses mock mode.
    """
    if mock_mode:
        return generate_mock_image(prompt)
        
    pipe = get_pipeline()
    
    # Combine user prompt with style preset
    full_prompt = f"{prompt}, {style_preset}"
    negative_prompt = "deformed, bad anatomy, disfigured, low contrast, low quality, blurry, text, watermark, signature"
    
    # Generate image (1024x576 is native 16:9 for SDXL)
    # Using 12 steps with DPM solver is extremely fast and high-quality
    with torch.inference_mode():
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=12,
            guidance_scale=7.0,
            width=1024,
            height=576
        ).images[0]
        
    return image
