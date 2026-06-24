import torch
from diffusers import DiffusionPipeline
import streamlit as st
import os
import gc

# Helper to check if running inside Streamlit
def is_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

def clear_vram():
    """
    Cleans up VRAM cache to prevent OOM conflicts between different models.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def get_custom_pipeline(model_id):
    """
    Loads and caches a custom Stable Diffusion / Z-Image pipeline.
    Ensures safe VRAM loading on the RTX 4060 using CPU offloading if needed.
    """
    # Clean up previous models from VRAM before loading a new one
    clear_vram()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    msg = f"⏳ Loading custom model '{model_id}' (May download several gigabytes of weights if first run)..."
    in_streamlit = is_in_streamlit()
    is_z_image = "Z-Image" in model_id or "Tongyi-MAI" in model_id
    
    try:
        if in_streamlit:
            ctx = st.spinner(msg)
        else:
            class DummyCtx:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            ctx = DummyCtx()
            
        with ctx:
            # Pre-download the model with percentage progress bar
            from core.downloader import check_and_download_model
            use_fp16 = ("stable-diffusion-xl" in model_id.lower() or "sdxl" in model_id.lower()) and device == "cuda" and not is_z_image
            variant = "fp16" if use_fp16 else None
            check_and_download_model(model_id, variant=variant)

            if is_z_image:
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="balanced"
                )
            else:
                dtype = torch.float16 if device == "cuda" else torch.float32
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    variant=variant
                )
            
            if device == "cuda":
                if is_z_image:
                    # device_map="balanced" handles offloading/placement automatically
                    pass
                else:
                    pipeline.enable_model_cpu_offload()
                    if hasattr(pipeline, "enable_attention_slicing"):
                        pipeline.enable_attention_slicing()
            else:
                if not is_z_image:
                    pipeline = pipeline.to("cpu")
                
            return pipeline
    except Exception as e:
        if in_streamlit:
            st.error(f"Failed to load model {model_id}: {e}")
        raise e

def generate_custom_image(prompt, model_id, negative_prompt="", num_inference_steps=20, guidance_scale=7.5, width=1024, height=1024):
    """
    Generates an image using a custom Hugging Face model.
    """
    pipe = get_custom_pipeline(model_id)
    
    with torch.inference_mode():
        is_z_image = "Z-Image" in model_id or "Tongyi-MAI" in model_id
        
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height
        }
        
        # Only pass negative prompt if not empty
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
            
        kwargs["guidance_scale"] = guidance_scale
            
        image = pipe(**kwargs).images[0]
        
    return image
