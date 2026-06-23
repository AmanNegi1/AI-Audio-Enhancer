import torch
import gc
import os
import streamlit as st
from PIL import Image


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_i2v_pipeline(model_id: str):
    """
    Returns (PipelineClass, load_kwargs).

    Routing:
      THUDM/CogVideoX-*-I2V         → CogVideoXImageToVideoPipeline  (best quality, 720x480)
      Lightricks/LTX-Video          → LTXImageToVideoPipeline         (fastest, 768x512)
      Wan-AI/Wan2.1-I2V-14B-480P   → WanImageToVideoPipeline         (SOTA, 832x480, ~28 GB)
      Wan-AI/Wan2.1-I2V-14B-720P   → WanImageToVideoPipeline         (SOTA, 1280x720, ~28 GB)
      stabilityai/stable-video-*    → StableVideoDiffusionPipeline    (SVD, reliable)
    """
    mid = model_id.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if "cogvideo" in mid or "cogvideox" in mid:
        from diffusers import CogVideoXImageToVideoPipeline
        return CogVideoXImageToVideoPipeline, {"torch_dtype": dtype}

    if "ltx" in mid:
        from diffusers import LTXImageToVideoPipeline
        return LTXImageToVideoPipeline, {"torch_dtype": dtype}

    if "wan" in mid:
        from diffusers import WanImageToVideoPipeline
        return WanImageToVideoPipeline, {"torch_dtype": dtype}

    # Default: Stable Video Diffusion
    from diffusers import StableVideoDiffusionPipeline
    fp16_dtype = torch.float16 if device == "cuda" else torch.float32
    load_kw: dict = {"torch_dtype": fp16_dtype}
    if device == "cuda":
        load_kw["variant"] = "fp16"
    return StableVideoDiffusionPipeline, load_kw


def _is_large_i2v_model(model_id: str) -> bool:
    """Returns True for models requiring sequential CPU offloading on 8 GB VRAM."""
    mid = model_id.lower()
    return "14b" in mid or "13b" in mid or "12b" in mid


@st.cache_resource
def get_image_to_video_pipeline(model_id: str):
    """
    Loads and caches the image-to-video pipeline with CPU offloading and VAE optimisations.
    """
    clear_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PipelineClass, load_kwargs = _resolve_i2v_pipeline(model_id)

    with st.spinner(f"⏳ Loading `{model_id}` (first run downloads weights)..."):
        pipe = PipelineClass.from_pretrained(model_id, **load_kwargs)

        if device == "cuda":
            # 14B+ models need sequential offload so they don't OOM on 8 GB VRAM
            if _is_large_i2v_model(model_id):
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        else:
            pipe = pipe.to("cpu")

        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

    return pipe


def generate_image_to_video(
    input_image: Image.Image,
    model_id: str = "THUDM/CogVideoX-5b-I2V",
    num_frames: int = 49,
    # SVD-specific params
    motion_bucket_id: int = 127,
    fps: int = 8,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 8,
    # CogVideoX / LTX shared params
    prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    output_path: str = "output_i2v.mp4",
) -> str:
    """
    Animates a single image into a short video using the selected model.

    - CogVideoX-5b-I2V:           best quality, text-guided, 720×480
    - LTX-Video I2V:              fastest, text-guided, 768×512
    - Wan2.1-I2V-14B-480P/720P:  SOTA Alibaba model, text-guided, needs ~28 GB offload
    - SVD-XT:                     reliable classic, motion-bucket control

    Returns the path to the saved MP4.
    """
    from diffusers.utils import export_to_video

    mid = model_id.lower()
    pipe = get_image_to_video_pipeline(model_id)

    if "cogvideo" in mid:
        # CogVideoX I2V — 720x480, 49 frames
        img = input_image.convert("RGB").resize((720, 480))
        output = pipe(
            image=img,
            prompt=prompt if prompt.strip() else "Smooth cinematic camera motion",
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    elif "ltx" in mid:
        # LTX I2V — 768x512, 25 frames default, 24fps
        img = input_image.convert("RGB").resize((768, 512))
        output = pipe(
            image=img,
            prompt=prompt if prompt.strip() else "Smooth cinematic camera motion",
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        fps = max(fps, 24)  # LTX looks best at 24fps

    elif "wan" in mid:
        # Wan2.1 I2V — 480P: 832×480, 720P: 1280×720
        if "720p" in mid:
            w, h = 1280, 720
        else:
            w, h = 832, 480
        img = input_image.convert("RGB").resize((w, h))
        output = pipe(
            image=img,
            prompt=prompt if prompt.strip() else "Smooth cinematic camera motion, gentle ambient movement",
            negative_prompt="blurry, low quality, watermark, text, distorted, deformed, worst quality",
            width=w,
            height=h,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    else:
        # SVD: no text prompt, uses motion_bucket_id
        img = input_image.convert("RGB").resize((1024, 576))
        output = pipe(
            img,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
        )

    frames = output.frames[0]
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    export_to_video(frames, output_path, fps=fps)
    return output_path