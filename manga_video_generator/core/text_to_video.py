import torch
import gc
import os
import streamlit as st


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_pipeline(model_id: str):
    """
    Returns (PipelineClass, from_pretrained_kwargs, supports_neg_prompt).

    Model routing:
      Lightricks/LTX-Video          → LTXPipeline          (fast, 768x512, 24fps)
      THUDM/CogVideoX-*             → CogVideoXPipeline    (high quality, 720x480, 8fps)
      genmo/mochi-1-preview         → MochiPipeline        (fluid motion, 848x480, 30fps)
      Wan-AI/Wan2.1-T2V-1.3B       → WanPipeline          (lightweight SOTA, 480p, ~2.6 GB)
      Wan-AI/Wan2.1-T2V-14B        → WanPipeline          (best quality SOTA, 480p, ~28 GB)
      cerspense/zeroscope_*         → DiffusionPipeline    (576x320, legacy)
      damo-vilab/text-to-video-*    → DiffusionPipeline    (256x256, fast draft)
    """
    mid = model_id.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "ltx" in mid:
        from diffusers import LTXPipeline
        return LTXPipeline, {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}, True

    if "cogvideo" in mid:
        from diffusers import CogVideoXPipeline
        supports_neg = "5b" in mid
        return CogVideoXPipeline, {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}, supports_neg

    if "mochi" in mid:
        from diffusers import MochiPipeline
        return MochiPipeline, {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}, True

    if "wan" in mid and "i2v" not in mid:
        from diffusers import WanPipeline
        return WanPipeline, {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}, True

    # Legacy fallback: ModelScope, ZeroScope, etc.
    from diffusers import DiffusionPipeline
    dtype = torch.float16 if device == "cuda" else torch.float32
    return DiffusionPipeline, {"torch_dtype": dtype}, "zeroscope" in mid or "zero" in mid


def _is_large_model(model_id: str) -> bool:
    """Returns True for models that require sequential (aggressive) CPU offloading on 8 GB VRAM."""
    mid = model_id.lower()
    return "14b" in mid or "13b" in mid or "12b" in mid


@st.cache_resource
def get_text_to_video_pipeline(model_id: str):
    """
    Loads and caches the appropriate text-to-video pipeline for the given model.
    CPU offloading and VAE tiling/slicing are applied automatically.
    """
    import psutil
    if _is_large_model(model_id):
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_ram_gb < 28:
            raise RuntimeError(
                f"\u274c Insufficient RAM: {available_ram_gb:.1f} GB available, "
                f"but {model_id} requires at least 28 GB of free RAM. "
                "Please select a smaller model (e.g. Wan2.1-T2V-1.3B or LTX-Video)."
            )
    clear_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PipelineClass, load_kwargs, _ = _resolve_pipeline(model_id)

    # Pre-download the model with percentage progress bar
    from core.downloader import check_and_download_model
    check_and_download_model(model_id)

    with st.spinner(f"⏳ Loading `{model_id}` (first run downloads weights)..."):
        mid_lower = model_id.lower()
        # Wan requires VAE in float32 for quality; load separately then inject
        if "wan" in mid_lower and "i2v" not in mid_lower:
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = WanPipeline.from_pretrained(
                model_id, vae=vae,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
            )
            # flow_shift=3.0 for 480P, 5.0 for 720P
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=3.0
            )
        else:
            pipe = PipelineClass.from_pretrained(model_id, **load_kwargs)

        if device == "cuda":
            # Enable sequential CPU offload for maximum VRAM savings on 8GB GPUs
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    pipe.enable_model_cpu_offload()
            elif hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        else:
            pipe = pipe.to("cpu")

        # VAE memory optimisations — handle both pipeline-level and vae-level
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

    return pipe


def generate_text_to_video(
    prompt: str,
    model_id: str = "Lightricks/LTX-Video",
    num_frames: int = 25,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    fps: int = 24,
    width: int = 768,
    height: int = 512,
    output_path: str = "output_t2v.mp4",
    negative_prompt: str = "blurry, low quality, watermark, text, distorted, deformed, worst quality",
) -> str:
    """
    Generates a video from a text prompt and saves it as an MP4.
    Returns the output path.
    """
    from diffusers.utils import export_to_video

    _, _, supports_neg = _resolve_pipeline(model_id)
    pipe = get_text_to_video_pipeline(model_id)

    mid = model_id.lower()

    # Build call kwargs — each model family has slightly different accepted args
    kwargs: dict = dict(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    if supports_neg and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    # Width/height are supported by LTX, CogVideoX, Wan, Mochi — but NOT legacy models
    if "ltx" in mid or "cogvideo" in mid or "wan" in mid or "mochi" in mid:
        kwargs["width"] = width
        kwargs["height"] = height

    clear_vram()
    with torch.inference_mode():
        output = pipe(**kwargs)
    frames = output.frames[0]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    export_to_video(frames, output_path, fps=fps)
    clear_vram()
    return output_path