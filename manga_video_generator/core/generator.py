import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
import streamlit as st
import os
import base64
from io import BytesIO

# SD 1.5-based models (non-SDXL) — require StableDiffusionPipeline and smaller resolution
_SD15_MODEL_IDS = {
    "stablediffusionapi/anything-v5",
    "anything-v5",
}

def _is_sdxl(model_id: str) -> bool:
    return model_id.lower() not in {m.lower() for m in _SD15_MODEL_IDS}


# Helper to check if running inside Streamlit
def is_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

@st.cache_resource
def get_pipeline(model_id="segmind/SSD-1B"):
    """
    Loads and caches the Stable Diffusion pipeline (SDXL or SD 1.5).
    Optimizes for 8GB VRAM (RTX 4060) using float16 and attention slicing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    PipelineClass = StableDiffusionXLPipeline if _is_sdxl(model_id) else StableDiffusionPipeline

    msg = f"⏳ Loading image model {model_id} (First run may download model weights. Please wait)..."

    if is_in_streamlit():
        with st.spinner(msg):
            pipeline = PipelineClass.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            pipeline = pipeline.to(device)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            if device == "cuda":
                pipeline.enable_attention_slicing()
    else:
        pipeline = PipelineClass.from_pretrained(
            model_id,
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

GEMINI_IMAGE_MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-3.0-generate-001",
]

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

def generate_gemini_image(prompt, style_preset, api_key, model_id="imagen-4.0-generate-001"):
    """
    Generates an image using Google's Imagen API through the Gemini developer API.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Gemini API key is required for Gemini image generation.")

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError("Install google-genai to use Gemini image generation: pip install google-genai") from exc

    from core.gemini_manager import run_with_rotation

    full_prompt = (
        f"Create a high quality 16:9 YouTube video scene image. {prompt}, {style_preset}. "
        "No text, no captions, no watermark, no logos."
    )
    model_candidates = [model_id] + [name for name in GEMINI_IMAGE_MODELS if name != model_id]
    
    def _call_gemini_image(key):
        client = genai.Client(api_key=key.strip())
        last_err = None
        not_found_models = []
        for candidate_model_id in model_candidates:
            try:
                response = client.models.generate_images(
                    model=candidate_model_id,
                    prompt=full_prompt,
                    config={"number_of_images": 1, "aspect_ratio": "16:9"},
                )
                return response
            except Exception as error:
                err_str = str(error).lower()
                if "not_found" in err_str or "not found" in err_str or "404" in err_str:
                    not_found_models.append(candidate_model_id)
                    last_err = error
                else:
                    # Non-404 error (quota, auth, etc.) — let rotation handle it
                    raise error
        # All models returned 404 — this is a model access issue, not a key quota issue
        raise ValueError(
            f"None of the Imagen models are accessible with this API key. "
            f"Models tried: {not_found_models}. "
            "Imagen requires a paid Google Cloud / AI Studio billing account. "
            f"Last error: {last_err}"
        )

    try:
        response = run_with_rotation(_call_gemini_image)
    except Exception as exc:
        raise ValueError(
            "Gemini Imagen generation failed. Tried all keys/models. "
            f"Last error: {exc}"
        )

    if not getattr(response, "generated_images", None):
        raise ValueError("Gemini image generation returned no images.")

    generated_image = response.generated_images[0]
    image_data = generated_image.image.image_bytes
    if isinstance(image_data, str):
        image_data = base64.b64decode(image_data)

    return Image.open(BytesIO(image_data)).convert("RGB").resize((1024, 576))

def generate_openai_image(prompt, style_preset, api_key, model_id="dall-e-3"):
    """
    Generates an image using OpenAI's DALL-E API with automatic key rotation.
    """
    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API key is required for DALL-E image generation.")

    from core.openai_manager import run_with_openai_rotation
    import requests

    full_prompt = (
        f"Create a high quality 16:9 YouTube video scene image. {prompt}, {style_preset}. "
        "No text, no captions, no watermark, no logos."
    )

    def _call_openai_image(key):
        models_to_try = [model_id] + [name for name in ["dall-e-3", "dall-e-2"] if name != model_id]
        last_err = None
        for model in models_to_try:
            size = "1792x1024" if model == "dall-e-3" else "1024x1024"
            try:
                response = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "prompt": full_prompt,
                        "n": 1,
                        "size": size,
                    },
                    timeout=45,
                )
                response.raise_for_status()
                img_url = response.json()["data"][0]["url"]
                
                # Download image content
                img_res = requests.get(img_url, timeout=30)
                img_res.raise_for_status()
                return img_res.content
            except Exception as e:
                err_msg = str(e).lower()
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        err_msg += " " + e.response.text.lower()
                    except:
                        pass
                if "does not exist" in err_msg or "invalid_value" in err_msg or "not found" in err_msg:
                    last_err = e
                    continue
                else:
                    raise e
        raise last_err

    try:
        image_bytes = run_with_openai_rotation(_call_openai_image, passed_key=api_key)
    except Exception as exc:
        raise ValueError(
            f"OpenAI DALL-E generation failed. Tried all keys/models. Last error: {exc}"
        )

    return Image.open(BytesIO(image_bytes)).convert("RGB").resize((1024, 576))

def generate_scene_image(
    prompt,
    style_preset="anime style, highly detailed digital painting, vibrant color scheme, 16:9 aspect ratio",
    mock_mode=False,
    image_backend="local",
    model_id="segmind/SSD-1B",
    api_key="",
):
    """
    Generates a 16:9 image using local Diffusers, Gemini Imagen, OpenAI DALL-E, or mock mode.
    """
    if mock_mode:
        return generate_mock_image(prompt)

    if image_backend == "openai":
        try:
            return generate_openai_image(prompt, style_preset, api_key, model_id=model_id)
        except Exception as e:
            try:
                import streamlit as st
                st.warning(f"⚠️ OpenAI DALL-E generation failed: {e}. Falling back to local GPU / draft card...", icon="⚠️")
            except:
                print(f"[WARNING] OpenAI DALL-E failed: {e}. Falling back to local GPU / draft...")
            try:
                # Try to fall back to the fast local GPU model (SSD-1B)
                return generate_scene_image(prompt, style_preset, mock_mode=False, image_backend="local", model_id="segmind/SSD-1B")
            except Exception as local_err:
                # If local generation fails (no GPU or CUDA error), fall back to draft placeholder
                return generate_mock_image(prompt)

    if image_backend == "gemini":
        try:
            return generate_gemini_image(prompt, style_preset, api_key, model_id=model_id)
        except Exception as e:
            try:
                import streamlit as st
                st.warning(f"⚠️ Gemini Imagen generation failed: {e}. Falling back to local GPU / draft card...", icon="⚠️")
            except:
                print(f"[WARNING] Gemini Imagen failed: {e}. Falling back to local GPU / draft...")
            try:
                # Try to fall back to the fast local GPU model (SSD-1B)
                return generate_scene_image(prompt, style_preset, mock_mode=False, image_backend="local", model_id="segmind/SSD-1B")
            except Exception as local_err:
                # If local generation fails (no GPU or CUDA error), fall back to draft placeholder
                return generate_mock_image(prompt)
        
    pipe = get_pipeline(model_id)

    # Combine user prompt with style preset
    full_prompt = f"{prompt}, {style_preset}"
    negative_prompt = "deformed, bad anatomy, disfigured, low contrast, low quality, blurry, text, watermark, signature"

    # SDXL: native 1024x576 (16:9), 12 fast steps
    # SD 1.5: native 768x432 (16:9), 20 steps for quality, then upscale
    if _is_sdxl(model_id):
        width, height, steps = 1024, 576, 12
    else:
        width, height, steps = 768, 432, 20

    with torch.inference_mode():
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=7.0,
            width=width,
            height=height,
        ).images[0]

    # Upscale SD 1.5 output to standard 1024x576
    if not _is_sdxl(model_id):
        from PIL import Image as PILImage
        image = image.resize((1024, 576), PILImage.LANCZOS)

    return image