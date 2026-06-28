import os
import sys
import subprocess
import shutil
import json
from pathlib import Path

# Load .env file manually if it exists
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

# Set defaults if not already loaded from .env
# Prefer D:\ if available, otherwise fall back to a .cache folder next to this file
_workspace = os.path.dirname(os.path.abspath(__file__))
_d_drive = "D:\\"
_cache_root = _d_drive if os.path.exists(_d_drive) else _workspace

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(_cache_root, ".cache", "huggingface")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = os.path.join(_cache_root, ".cache")

import streamlit as st
import torch
from PIL import Image

# Import core modules
from core.parser import parse_script_to_prompts, refine_scene_pacing
from core.aligner import align_audio_segments
from core.generator import generate_scene_image
from core.custom_generator import generate_custom_image
from core.assembler import assemble_video
from core.audio_first_recaper import create_audio_first_scenes
from core.panel_remixer import create_panel_remix_scenes
from core.text_to_video import generate_text_to_video
from core.image_to_video import generate_image_to_video
from tabs_tools import render_txt2img, render_chat, render_txt2vid, render_img2vid
from utils import render_panel_remix, render_lora
from core.gemini_manager import GeminiKeyManager
from core.openai_manager import OpenAIKeyManager

# Demo Sample Constants
DEMO_SCRIPT = (
    "Hello guys, welcome back to the channel. Today we are looking at Chihiro "
    "and the magical goldfish in Kagurabachi. Prepare for the fight."
)

DEMO_SCENES = [
    {
        "text_segment": "Hello guys, welcome back to the channel.",
        "image_prompt": "minimalist vector art, silhouette of a host standing on a stage with a large screen, dark blue and gold color scheme"
    },
    {
        "text_segment": "Today we are looking at Chihiro and the magical goldfish in Kagurabachi.",
        "image_prompt": "anime style, a boy with spiky black hair wielding a katana, glowing black goldfish swimming around him, detailed"
    },
    {
        "text_segment": "Prepare for the fight.",
        "image_prompt": "anime style dramatic action scene, close up on eyes, combat pose, detailed background"
    }
]

# Initialize Session State
if "script_text" not in st.session_state:
    st.session_state["script_text"] = ""
if "use_demo_audio" not in st.session_state:
    st.session_state["use_demo_audio"] = False
if "use_demo_scenes" not in st.session_state:
    st.session_state["use_demo_scenes"] = False

# Setup folders on D Drive
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(WORKSPACE_DIR, "temp")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "loras"), exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="AI Manga Recap Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Dark Mode Glassmorphism Theme */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #1e1b4b 0%, #09090b 100%);
        color: #f4f4f5;
    }
    div[data-testid="stSidebar"] {
        background-color: rgba(9, 9, 11, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        border: none;
        color: white;
    }
    .status-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .scene-preview-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎬 Manga Video Generator")
    st.write("Automatically render synchronized, copyright-safe recap videos using local AI.")
    
    st.markdown("---")
    st.subheader("🎁 Quick Start Demo")
    st.write("Test the engine instantly without uploading audio or requiring a Gemini Key.")
    
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("🎁 Load Demo"):
            st.session_state["script_text"] = DEMO_SCRIPT
            st.session_state["use_demo_audio"] = True
            st.session_state["use_demo_scenes"] = True
            st.rerun()
    with col_demo2:
        if st.button("🧹 Clear Demo"):
            st.session_state["script_text"] = ""
            st.session_state["use_demo_audio"] = False
            st.session_state["use_demo_scenes"] = False
            st.rerun()
            
    st.markdown("---")
    st.subheader("🔑 API Setup")
    
    # Initialize key manager
    GeminiKeyManager.initialize()
    
    custom_val = st.session_state.get("custom_gemini_key", os.environ.get("GEMINI_API_KEY", ""))
    gemini_key_input = st.text_input(
        "Gemini API Key (Custom)", 
        value=custom_val,
        type="password",
        help="Provide your own Gemini API key. If it exceeds quota, the system will fall back to backup keys."
    )
    
    # Sync with key manager
    GeminiKeyManager.set_custom_key(gemini_key_input)
    
    # Resolve active key
    gemini_key = GeminiKeyManager.get_active_key()
    
    # Display backup key list and statuses
    st.caption("🔄 **Gemini API Key Rotation Pool:**")
    statuses = st.session_state["gemini_keys_status"]
    active_idx = st.session_state.get("current_key_index", 0)
    
    # Custom key status
    custom_status = st.session_state.get("custom_gemini_key_status", "Active")
    if gemini_key_input.strip():
        c_icon = "🟢" if custom_status == "Active" else "🔴"
        c_label = "Active (Custom)" if custom_status == "Active" else "Exhausted"
        st.markdown(f"{c_icon} **Custom Key:** `{gemini_key_input[:6]}...{gemini_key_input[-4:]}` ({c_label})")
        
    for idx, k in enumerate(statuses):
        is_active = (idx == active_idx) and (not gemini_key_input.strip() or custom_status != "Active")
        status_icon = "🟢" if k["status"] == "Active" else "🔴"
        active_badge = " ⚡ *[Current]*" if is_active else ""
        masked_key = f"`{k['key'][:6]}...{k['key'][-4:]}`"
        st.markdown(f"{status_icon} **{k['owner']}**: {masked_key}{active_badge}")

    st.markdown("---")
    st.caption("🔑 **OpenAI (ChatGPT) API Key Configuration**")
    
    # Initialize OpenAI key manager
    OpenAIKeyManager.initialize()
    
    custom_openai_val = st.session_state.get("custom_openai_key", os.environ.get("OPENAI_API_KEY", ""))
    openai_key_input = st.text_input(
        "OpenAI API Key (Custom)", 
        value=custom_openai_val,
        type="password",
        help="Provide your own OpenAI API key. If it exceeds quota, the system will fall back to backup keys."
    )
    
    # Sync with key manager
    OpenAIKeyManager.set_custom_key(openai_key_input)
    
    # Resolve active key
    openai_key = OpenAIKeyManager.get_active_key()
    
    # Display backup key list and statuses
    st.caption("🔄 **OpenAI API Key Rotation Pool:**")
    openai_statuses = st.session_state["openai_keys_status"]
    openai_active_idx = st.session_state.get("openai_current_key_index", 0)
    
    # Custom key status
    custom_openai_status = st.session_state.get("custom_openai_key_status", "Active")
    if openai_key_input.strip():
        c_openai_icon = "🟢" if custom_openai_status == "Active" else "🔴"
        c_openai_label = "Active (Custom)" if custom_openai_status == "Active" else "Exhausted"
        st.markdown(f"{c_openai_icon} **Custom Key:** `{openai_key_input[:6]}...{openai_key_input[-4:]}` ({c_openai_label})")
        
    for idx, k in enumerate(openai_statuses):
        is_openai_active = (idx == openai_active_idx) and (not openai_key_input.strip() or custom_openai_status != "Active")
        status_openai_icon = "🟢" if k["status"] == "Active" else "🔴"
        active_openai_badge = " ⚡ *[Current]*" if is_openai_active else ""
        masked_openai_key = f"`{k['key'][:6]}...{k['key'][-4:]}`"
        st.markdown(f"{status_openai_icon} **{k['owner']}**: {masked_openai_key}{active_openai_badge}")

    st.markdown("---")
    hf_token = st.text_input(
        "Hugging Face Token (Optional)",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="Get a free Read token from huggingface.co/settings/tokens to enable fast, unthrottled model downloads (e.g. 50MB/s vs 100KB/s)."
    )
    
    st.markdown("---")
    st.subheader("⚙️ GPU Status")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"GPU Detected: **{gpu_name}**")
        st.info("System is configured to run Whisper & SDXL locally on VRAM.")
    else:
        st.warning("No GPU detected. Running on CPU (will be slow).")
        
    st.markdown("---")
    # Quick Clean folder helper
    if st.button("🧹 Clear Temp Cache"):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.sidebar.success("Cache cleared successfully!")

# ---------------- Main Dashboard ----------------
st.title("🚀 AI Creative Studio & Recap Engine")

tab_recap, tab_audio_first, tab_panel_remix, tab_txt2img, tab_chat, tab_txt2vid, tab_img2vid, tab_lora = st.tabs([
    "🎬 Manga Video Recap Generator",
    "🎧 Audio-First YouTube Studio",
    "🧩 Panel Remix Studio",
    "🎨 Text to Image",
    "💬 AI Conversation",
    "🎥 Text to Video",
    "🖼️ Image to Video",
    "🧬 LoRA Training Studio",
])

with tab_recap:
    st.write("Convert voiceovers and scripts into moving anime-style videos.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown("<div class='status-card'>", unsafe_allow_html=True)
        st.subheader("📝 Inputs")
        
        # 0. Gemini API Key
        gemini_key_main = st.text_input(
            "Gemini API Key", 
            value=gemini_key if gemini_key else os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Get a free API key from Google AI Studio (aistudio.google.com)",
            key="gemini_key_recap"
        )
        
        hf_token_main = st.text_input(
            "Hugging Face Token (Optional)",
            value=hf_token if hf_token else os.environ.get("HF_TOKEN", ""),
            type="password",
            help="Get a free Read token from huggingface.co/settings/tokens to enable fast, unthrottled model downloads (e.g. 50MB/s vs 100KB/s).",
            key="hf_token_recap"
        )
        
        # 1. Audio Voiceover Upload
        if st.session_state.get("use_demo_audio", False):
            st.info("🔊 **Demo Voiceover Active** (Using auto-generated audio)")
            uploaded_audio = st.file_uploader(
                "Replace Demo Voiceover with custom Audio (.mp3, .wav, .m4a, .mpeg)", 
                type=["mp3", "wav", "m4a", "mpeg"]
            )
            if uploaded_audio is not None:
                st.session_state["use_demo_audio"] = False
                st.session_state["use_demo_scenes"] = False
                audio_file = uploaded_audio
                st.rerun()
            else:
                audio_file = None
        else:
            audio_file = st.file_uploader(
                "Upload Voiceover Audio (.mp3, .wav, .m4a, .mpeg) — optional, auto-generates voice if skipped", 
                type=["mp3", "wav", "m4a", "mpeg"]
            )
        
        # TTS backend selector (shown only when no audio is uploaded)
        if not st.session_state.get("use_demo_audio", False) and audio_file is None:
            from core.tts_manager import OPENAI_VOICES, BARK_PRESETS
            tts_backend = st.selectbox(
                "🔊 Auto-generate voiceover engine",
                options=["gtts", "openai", "bark"],
                format_func=lambda x: {
                    "gtts": "gTTS (free, Indian English accent)",
                    "openai": "OpenAI TTS (tts-1-hd, high quality)",
                    "bark": "Bark (local GPU, very expressive — slow)",
                }[x],
                key="recap_tts_backend",
            )
            if tts_backend == "openai":
                tts_voice = st.selectbox("Voice", OPENAI_VOICES, index=0, key="recap_tts_voice_openai",
                    help="onyx/echo are deep narrative voices.")
                st.caption("Uses your OpenAI key. Works with any language text.")
            elif tts_backend == "bark":
                tts_voice = st.selectbox("Speaker preset", BARK_PRESETS, index=0, key="recap_tts_voice_bark",
                    help="Runs locally on GPU. ~30-60s per sentence. Very expressive.")
                st.caption("⚠️ Long scripts are auto-split into sentence chunks to avoid the 13s truncation limit.")
            else:
                tts_voice = None
        else:
            tts_backend = "gtts"
            tts_voice = None  # won't be used if audio is uploaded
        
        # 2. Script Input
        script_text = st.text_area(
            "Paste Script (use double-line breaks to outline new paragraphs/scene ideas)",
            value=st.session_state["script_text"],
            height=250,
            placeholder="Paragraph 1 description...\n\nParagraph 2 description..."
        )
        st.session_state["script_text"] = script_text
        
        # 3. Style Presets
        style_preset = st.selectbox(
            "Select Art Style preset",
            options=[
                "anime style, highly detailed digital painting, vibrant color scheme, 16:9 aspect ratio",
                "minimalist vector illustration, silhouette art style, dark indigo and gold color palette, graphic novel layout",
                "cyberpunk neon aesthetic, futuristic manga illustration, dark shadows, heavy ink contours",
                "watercolor manga sketch, soft pastel colors, hand-drawn paper texture, highly expressive lines"
            ]
        )

        image_backend_choice = st.selectbox(
            "Image generation model for video",
            options=[
                "Local GPU - segmind/SSD-1B",
                "Local GPU - Anything V5 (anime SD1.5)",
                "Local GPU - CounterfeitXL (anime SDXL)",
                "Gemini API - Imagen 3",
                "Local GPU - Custom Hugging Face model",
            ],
            key="recap_image_backend_choice"
        )
        if image_backend_choice == "Gemini API - Imagen 3":
            recap_image_backend = "gemini"
            recap_image_model_id = "imagen-4.0-generate-001"
            st.caption("Uses your Gemini API key for cloud image generation. Good fallback when local SDXL images are weak.")
        elif image_backend_choice == "Local GPU - Anything V5 (anime SD1.5)":
            recap_image_backend = "local"
            recap_image_model_id = "stablediffusionapi/anything-v5"
            st.caption("SD 1.5-based anime model. Great for stylized manga art. Lower VRAM than SDXL (~4GB).")
        elif image_backend_choice == "Local GPU - CounterfeitXL (anime SDXL)":
            recap_image_backend = "local"
            recap_image_model_id = "gsdf/CounterfeitXL"
            st.caption("SDXL-based anime model with high detail and consistent character rendering.")
        elif image_backend_choice == "Local GPU - Custom Hugging Face model":
            recap_image_backend = "local"
            recap_image_model_id = st.text_input(
                "Custom Hugging Face image model ID",
                value="segmind/SSD-1B",
                key="recap_custom_image_model_id"
            )
        else:
            recap_image_backend = "local"
            recap_image_model_id = "segmind/SSD-1B"
        
        # 4. Draft Mode Toggle
        draft_mode = st.checkbox("⚡ Fast Draft Mode (Instant testing - bypasses 2.5 GB model download/loading)", value=False)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Run Generation Button
        generate_btn = st.button("Generate Video 🎬")
    
    with col_result:
        if generate_btn:
            active_key = gemini_key_main if gemini_key_main else gemini_key
            active_hf_token = hf_token_main if hf_token_main else hf_token
            is_demo = st.session_state.get("use_demo_audio", False)
            
            # Configure Hugging Face token dynamically to speed up downloads
            if active_hf_token.strip():
                os.environ["HF_TOKEN"] = active_hf_token.strip()
            elif "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            
            if not is_demo and not active_key:
                st.error("Please provide a Gemini API Key.")
            elif not script_text.strip():
                st.error("Please paste your script.")
            else:
                st.subheader("⚡ Generation Pipeline")
                
                # Setup audio path
                if is_demo or audio_file is None:
                    # No audio uploaded — auto-generate TTS
                    audio_path = os.path.join(TEMP_DIR, "auto_voiceover.mp3")
                    try:
                        from core.tts_manager import generate_tts
                        _backend_label = {"gtts": "gTTS", "openai": "OpenAI TTS", "bark": "Bark"}.get(tts_backend, tts_backend)
                        st.info(f"**Auto-generating voiceover with {_backend_label}...**")

                        _bark_progress = st.empty()
                        _bark_bar = st.empty()

                        def _bark_cb(current, total, chunk_text):
                            if total == 0:
                                return
                            if current < total:
                                _bark_progress.caption(f"🔊 Bark chunk {current + 1}/{total}: *{chunk_text[:80]}...*")
                                _bark_bar.progress(current / total)
                            else:
                                _bark_progress.empty()
                                _bark_bar.empty()

                        audio_path = generate_tts(
                            script_text, audio_path,
                            backend=tts_backend,
                            voice=tts_voice,
                            openai_key=openai_key,
                            progress_callback=_bark_cb if tts_backend == "bark" else None,
                        )
                        st.success(f"✅ Voiceover generated ({_backend_label}).")
                    except Exception as e:
                        st.error(f"Error generating audio: {e}")
                        st.stop()
                else:
                    # Use uploaded audio file
                    audio_path = os.path.join(TEMP_DIR, audio_file.name)
                    with open(audio_path, "wb") as f:
                        f.write(audio_file.getbuffer())
                    
                status_container = st.container()
                
                # Step 1: Script Parsing
                with status_container:
                    st.write("🔍 **Step 1:** Parsing script into scenes...")
                    try:
                        # Bypass Gemini API key in demo mode if key is not provided
                        if is_demo and st.session_state.get("use_demo_scenes", False) and not active_key:
                            import copy
                            scenes = copy.deepcopy(DEMO_SCENES)
                            scenes = refine_scene_pacing(scenes)
                            st.info("Bypassed Gemini script parser (using pre-defined Kagurabachi scene prompts).")
                        else:
                            scenes = parse_script_to_prompts(script_text, active_key)
                        st.success(f"Found {len(scenes)} scenes/beats.")
                    except Exception as e:
                        st.error(f"Error parsing script: {e}")
                        st.stop()
                        
                # Step 2: Timeline Alignment
                with status_container:
                    st.write("⏱️ **Step 2:** Aligning audio timeline with Whisper...")
                    try:
                        scenes = align_audio_segments(audio_path, scenes)
                        st.success("Audio aligned successfully.")
                        
                        # Display aligned script timestamps in an expander
                        with st.expander("Show Scene Timeline"):
                            for idx, s in enumerate(scenes):
                                st.text(f"Scene #{idx+1} [{s['start']}s - {s['end']}s]: {s['text_segment'][:80]}...")
                    except Exception as e:
                        st.error(f"Error aligning timeline: {e}")
                        st.stop()
                        
                # Step 3: Local Image Generation
                with status_container:
                    selected_model_label = "Gemini Imagen 3" if recap_image_backend == "gemini" else recap_image_model_id
                    st.write(f"🎨 **Step 3:** Generating original scenes with {selected_model_label}...")
                    progress_bar = st.progress(0.0)
                    
                    try:
                        for idx, scene in enumerate(scenes):
                            # Generate unique filename
                            image_filename = f"scene_{idx}_output.png"
                            image_path = os.path.join(TEMP_DIR, image_filename)
                            
                            prompt = scene['image_prompt']
                            st.write(f"🖌️ *Generating Scene #{idx+1} prompt:* `{prompt[:90]}...`")
                            
                            # Call local generator
                            img = generate_scene_image(
                                prompt,
                                style_preset,
                                mock_mode=draft_mode,
                                image_backend=recap_image_backend,
                                model_id=recap_image_model_id,
                                api_key=active_key,
                            )
                            img.save(image_path)
                            
                            scene['image_path'] = image_path
                            progress_bar.progress((idx + 1) / len(scenes))
                            
                        st.success(f"Generated {len(scenes)} original images locally.")
                    except Exception as e:
                        st.error(f"Error generating images: {e}")
                        st.stop()
                        
                # Step 4: Video Stitching
                with status_container:
                    st.write("🎬 **Step 4:** Compiling final video with MoviePy (applying Ken Burns transitions)...")
                    try:
                        video_output_path = os.path.join(OUTPUT_DIR, "final_recap.mp4")
                        assemble_video(scenes, audio_path, video_output_path)
                        st.success("Video compiled successfully!")
                        
                        # Play Video
                        st.subheader("🎉 Final Output Video")
                        st.video(video_output_path)
                        
                        # Download Button
                        with open(video_output_path, "rb") as f:
                            st.download_button(
                                label="Download Full MP4 Video 📥",
                                data=f,
                                file_name="manga_recap_video.mp4",
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"Error compiling video: {e}")
                        st.stop()
        else:
            st.info("Configure your script and audio on the left, then click 'Generate Video' to begin.")
            
    # ---------------- Image Gallery Display ----------------
    if 'scenes' in locals() and scenes:
        st.markdown("---")
        st.subheader("🖼️ Generated Scene Details")
        for idx, scene in enumerate(scenes):
            with st.container():
                st.markdown("<div class='scene-preview-card'>", unsafe_allow_html=True)
                col_text, col_img = st.columns([2, 1])
                with col_text:
                    st.markdown(f"### Scene #{idx+1} `[{scene.get('start', 0.0)}s - {scene.get('end', 5.0)}s]`")
                    st.markdown(f"**Narration Segment:**\n*{scene['text_segment']}*")
                    st.markdown(f"**AI Image Prompt:**\n`{scene['image_prompt']}`")
                with col_img:
                    image_path = scene.get('image_path')
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

with tab_audio_first:
    st.write("Build YouTube-ready visuals from the real voiceover timing first, then generate images and keyframed edits from those audio beats.")

    col_audio_first_in, col_audio_first_out = st.columns([1, 1])

    with col_audio_first_in:
        st.markdown("<div class='status-card'>", unsafe_allow_html=True)
        st.subheader("🎧 Audio-First YouTube Inputs")

        audio_first_key = st.text_input(
            "Gemini API Key (optional for better shot prompts)",
            value=gemini_key if gemini_key else os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            key="gemini_key_audio_first"
        )
        audio_first_file = st.file_uploader(
            "Upload voiceover audio (.mp3, .wav, .m4a, .mpeg)",
            type=["mp3", "wav", "m4a", "mpeg"],
            key="audio_first_voiceover"
        )
        audio_first_content_type = st.selectbox(
            "Video content type",
            options=[
                "Anime / Manga Recap",
                "Education / Explainer",
                "History / Documentary",
                "Tech / Business",
                "Motivation / Self Improvement",
                "General YouTube",
            ],
            index=0,
            key="audio_first_content_type"
        )
        audio_first_style = st.selectbox(
            "Select Art Style preset",
            options=[
                "YouTube retention style, strong first-frame clarity, bold focal subject, cinematic lighting, 16:9 aspect ratio",
                "clean educational explainer style, modern editorial composition, clear visual metaphor, bright professional lighting",
                "documentary realism, authentic environment, cinematic color grade, natural light, detailed scene context",
                "polished business and tech editorial style, sleek composition, premium lighting, crisp details",
                "anime style, highly detailed digital painting, expressive character acting, vibrant color scheme, 16:9 aspect ratio",
                "dark cinematic manga panels, inked linework, dramatic shadows, high contrast",
            ],
            index=4,
            key="audio_first_style"
        )
        audio_first_image_backend_choice = st.selectbox(
            "Image generation model for video",
            options=[
                "Local GPU - segmind/SSD-1B",
                "Local GPU - Anything V5 (anime SD1.5)",
                "Local GPU - CounterfeitXL (anime SDXL)",
                "Gemini API - Imagen 3",
                "Local GPU - Custom Hugging Face model",
            ],
            key="audio_first_image_backend_choice"
        )
        if audio_first_image_backend_choice == "Gemini API - Imagen 3":
            audio_first_image_backend = "gemini"
            audio_first_image_model_id = "imagen-4.0-generate-001"
            st.caption("Uses your Gemini API key for cloud image generation. Keep scene limits low while testing cost/quality.")
        elif audio_first_image_backend_choice == "Local GPU - Anything V5 (anime SD1.5)":
            audio_first_image_backend = "local"
            audio_first_image_model_id = "stablediffusionapi/anything-v5"
            st.caption("SD 1.5-based anime model. Great for stylized manga art. Lower VRAM than SDXL (~4GB).")
        elif audio_first_image_backend_choice == "Local GPU - CounterfeitXL (anime SDXL)":
            audio_first_image_backend = "local"
            audio_first_image_model_id = "gsdf/CounterfeitXL"
            st.caption("SDXL-based anime model with high detail and consistent character rendering.")
        elif audio_first_image_backend_choice == "Local GPU - Custom Hugging Face model":
            audio_first_image_backend = "local"
            audio_first_image_model_id = st.text_input(
                "Custom Hugging Face image model ID",
                value="segmind/SSD-1B",
                key="audio_first_custom_image_model_id"
            )
        else:
            audio_first_image_backend = "local"
            audio_first_image_model_id = "segmind/SSD-1B"
        audio_first_youtube_mode = st.checkbox(
            "YouTube retention mode",
            value=True,
            help="Uses faster visual pacing and stronger viewer-facing edit defaults.",
            key="audio_first_youtube_mode"
        )
        col_beat_a, col_beat_b = st.columns(2)
        with col_beat_a:
            audio_first_max_duration = st.slider(
                "Max seconds per image",
                min_value=3.0,
                max_value=10.0,
                value=4.5,
                step=0.5,
                key="audio_first_max_duration"
            )
        with col_beat_b:
            audio_first_max_words = st.slider(
                "Max words per image",
                min_value=18,
                max_value=70,
                value=30,
                step=2,
                key="audio_first_max_words"
            )
        audio_first_captions = st.checkbox(
            "Burn captions into video",
            value=True,
            help="Recommended for YouTube retention, especially mobile viewers watching without sound.",
            key="audio_first_captions"
        )
        audio_first_draft = st.checkbox(
            "⚡ Fast Draft Mode (placeholder text cards, skips real image generation)",
            value=False,
            key="audio_first_draft"
        )
        audio_first_limit = st.slider(
            "Limit scenes for test run",
            min_value=0,
            max_value=30,
            value=8,
            step=1,
            help="Use 0 for full audio. Keep this small while testing quality and timing.",
            key="audio_first_limit"
        )

        audio_first_generate = st.button("Generate Audio-First Video 🎬", key="audio_first_generate")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_audio_first_out:
        if audio_first_generate:
            if audio_first_file is None:
                st.error("Please upload a voiceover audio file.")
            else:
                st.subheader("⚡ Audio-First Pipeline")
                audio_first_path = os.path.join(TEMP_DIR, f"audio_first_{audio_first_file.name}")
                with open(audio_first_path, "wb") as f:
                    f.write(audio_first_file.getbuffer())

                try:
                    st.write("🎧 **Step 1:** Transcribing audio and creating timed visual beats...")
                    effective_max_duration = min(audio_first_max_duration, 4.5) if audio_first_youtube_mode else audio_first_max_duration
                    effective_max_words = min(audio_first_max_words, 30) if audio_first_youtube_mode else audio_first_max_words
                    audio_first_scenes, transcript_text = create_audio_first_scenes(
                        audio_first_path,
                        api_key=audio_first_key,
                        max_duration=effective_max_duration,
                        max_words=effective_max_words,
                        content_type=audio_first_content_type,
                    )
                    if audio_first_limit > 0:
                        audio_first_scenes = audio_first_scenes[:audio_first_limit]
                    st.success(f"Created {len(audio_first_scenes)} timestamped beats from the voiceover.")

                    with st.expander("Show audio-first beat plan", expanded=True):
                        for idx, scene in enumerate(audio_first_scenes):
                            st.text(f"Scene #{idx+1} [{scene['start']}s - {scene['end']}s] {scene.get('motion', 'auto')}: {scene['text_segment'][:100]}...")
                    with st.expander("Show transcript"):
                        st.write(transcript_text)
                except Exception as e:
                    st.error(f"Error creating audio-first beat plan: {e}")
                    st.stop()

                try:
                    selected_model_label = "Gemini Imagen 3" if audio_first_image_backend == "gemini" else audio_first_image_model_id
                    st.write(f"🎨 **Step 2:** Generating images for audio-timed beats with {selected_model_label}...")
                    progress_bar = st.progress(0.0)
                    for idx, scene in enumerate(audio_first_scenes):
                        image_path = os.path.join(TEMP_DIR, f"audio_first_scene_{idx}_output.png")
                        prompt = f"{scene['image_prompt']}, {audio_first_style}"
                        st.write(f"🖌️ Scene #{idx+1}: `{prompt[:100]}...`")
                        img = generate_scene_image(
                            prompt,
                            audio_first_style,
                            mock_mode=audio_first_draft,
                            image_backend=audio_first_image_backend,
                            model_id=audio_first_image_model_id,
                            api_key=audio_first_key,
                        )
                        img.save(image_path)
                        scene["image_path"] = image_path
                        progress_bar.progress((idx + 1) / len(audio_first_scenes))
                    st.success(f"Generated {len(audio_first_scenes)} images.")
                except Exception as e:
                    st.error(f"Error generating audio-first images: {e}")
                    st.stop()

                try:
                    st.write("🎬 **Step 3:** Assembling keyframed audio-first recap video...")
                    output_path = os.path.join(OUTPUT_DIR, "audio_first_recap.mp4")
                    assemble_video(audio_first_scenes, audio_first_path, output_path, add_captions=audio_first_captions)
                    st.success("Audio-first recap video compiled successfully!")
                    st.video(output_path)
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Audio-First MP4 📥",
                            data=f,
                            file_name="audio_first_youtube_video.mp4",
                            mime="video/mp4"
                        )
                except Exception as e:
                    st.error(f"Error assembling audio-first video: {e}")
                    st.stop()
        else:
            st.info("Upload a voiceover, choose an image model, then generate the audio-first version. Turn on Draft Mode only when you want placeholder text cards for a quick timing test.")

    if 'audio_first_scenes' in locals() and audio_first_scenes:
        st.markdown("---")
        st.subheader("🖼️ Audio-First Scene Details")
        for idx, scene in enumerate(audio_first_scenes):
            with st.container():
                st.markdown("<div class='scene-preview-card'>", unsafe_allow_html=True)
                col_scene_text, col_scene_img = st.columns([2, 1])
                with col_scene_text:
                    st.markdown(f"### Scene #{idx+1} `[{scene.get('start', 0.0)}s - {scene.get('end', 5.0)}s]`")
                    st.markdown(f"**Motion:** `{scene.get('motion', 'auto')}`")
                    st.markdown(f"**Narration Segment:**\n*{scene['text_segment']}*")
                    st.markdown(f"**AI Image Prompt:**\n`{scene['image_prompt']}`")
                with col_scene_img:
                    image_path = scene.get('image_path')
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

render_panel_remix(tab_panel_remix, TEMP_DIR, OUTPUT_DIR, gemini_key, openai_key)
render_txt2img(tab_txt2img, TEMP_DIR, OUTPUT_DIR, hf_token)
render_chat(tab_chat, gemini_key)
render_txt2vid(tab_txt2vid, OUTPUT_DIR)
render_img2vid(tab_img2vid, OUTPUT_DIR)

# ── LoRA Training Studio Tab ──────────────────────────────────────────────────
render_lora(tab_lora, TEMP_DIR)