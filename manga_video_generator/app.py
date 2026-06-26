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
        
        # TTS Voice selector (shown only when no audio is uploaded)
        if not st.session_state.get("use_demo_audio", False) and audio_file is None:
            tts_voice = st.selectbox(
                "Auto-generate voice accent (used when no audio uploaded)",
                options=[
                    "🇮🇳 Indian English",
                    "🇮🇳 Hinglish (Roman script)",
                    "🇮🇳 Hindi (Devanagari script)",
                ],
                index=0,
                help="Indian English & Hinglish both use an Indian-accented English voice — best for Roman-script Hinglish. Hindi mode expects Devanagari text."
            )
        else:
            tts_voice = "🇮🇳 Indian English"  # default, won't be used if audio is uploaded
        
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
                    # No audio uploaded — auto-generate TTS based on selected accent
                    audio_path = os.path.join(TEMP_DIR, "auto_voiceover.mp3")
                    try:
                        from gtts import gTTS
                        # Map UI selection to gTTS parameters
                        _tts_configs = {
                            "🇮🇳 Indian English":             {"lang": "en", "tld": "co.in"},
                            "🇮🇳 Hinglish (Roman script)":   {"lang": "en", "tld": "co.in"},  # best gTTS approximation for Roman Hinglish
                            "🇮🇳 Hindi (Devanagari script)": {"lang": "hi", "tld": "com"},
                        }
                        cfg = _tts_configs.get(tts_voice, {"lang": "en", "tld": "co.in"})
                        accent_label = tts_voice if not is_demo else "🇮🇳 Indian English"
                        st.info(f"**Auto-generating voiceover — {accent_label}...**")
                        # TODO: Replace gTTS with custom voice model here
                        # e.g. tts = CustomVoiceModel(voice_id="your_voice").generate(script_text)
                        tts = gTTS(text=script_text, **cfg)
                        tts.save(audio_path)
                        st.success(f"✅ Voiceover generated ({accent_label}).")
                    except Exception as e:
                        st.error(f"Error generating audio with gTTS: {e}")
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

with tab_panel_remix:
    st.header("🧩 Panel Remix Studio")
    st.write("Use uploaded manga panels as storyboard references, then generate new original anime-style recap visuals with richer prompts.")

    col_panel_in, col_panel_out = st.columns([1, 1])

    with col_panel_in:
        st.markdown("<div class='status-card'>", unsafe_allow_html=True)
        st.subheader("📚 Source Panels & Narration")

        panel_files = st.file_uploader(
            "Upload 10-20 black-and-white manga panel images (.png, .jpg, .webp)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="panel_remix_uploads"
        )
        panel_voiceover = st.file_uploader(
            "Optional voiceover audio for timing (.mp3, .wav, .m4a, .mpeg)",
            type=["mp3", "wav", "m4a", "mpeg"],
            key="panel_remix_voiceover"
        )
        panel_narration_text = st.text_area(
            "Optional script / narration text (used when no voiceover is uploaded)",
            height=150,
            placeholder="Paste the recap narration here, or upload voiceover audio above.",
            key="panel_remix_narration"
        )

        st.markdown("---")
        st.subheader("🧠 Panel Analysis")
        panel_analyzer_backend_choice = st.selectbox(
            "Panel understanding model",
            options=["Gemini API - Vision", "OpenAI API - Vision"],
            key="panel_remix_analyzer_backend"
        )
        panel_analyzer_backend = "openai" if "OpenAI" in panel_analyzer_backend_choice else "gemini"
        panel_analyzer_key = st.text_input(
            "Panel analysis API key",
            value=gemini_key if panel_analyzer_backend == "gemini" else openai_key,
            type="password",
            help="Gemini key for Gemini Vision, or OpenAI key for OpenAI Vision.",
            key="panel_remix_analyzer_key"
        )
        panel_openai_model = "gpt-4o-mini"
        if panel_analyzer_backend == "openai":
            panel_openai_model = st.text_input(
                "OpenAI vision model",
                value="gpt-4o-mini",
                key="panel_remix_openai_model"
            )

        col_panel_detail_a, col_panel_detail_b = st.columns(2)
        with col_panel_detail_a:
            panel_detail_strength = st.selectbox(
                "Prompt detail strength",
                options=["Medium", "High", "Ultra"],
                index=1,
                key="panel_remix_detail_strength"
            )
        with col_panel_detail_b:
            panel_originality = st.slider(
                "Originality strength",
                min_value=60,
                max_value=100,
                value=88,
                step=2,
                help="Higher values push stronger redesign of faces, hair, outfits, props, background, and framing.",
                key="panel_remix_originality"
            )

        panel_art_style = st.selectbox(
            "Generated art style",
            options=[
                "anime style, highly detailed digital painting, cinematic lighting, expressive character acting, 16:9 aspect ratio",
                "dark cinematic manga-inspired anime art, dramatic shadows, inked energy, rich background detail, 16:9 aspect ratio",
                "premium YouTube anime recap visual, bold focal subject, dramatic color, high-retention composition, 16:9 aspect ratio",
                "watercolor manga-inspired anime illustration, soft atmospheric lighting, hand-painted texture, 16:9 aspect ratio",
            ],
            index=0,
            key="panel_remix_art_style"
        )

        st.markdown("---")
        st.subheader("🎨 Image Generation")
        panel_image_backend_choice = st.selectbox(
            "Image generation model",
            options=[
                "Local GPU - segmind/SSD-1B",
                "Local GPU - Anything V5 (anime SD1.5)",
                "Local GPU - CounterfeitXL (anime SDXL)",
                "Gemini API - Imagen",
                "OpenAI API - DALL-E 3",
                "Local GPU - Custom Hugging Face model",
            ],
            key="panel_remix_image_backend_choice"
        )
        if panel_image_backend_choice == "Gemini API - Imagen":
            panel_image_backend = "gemini"
            panel_image_model_id = "imagen-4.0-generate-001"
            panel_image_api_key = st.text_input(
                "Gemini image API key",
                value=gemini_key,
                type="password",
                key="panel_remix_image_key"
            )
        elif panel_image_backend_choice == "OpenAI API - DALL-E 3":
            panel_image_backend = "openai"
            panel_image_model_id = "dall-e-3"
            panel_image_api_key = st.text_input(
                "OpenAI image API key",
                value=openai_key,
                type="password",
                key="panel_remix_image_key"
            )
        elif panel_image_backend_choice == "Local GPU - Anything V5 (anime SD1.5)":
            panel_image_backend = "local"
            panel_image_model_id = "stablediffusionapi/anything-v5"
            panel_image_api_key = ""
            st.caption("SD 1.5-based anime model. Great for stylized manga art. Lower VRAM than SDXL (~4GB).")
        elif panel_image_backend_choice == "Local GPU - CounterfeitXL (anime SDXL)":
            panel_image_backend = "local"
            panel_image_model_id = "gsdf/CounterfeitXL"
            panel_image_api_key = ""
            st.caption("SDXL-based anime model with high detail and consistent character rendering.")
        elif panel_image_backend_choice == "Local GPU - Custom Hugging Face model":
            panel_image_backend = "local"
            panel_image_model_id = st.text_input(
                "Custom Hugging Face image model ID",
                value="segmind/SSD-1B",
                key="panel_remix_custom_image_model_id"
            )
            panel_image_api_key = ""
        else:
            panel_image_backend = "local"
            panel_image_model_id = "segmind/SSD-1B"
            panel_image_api_key = ""

        # ── Character LoRA (optional) ──────────────────────────────────────────
        st.markdown("---")
        _loras_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
        _available_loras = sorted([
            d.name for d in Path(_loras_dir).iterdir()
            if d.is_dir() and (d / "pytorch_lora_weights.safetensors").exists()
        ]) if os.path.isdir(_loras_dir) else []
        panel_lora_choice = st.selectbox(
            "🧬 Character LoRA — maintain consistent character appearance (optional)",
            ["None"] + _available_loras,
            key="panel_lora_choice",
            help="Train a LoRA in the LoRA Training Studio tab, then select it here for character consistency.",
        )
        if panel_lora_choice != "None":
            panel_lora_path = os.path.join(_loras_dir, panel_lora_choice)
            _col_ls, _col_lt = st.columns(2)
            with _col_ls:
                panel_lora_scale = st.slider(
                    "LoRA strength", 0.0, 1.5, 0.8, step=0.05, key="panel_lora_scale"
                )
            with _col_lt:
                panel_lora_trigger = st.text_input(
                    "Trigger word",
                    key="panel_lora_trigger",
                    help="The instance token used during training, e.g. 'chihiro_kb'",
                )
        else:
            panel_lora_path = None
            panel_lora_scale = 0.8
            panel_lora_trigger = ""
        st.markdown("---")

        # ── IP-Adapter ───────────────────────────────────────────────────────
        panel_ip_adapter = st.checkbox(
            "🎭 IP-Adapter — auto reference each scene's source panel for character consistency",
            value=False,
            key="panel_ip_adapter_enabled",
            help=(
                "Each generated scene uses its matching manga panel as a visual reference. "
                "No training required — works for all characters instantly. "
                "Best combined with LoRA for maximum consistency (~80–85% alone, ~95% with LoRA)."
            ),
        )
        if panel_ip_adapter:
            panel_ip_scale = st.slider(
                "IP-Adapter strength",
                min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                key="panel_ip_scale",
                help="0.5–0.7 balances character likeness with prompt creativity. Go higher for strict character lock-in.",
            )
        else:
            panel_ip_scale = 0.6
        st.markdown("---")

        col_panel_limit_a, col_panel_limit_b = st.columns(2)
        with col_panel_limit_a:
            panel_scene_limit = st.slider(
                "Limit scenes for test run",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
                help="Use 0 for all beats (recommended — generates an image for every audio beat). Set a small number only while quickly testing prompt quality or managing API cost.",
                key="panel_remix_scene_limit"
            )
        with col_panel_limit_b:
            panel_captions = st.checkbox(
                "Burn captions into MP4",
                value=True,
                key="panel_remix_captions"
            )
        panel_draft = st.checkbox(
            "⚡ Fast Draft Mode (placeholder text cards, skips real image generation)",
            value=False,
            key="panel_remix_draft"
        )

        panel_generate = st.button("Generate Panel Remix Visuals 🎨", key="panel_remix_generate")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_panel_out:
        if panel_generate:
            if not panel_files:
                st.error("Please upload at least one manga panel image.")
            elif not panel_analyzer_key.strip():
                st.error("Please provide an API key for panel analysis.")
            elif panel_voiceover is None and not panel_narration_text.strip():
                st.error("Please upload a voiceover or paste narration text.")
            else:
                st.subheader("⚡ Panel Remix Pipeline")
                panel_images = []
                for panel_file in panel_files:
                    try:
                        panel_images.append(Image.open(panel_file).convert("RGB"))
                    except Exception as e:
                        st.error(f"Could not read panel image {panel_file.name}: {e}")
                        st.stop()

                panel_audio_path = None
                if panel_voiceover is not None:
                    panel_audio_path = os.path.join(TEMP_DIR, f"panel_remix_{panel_voiceover.name}")
                    with open(panel_audio_path, "wb") as f:
                        f.write(panel_voiceover.getbuffer())

                try:
                    st.write("🧠 **Step 1:** Analyzing panels and building rich original prompts...")
                    panel_scenes, panel_transcript = create_panel_remix_scenes(
                        panel_images,
                        api_key=panel_analyzer_key,
                        analyzer_backend=panel_analyzer_backend,
                        narration_text=panel_narration_text,
                        audio_path=panel_audio_path,
                        detail_strength=panel_detail_strength,
                        art_style=panel_art_style,
                        originality_strength=panel_originality,
                        max_duration=4.5,
                        max_words=30,
                        openai_model=panel_openai_model,
                    )
                    if panel_scene_limit > 0:
                        panel_scenes = panel_scenes[:panel_scene_limit]
                    st.success(f"Created {len(panel_scenes)} panel-guided original scene prompts.")

                    with st.expander("Show generated prompt plan", expanded=True):
                        for idx, scene in enumerate(panel_scenes):
                            st.markdown(f"**Scene #{idx+1}** · Source panel #{scene.get('source_panel_index', 0) + 1}")
                            st.caption(scene.get("panel_description", ""))
                            scene["image_prompt"] = st.text_area(
                                f"Editable image prompt #{idx+1}",
                                value=scene.get("image_prompt", ""),
                                height=170,
                                key=f"panel_remix_prompt_{idx}"
                            )
                    if panel_transcript:
                        with st.expander("Show transcript / narration"):
                            st.write(panel_transcript)
                except Exception as e:
                    st.error(f"Error creating panel remix prompts: {e}")
                    st.stop()

                try:
                    selected_panel_model = "Gemini Imagen" if panel_image_backend == "gemini" else panel_image_model_id
                    st.write(f"🎨 **Step 2:** Generating remixed original images with {selected_panel_model}...")
                    progress_bar = st.progress(0.0)
                    for idx, scene in enumerate(panel_scenes):
                        image_path = os.path.join(TEMP_DIR, f"panel_remix_scene_{idx}_output.png")
                        st.write(f"🖌️ Scene #{idx+1}: `{scene['image_prompt'][:110]}...`")
                        # Resolve the source panel for IP-Adapter reference
                        _ip_ref = None
                        if panel_ip_adapter and panel_images:
                            _src_idx = min(scene.get("source_panel_index", 0), len(panel_images) - 1)
                            _ip_ref = panel_images[_src_idx]
                        img = generate_scene_image(
                            scene["image_prompt"],
                            panel_art_style,
                            mock_mode=panel_draft,
                            image_backend=panel_image_backend,
                            model_id=panel_image_model_id,
                            api_key=panel_image_api_key,
                            lora_path=panel_lora_path,
                            lora_scale=panel_lora_scale,
                            lora_trigger=panel_lora_trigger,
                            ip_adapter_image=_ip_ref,
                            ip_adapter_scale=panel_ip_scale,
                        )
                        img.save(image_path)
                        scene["image_path"] = image_path
                        progress_bar.progress((idx + 1) / len(panel_scenes))
                    st.success(f"Generated {len(panel_scenes)} remixed images.")
                except Exception as e:
                    st.error(f"Error generating panel remix images: {e}")
                    st.stop()

                if panel_audio_path:
                    try:
                        st.write("🎬 **Step 3:** Assembling panel remix recap video...")
                        panel_output_path = os.path.join(OUTPUT_DIR, "panel_remix_recap.mp4")
                        assemble_video(panel_scenes, panel_audio_path, panel_output_path, add_captions=panel_captions)
                        st.success("Panel remix recap video compiled successfully!")
                        st.video(panel_output_path)
                        with open(panel_output_path, "rb") as f:
                            st.download_button(
                                label="Download Panel Remix MP4 📥",
                                data=f,
                                file_name="panel_remix_recap.mp4",
                                mime="video/mp4",
                                key="panel_remix_download"
                            )
                    except Exception as e:
                        st.error(f"Error assembling panel remix video: {e}")
                        st.stop()
                else:
                    st.info("Images generated. Upload voiceover audio next time if you also want an assembled MP4.")
        else:
            st.info("Upload manga panels, add voiceover or narration text, then generate original panel-guided visuals.")

    if 'panel_scenes' in locals() and panel_scenes:
        st.markdown("---")
        st.subheader("🖼️ Panel Remix Scene Details")
        for idx, scene in enumerate(panel_scenes):
            with st.container():
                st.markdown("<div class='scene-preview-card'>", unsafe_allow_html=True)
                col_panel_text, col_panel_img = st.columns([2, 1])
                with col_panel_text:
                    st.markdown(f"### Scene #{idx+1} `[{scene.get('start', 0.0)}s - {scene.get('end', 5.0)}s]`")
                    st.markdown(f"**Source Panel:** #{scene.get('source_panel_index', 0) + 1}")
                    st.markdown(f"**Narration Segment:**\n*{scene.get('text_segment', '')}*")
                    st.markdown(f"**Panel Analysis:**\n{scene.get('panel_description', '')}")
                    st.markdown(f"**Generated Prompt:**\n`{scene.get('image_prompt', '')}`")
                with col_panel_img:
                    image_path = scene.get('image_path')
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

render_txt2img(tab_txt2img, TEMP_DIR, OUTPUT_DIR, hf_token)
render_chat(tab_chat, gemini_key)
render_txt2vid(tab_txt2vid, OUTPUT_DIR)
render_img2vid(tab_img2vid, OUTPUT_DIR)

# ── LoRA Training Studio Tab ──────────────────────────────────────────────────
with tab_lora:
    st.subheader("🧬 LoRA Training Studio")
    st.markdown(
        """
        Train a lightweight **DreamBooth LoRA** on your character's manga panels so every generated
        scene shows the same face / costume. Training runs in the background — you can continue
        using other tabs.

        **Recommended workflow:**
        1. Crop 10–20 clean panels of the character you want to be consistent.
        2. Enter a unique trigger word (e.g. `chihiro_kb`), pick the same base model you use in Panel Remix.
        3. Hit **Start Training** — takes ~10–30 min on an RTX GPU.
        4. In **Panel Remix**, select the trained LoRA and paste the same trigger word.
        """
    )

    # Check peft availability
    try:
        import peft  # noqa: F401
        _peft_ok = True
    except ImportError:
        _peft_ok = False

    if not _peft_ok:
        st.warning(
            "⚠️ The `peft` package is required for LoRA training. "
            "Install it with:\n```\npip install peft>=0.7.0 accelerate>=0.30.0\n```",
            icon="⚠️",
        )

    with st.container():
        lora_ref_images = st.file_uploader(
            "Upload 10–20 character reference images (PNG / JPG)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="lora_ref_images",
        )
        _effective_images = list(lora_ref_images) if lora_ref_images else []
        if _effective_images:
            st.success(f"📸 **{len(_effective_images)} image(s)** ready for training.")

        lora_col1, lora_col2 = st.columns(2)
        with lora_col1:
            lora_name = st.text_input(
                "LoRA name (used as folder name in `loras/`)",
                value="my_character",
                key="lora_name",
            )
            lora_trigger = st.text_input(
                "Instance / trigger word",
                value="mychar",
                key="lora_trigger_word",
                help="Short unique token prepended to prompts during generation, e.g. 'chihiro_kb'",
            )
            lora_base_model_choice = st.selectbox(
                "Base model (must match what you use in Panel Remix)",
                [
                    "Local GPU - CounterfeitXL (anime SDXL)",
                    "Local GPU - Anything V5 (anime SD1.5)",
                    "Local GPU - SSD-1B (fast SDXL)",
                    "Local GPU - Custom Hugging Face model",
                ],
                key="lora_base_model_choice",
            )
            if lora_base_model_choice == "Local GPU - CounterfeitXL (anime SDXL)":
                _lora_base_id = "gsdf/CounterfeitXL"
                _lora_is_sdxl = True
            elif lora_base_model_choice == "Local GPU - Anything V5 (anime SD1.5)":
                _lora_base_id = "stablediffusionapi/anything-v5"
                _lora_is_sdxl = False
            elif lora_base_model_choice == "Local GPU - SSD-1B (fast SDXL)":
                _lora_base_id = "segmind/SSD-1B"
                _lora_is_sdxl = True
            else:
                _lora_base_id = st.text_input(
                    "Custom HF model ID", value="segmind/SSD-1B", key="lora_custom_base_id"
                )
                _lora_is_sdxl = st.checkbox("Is SDXL-based?", value=True, key="lora_custom_is_sdxl")

        with lora_col2:
            lora_steps = st.slider(
                "Training steps", min_value=100, max_value=1500, value=400, step=50,
                key="lora_steps",
                help="400–600 steps is a good default for 15 reference images.",
            )
            lora_lr = st.select_slider(
                "Learning rate",
                options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                value=1e-4,
                format_func=lambda v: f"{v:.0e}",
                key="lora_lr",
            )
            lora_rank = st.select_slider(
                "LoRA rank",
                options=[2, 4, 8, 16],
                value=4,
                key="lora_rank",
                help="Higher rank = more capacity but more VRAM. Rank 4 works well for most cases.",
            )

    # ── Status display ────────────────────────────────────────────────────────
    _lora_status_file = os.path.join(TEMP_DIR, f"lora_status_{lora_name}.json")
    _lora_images_dir = os.path.join(TEMP_DIR, f"lora_images_{lora_name}")
    _lora_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras", lora_name)

    _status_placeholder = st.empty()

    def _read_lora_status():
        try:
            with open(_lora_status_file) as _f:
                return json.load(_f)
        except Exception:
            return None

    def _render_status(status_data):
        if not status_data:
            return
        s = status_data.get("status", "")
        msg = status_data.get("message", "")
        step = status_data.get("step", 0)
        total = status_data.get("total_steps", 1)
        loss = status_data.get("loss")
        if s == "completed":
            _status_placeholder.success(f"✅ {msg}")
        elif s == "error":
            _status_placeholder.error(f"❌ Training error: {msg}")
        elif s in ("training", "initializing"):
            pct = step / max(total, 1)
            _status_placeholder.info(f"⚙️ {msg}")
            st.progress(pct, text=f"Step {step}/{total}" + (f" | loss {loss:.4f}" if loss else ""))

    # Restore status from any ongoing run
    _current_status = _read_lora_status()
    _render_status(_current_status)

    # Check if a training process is already running
    _proc = st.session_state.get("lora_training_process")
    _is_running = _proc is not None and _proc.poll() is None

    lora_btn_col, lora_refresh_col, lora_stop_col = st.columns([3, 1, 1])

    with lora_btn_col:
        _start_disabled = _is_running or not _peft_ok
        _btn_label = "⏳ Training in progress..." if _is_running else "🚀 Start Training"
        lora_start = st.button(_btn_label, disabled=_start_disabled, key="lora_start")

    with lora_refresh_col:
        lora_refresh = st.button("🔄 Refresh", key="lora_refresh")

    with lora_stop_col:
        lora_stop = st.button("⏹ Stop", disabled=not _is_running, key="lora_stop")

    if lora_refresh:
        st.rerun()

    if lora_stop and _is_running:
        _proc.terminate()
        st.session_state["lora_training_process"] = None
        st.warning("Training stopped.")
        st.rerun()

    if lora_start:
        if not _effective_images:
            st.error("No images found. Upload panels here or use panels from the Panel Remix tab.")
        elif not lora_name.strip():
            st.error("Please enter a LoRA name.")
        elif not lora_trigger.strip():
            st.error("Please enter a trigger / instance word.")
        else:
            # Save all effective images to temp dir
            os.makedirs(_lora_images_dir, exist_ok=True)
            for _uf in _effective_images:
                _dest = os.path.join(_lora_images_dir, _uf.name)
                with open(_dest, "wb") as _fh:
                    _fh.write(_uf.getbuffer())

            # Build subprocess command
            _train_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "core", "train_lora.py"
            )
            _cmd = [
                sys.executable, _train_script,
                "--images_dir", _lora_images_dir,
                "--output_dir", _lora_output_dir,
                "--instance_prompt", f"{lora_trigger.strip()} person",
                "--base_model", _lora_base_id,
                "--max_steps", str(lora_steps),
                "--lr", str(lora_lr),
                "--lora_rank", str(lora_rank),
                "--status_file", _lora_status_file,
            ]
            if _lora_is_sdxl:
                _cmd.append("--is_sdxl")

            # Write initial status so the Refresh poll works immediately
            with open(_lora_status_file, "w") as _sf:
                json.dump({"status": "initializing", "step": 0,
                           "total_steps": lora_steps, "message": "Launching training process..."}, _sf)

            _proc = subprocess.Popen(
                _cmd,
                stdout=open(os.path.join(TEMP_DIR, f"lora_log_{lora_name}.txt"), "w"),
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            st.session_state["lora_training_process"] = _proc
            st.success(
                f"Training started in background (PID {_proc.pid}). "
                "Click **Refresh** to poll progress."
            )
            st.rerun()

    # ── Trained LoRA library ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Trained LoRA Library")
    _loras_dir_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
    _trained = [
        d for d in Path(_loras_dir_lib).iterdir()
        if d.is_dir() and (d / "pytorch_lora_weights.safetensors").exists()
    ] if os.path.isdir(_loras_dir_lib) else []
    if _trained:
        for _ld in sorted(_trained):
            _sz = (Path(_ld) / "pytorch_lora_weights.safetensors").stat().st_size / 1024 / 1024
            st.markdown(f"- **{_ld.name}** — `{_sz:.1f} MB`")
    else:
        st.info("No trained LoRAs yet. Upload reference images and start training above.")