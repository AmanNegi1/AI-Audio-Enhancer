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

import streamlit as st
import json
import torch
import shutil
from PIL import Image

# Import core modules
from core.parser import parse_script_to_prompts
from core.aligner import align_audio_segments
from core.generator import generate_scene_image
from core.custom_generator import generate_custom_image
from core.assembler import assemble_video

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
    gemini_key = st.text_input(
        "Gemini API Key", 
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free API key from Google AI Studio (aistudio.google.com)"
    )
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

tab_recap, tab_txt2img = st.tabs(["🎬 Manga Video Recap Generator", "🎨 Text to Image"])

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
                "Replace Demo Voiceover with custom Audio (.mp3, .wav, .m4a)", 
                type=["mp3", "wav", "m4a"]
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
                "Upload Voiceover Audio (.mp3, .wav, .m4a)", 
                type=["mp3", "wav", "m4a"]
            )
        
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
            elif not is_demo and not audio_file:
                st.error("Please upload a voiceover audio file.")
            elif not script_text.strip():
                st.error("Please paste your script.")
            else:
                st.subheader("⚡ Generation Pipeline")
                
                # Setup audio path
                if is_demo:
                    audio_path = os.path.join(TEMP_DIR, "demo_voiceover.mp3")
                    # Generate voiceover using gTTS dynamically
                    try:
                        from gtts import gTTS
                        st.write("🔊 **Synthesizing demo voiceover using gTTS...**")
                        tts = gTTS(text=script_text, lang='en')
                        tts.save(audio_path)
                        st.info("Demo voiceover synthesized successfully.")
                    except Exception as e:
                        st.error(f"Error generating demo audio with gTTS: {e}")
                        st.stop()
                else:
                    # Save uploaded audio file to temp directory
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
                    st.write("🎨 **Step 3:** Generating original scenes locally on GPU (RTX 4060)...")
                    progress_bar = st.progress(0.0)
                    
                    try:
                        for idx, scene in enumerate(scenes):
                            # Generate unique filename
                            image_filename = f"scene_{idx}_output.png"
                            image_path = os.path.join(TEMP_DIR, image_filename)
                            
                            prompt = scene['image_prompt']
                            st.write(f"🖌️ *Generating Scene #{idx+1} prompt:* `{prompt[:90]}...`")
                            
                            # Call local generator
                            img = generate_scene_image(prompt, style_preset, mock_mode=draft_mode)
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

with tab_txt2img:
    st.header("🎨 Standalone Text to Image Generator")
    st.write("Generate high-quality artwork using local open-source models like Stable Diffusion XL or Z-Image-Turbo.")
    
    col_t2i_in, col_t2i_res = st.columns([1, 1])
    
    with col_t2i_in:
        st.markdown("<div class='status-card'>", unsafe_allow_html=True)
        st.subheader("📝 Inputs & Configuration")
        
        t2i_model_choice = st.selectbox(
            "Select AI Image Model",
            options=[
                "segmind/SSD-1B (Fast, cached, 1.3B parameters)",
                "Tongyi-MAI/Z-Image-Turbo (Distilled 6B parameter S3-DiT)",
                "stabilityai/sdxl-turbo (Instant 1-step SDXL)",
                "Custom Model ID (Hugging Face Repository Path)"
            ],
            index=0
        )
        
        if "Custom Model" in t2i_model_choice:
            t2i_model_id = st.text_input(
                "Enter Hugging Face Model ID",
                value="stabilityai/stable-diffusion-xl-base-1.0",
                placeholder="e.g., stabilityai/stable-diffusion-xl-base-1.0"
            )
        elif "SSD-1B" in t2i_model_choice:
            t2i_model_id = "segmind/SSD-1B"
        elif "Z-Image-Turbo" in t2i_model_choice:
            t2i_model_id = "Tongyi-MAI/Z-Image-Turbo"
            st.warning("⚠️ **Z-Image-Turbo is a 6B parameter model (~12 GB of weights)**. Loading it requires at least 20-24 GB of system RAM to prevent process crashes. If your laptop has 16 GB of RAM, this will likely crash the local server.")
        else:
            t2i_model_id = "stabilityai/sdxl-turbo"
            
        is_turbo = "turbo" in t2i_model_id.lower() or "Turbo" in t2i_model_id
        is_z_image = "Z-Image" in t2i_model_id or "Tongyi-MAI" in t2i_model_id
        
        default_steps = 8 if is_turbo else 20
        default_guidance = 0.0 if is_z_image else (1.5 if is_turbo else 7.5)
        
        t2i_prompt = st.text_area(
            "Prompt / Image Description",
            value="anime style, portrait of a warrior boy with spiky hair, wielding a glowing katana sword, dramatic lighting, highly detailed",
            height=120
        )
        
        t2i_negative = st.text_area(
            "Negative Prompt",
            value="deformed, blurry, text, watermark, low quality, bad anatomy",
            disabled=is_turbo,
            height=80
        )
        
        col_t2i_params1, col_t2i_params2 = st.columns(2)
        with col_t2i_params1:
            t2i_steps = st.slider("Inference Steps", min_value=1, max_value=50, value=default_steps)
        with col_t2i_params2:
            t2i_guidance = st.slider("Guidance Scale (CFG)", min_value=0.0, max_value=15.0, value=default_guidance, step=0.5)
            
        t2i_aspect = st.selectbox(
            "Aspect Ratio",
            options=["16:9 (Landscape) - 1024x576", "1:1 (Square) - 1024x1024", "9:16 (Portrait) - 576x1024"],
            index=0
        )
        
        if "16:9" in t2i_aspect:
            t2i_w, t2i_h = 1024, 576
        elif "1:1" in t2i_aspect:
            t2i_w, t2i_h = 1024, 1024
        else:
            t2i_w, t2i_h = 576, 1024
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        t2i_generate_btn = st.button("Generate Single Image 🎨")
        
    with col_t2i_res:
        if t2i_generate_btn:
            if not t2i_prompt.strip():
                st.error("Please enter an image prompt description.")
            else:
                st.subheader("⚡ Generation Progress")
                
                active_hf_token = hf_token_main if hf_token_main else hf_token
                if active_hf_token.strip():
                    os.environ["HF_TOKEN"] = active_hf_token.strip()
                
                try:
                    with st.status("🎨 Generating your image locally...", expanded=True) as status:
                        st.write("Loading model pipeline into memory (applying VRAM safety optimizations)...")
                        img = generate_custom_image(
                            prompt=t2i_prompt,
                            model_id=t2i_model_id,
                            negative_prompt=t2i_negative if not is_turbo else "",
                            num_inference_steps=t2i_steps,
                            guidance_scale=t2i_guidance,
                            width=t2i_w,
                            height=t2i_h
                        )
                        status.update(label="🎨 Generation Succeeded!", state="complete")
                    
                    st.subheader("🎉 Generated Image Output")
                    st.image(img, use_container_width=True)
                    
                    t2i_temp_path = os.path.join(TEMP_DIR, "t2i_output.png")
                    img.save(t2i_temp_path)
                    
                    with open(t2i_temp_path, "rb") as f:
                        st.download_button(
                            label="Download PNG Image 📥",
                            data=f,
                            file_name="generated_art.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
        else:
            st.info("Configure your prompt on the left, then click 'Generate Single Image' to run.")
