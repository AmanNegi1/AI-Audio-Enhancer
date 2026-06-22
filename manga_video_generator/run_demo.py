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
from gtts import gTTS
from core.aligner import align_audio_segments
from core.generator import generate_scene_image
from core.assembler import assemble_video

def run_demo():
    print("====================================================")
    print("AI Manga Video Generator - Running Local Demo")
    print("====================================================\n")
    
    # 1. Check GPU
    cuda_available = torch.cuda.is_available()
    print(f"[STATUS] CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"[STATUS] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] Running on CPU. This will be slow!")

    # Setup directories
    temp_dir = "temp"
    output_dir = "output"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Generate a voiceover file programmatically using gTTS
    audio_path = os.path.join(temp_dir, "demo_voiceover.mp3")
    script_text = (
        "Hello guys, welcome back to the channel. Today we are looking at Chihiro "
        "and the magical goldfish in Kagurabachi. Prepare for the fight."
    )
    
    print("\n[AUDIO] Step 1: Synthesizing demo voiceover using gTTS...")
    tts = gTTS(text=script_text, lang='en')
    tts.save(audio_path)
    print(f"[SUCCESS] Voiceover saved to: {audio_path}")
    
    # 3. Create mock scene breakdown (bypassing Gemini API for the test)
    scenes = [
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
    
    # 4. Run local Whisper to align timestamps
    print("\n[TIME] Step 2: Transcribing and aligning timeline using local Whisper...")
    aligned_scenes = align_audio_segments(audio_path, scenes)
    print("[SUCCESS] Aligned timeline details:")
    for idx, s in enumerate(aligned_scenes):
        print(f"  - Scene #{idx+1} [{s['start']}s -> {s['end']}s]: {s['text_segment']}")
        
    # 5. Generate images locally (Draft Mode by default)
    print("\n[IMAGE] Step 3: Generating draft images locally (Draft Mode)...")
    print("To render real GPU AI artwork, change mock_mode=False on line 73 of run_demo.py.\n")
    
    for idx, scene in enumerate(aligned_scenes):
        image_path = os.path.join(temp_dir, f"demo_scene_{idx}.png")
        print(f"[STAGE] Rendering Image {idx+1}/{len(aligned_scenes)}: '{scene['image_prompt'][:60]}...'")
        
        img = generate_scene_image(scene['image_prompt'], mock_mode=True)
        img.save(image_path)
        scene['image_path'] = image_path
        print(f"  |-- Saved image to: {image_path}")
        
    # 6. Stitch final video using MoviePy
    print("\n[VIDEO] Step 4: Compiling final video with MoviePy (applying panning transitions)...")
    output_video_path = os.path.join(output_dir, "demo_recap.mp4")
    
    assemble_video(aligned_scenes, audio_path, output_video_path)
    print(f"\n[SUCCESS] Aligned video generated successfully!")
    print(f"[OUTPUT] Video Location: {os.path.abspath(output_video_path)}")
    print("====================================================")

if __name__ == "__main__":
    run_demo()
