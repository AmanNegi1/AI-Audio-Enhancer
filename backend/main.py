from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import subprocess
import scipy.signal as signal
import numpy as np
import noisereduce as nr
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, HighShelfFilter, HighpassFilter, Limiter


import torch
import soundfile as sf
import imageio_ffmpeg
from denoiser import pretrained
from denoiser.dsp import convert_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load AI model — uses GPU if available
print("Loading AI denoiser model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pretrained.dns64().to(device)
model.eval()
print(f"Model ready on {device}!")


def convert_to_wav(input_path: str, output_path: str):
    """Use bundled ffmpeg to convert any audio format to 16-bit PCM WAV."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", input_path, "-ar", "48000", "-ac", "1",
         "-sample_fmt", "s16", output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


@app.post("/api/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    wav_path   = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    output_path = os.path.join(PROCESSED_DIR, f"{file_id}_enhanced.wav")

    # 1. Save upload
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Convert to WAV via ffmpeg (handles m4a, mp3, ogg, flac, etc.)
    convert_to_wav(input_path, wav_path)

    # 3. Read WAV with soundfile — no torchaudio needed
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    # data shape: (frames, channels) → we want (channels, frames)
    wav_tensor = torch.tensor(data.T, dtype=torch.float32)

    # 4. Resample / channel-match for model
    wav_tensor = convert_audio(wav_tensor.to(device), sr, model.sample_rate, model.chin)

    # 5. Run AI denoiser
    with torch.no_grad():
        enhanced = model(wav_tensor.unsqueeze(0))[0]

    # 6. Post‑process: AI denoiser leaves some artifacts, so we apply noisereduce + mastering
    enhanced_np = enhanced.cpu().numpy().T  # (frames, channels)

    # Apply spectral noise reduction to clean up remaining background hiss
    # noisereduce expects (channels, frames) so we transpose, process, then transpose back
    cleaned_np = nr.reduce_noise(y=enhanced_np.T, sr=model.sample_rate, prop_decrease=0.7).T

    # 7. Apply Studio Mastering Chain with Pedalboard
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),            # Remove low-end rumble
        NoiseGate(threshold_db=-45.0, ratio=1.5, release_ms=250), # Clean up silence
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0), # Add warmth
        HighShelfFilter(cutoff_frequency_hz=4000, gain_db=3.0), # Add presence/clarity
        Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=5.0, release_ms=50.0), # Smooth dynamics
        Limiter(threshold_db=-1.0) # Prevent clipping
    ])
    
    # pedalboard also expects (channels, frames)
    mastered_audio = board(cleaned_np.T, model.sample_rate).T
    
    # 8. Save final output
    sf.write(output_path, mastered_audio, model.sample_rate)


    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"enhanced_{os.path.splitext(file.filename)[0]}.wav"
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


# Serve the frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
