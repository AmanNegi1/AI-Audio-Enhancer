from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import subprocess
import numpy as np
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

    # 6. Normalise & save
    enhanced = enhanced.cpu()
    enhanced = enhanced / (torch.max(torch.abs(enhanced)) + 1e-8)
    enhanced_np = enhanced.numpy().T  # → (frames, channels)
    sf.write(output_path, enhanced_np, model.sample_rate)

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
