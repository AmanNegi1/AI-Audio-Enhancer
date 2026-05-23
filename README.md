# Voice Enhancer 🚀

A **studio‑quality voice enhancement** web app built with:
- **FastAPI** backend (Python) running a denoising model (`denoiser`).
- A premium, dark‑mode, glass‑morphism frontend (HTML/CSS/JS) with drag‑and‑drop upload and waveform visualiser.
- Automatic audio conversion (FFmpeg via `imageio‑ffmpeg`) so you can upload MP3, M4A, OGG, etc.

---

## 📂 Project layout
```
Voice Enhancer/
├─ backend/        # FastAPI server & model code
│   ├─ main.py
│   ├─ venv/       # virtual environment (not committed)
│   └─ requirements.txt (generated from pip list)
├─ frontend/       # static HTML/CSS/JS UI
│   ├─ index.html
│   ├─ style.css
│   └─ script.js
└─ README.md      # <‑‑ you are reading this!
```

---

## 🛠️ Prerequisites
- **Windows 10/11** with **PowerShell**.
- **Python 3.13+** (the script uses the built‑in `venv` module).
- **Git** (only needed if you plan to push to a remote).
- No separate **Node.js** installation required – the frontend is pure static files.

---

## ▶️ Quick start (local development)
1. **Open PowerShell** and `cd` into the project root:
   ```powershell
   cd "D:\Voice Enhancer"
   ```
2. **Create and activate a virtual environment** (if one does not already exist):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # activate the env
   ```
3. **Install the Python dependencies** (the required packages are already listed in the environment, but you can reinstall for safety):
   ```powershell
   pip install -U pip setuptools
   pip install fastapi uvicorn denoiser imageio-ffmpeg soundfile scipy torch torchaudio
   ```
4. **Run the backend server**:
   ```powershell
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   You should see:
   ```
   INFO: Started server process [...]\nINFO: Application startup complete.
   INFO: Uvicorn running on http://0.0.0.0:8000
   ```
5. **Open the web UI** in any browser:
   - Navigate to `http://localhost:8000`.
   - Drag‑and‑drop an audio file (MP3, M4A, WAV, etc.).
   - The server will convert it to WAV, run the denoiser, and return an enhanced file ready for download.

---

## 📦 Production (optional)
If you want to expose the service publicly:
1. Use a reverse proxy (e.g., **NGINX** or **Apache**) to forward port 80/443 to `localhost:8000`.
2. Run the server with a process manager such as **Gunicorn** or **Windows Service** for resilience.
3. Ensure the machine has a GPU‑compatible PyTorch build (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) if you wish to leverage your RTX 4060.

---

## 🧹 Clean‑up
- To deactivate the virtual environment: `deactivate`.
- The `venv/` directory can be deleted safely if you need to rebuild it.

---

## 📚 Further reading
- **FastAPI docs**: https://fastapi.tiangolo.com/
- **denoiser library**: https://github.com/facebookresearch/denoiser
- **imageio‑ffmpeg** (bundled FFmpeg): https://github.com/imageio/imageio-ffmpeg

---

Happy coding and enjoy crystal‑clear audio!
