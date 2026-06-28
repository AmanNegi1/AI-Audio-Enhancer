"""Unified TTS backend for auto-voiceover generation.

Supports gTTS (free), OpenAI TTS (tts-1-hd), and Bark (local GPU).
Language is NOT coupled to the backend — pass text in any language/script.
"""
import os

OPENAI_VOICES = ["onyx", "echo", "alloy", "fable", "nova", "shimmer"]
BARK_PRESETS = [
    "v2/en_speaker_6",
    "v2/en_speaker_0",
    "v2/en_speaker_3",
    "v2/en_speaker_9",
]


def generate_tts(text, output_path, backend="gtts", voice=None, openai_key=None,
                 progress_callback=None):
    """
    Generates a voiceover audio file.

    Args:
        text:              Narration text (any language/script).
        output_path:       Desired output file path (.mp3 preferred).
        backend:           "gtts" | "openai" | "bark"
        voice:
            - gtts:        ignored
            - openai:      voice name e.g. "onyx"
            - bark:        speaker preset e.g. "v2/en_speaker_6"
        openai_key:        Required when backend == "openai".
        progress_callback: Optional callable(current, total, chunk_text) for Bark
                           chunk-by-chunk progress reporting.

    Returns:
        Actual path of the generated file (may be .wav if Bark is used without pydub).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if backend == "openai":
        _generate_openai_tts(text, output_path, voice or "onyx", openai_key)
        return output_path
    elif backend == "bark":
        return _generate_bark_tts(text, output_path, voice or "v2/en_speaker_6",
                                   progress_callback=progress_callback)
    else:
        _generate_gtts(text, output_path)
        return output_path


def _generate_gtts(text, output_path):
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise ImportError("Install gtts: pip install gtts") from exc
    gTTS(text=text, lang="en", tld="co.in").save(output_path)


def _generate_openai_tts(text, output_path, voice, api_key):
    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API key is required for OpenAI TTS.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai>=1.0") from exc
    client = OpenAI(api_key=api_key.strip())
    response = client.audio.speech.create(model="tts-1-hd", voice=voice, input=text)
    with open(output_path, "wb") as f:
        f.write(response.content)


def _split_sentences(text, max_chars=200):
    """
    Splits text into chunks of at most max_chars characters, breaking at sentence
    boundaries (. ! ?) or at commas/newlines when no sentence boundary is found.
    Bark generates ~13s per call and silently truncates beyond ~200 chars.
    """
    import re
    # Normalise whitespace / newlines
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        # If a single sentence is too long, split further at commas
        if len(sentence) > max_chars:
            sub_parts = re.split(r',\s+', sentence)
            for part in sub_parts:
                if len(current) + len(part) + 2 <= max_chars:
                    current = (current + " " + part).strip() if current else part
                else:
                    if current:
                        chunks.append(current)
                    # If part itself still too long, hard-split
                    while len(part) > max_chars:
                        chunks.append(part[:max_chars])
                        part = part[max_chars:]
                    current = part
        else:
            if len(current) + len(sentence) + 1 <= max_chars:
                current = (current + " " + sentence).strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


_TORCH_LOAD_PATCHED = False

def _patch_torch_load_once():
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    import torch
    if hasattr(torch, "load"):
        _orig_load = torch.load
        def _safe_load(*args, **kwargs):
            try:
                return _orig_load(*args, **kwargs)
            except Exception:
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                    return _orig_load(*args, **kwargs)
                raise
        torch.load = _safe_load
    _TORCH_LOAD_PATCHED = True


def _generate_bark_tts(text, output_path, voice_preset, progress_callback=None):
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
    except ImportError as exc:
        raise ImportError(
            "Install Bark: pip install git+https://github.com/suno-ai/bark.git scipy"
        ) from exc
    import tempfile
    try:
        import scipy.io.wavfile as wav_io
    except ImportError as exc:
        raise ImportError("Install scipy: pip install scipy") from exc

    import numpy as np
    _patch_torch_load_once()
    preload_models()

    # Bark truncates at ~200 chars — split into sentence chunks and concatenate
    chunks = _split_sentences(text, max_chars=200)
    total = len(chunks)
    audio_parts = []
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, total, chunk)
        part = generate_audio(chunk, history_prompt=voice_preset)
        audio_parts.append(part)
    if progress_callback:
        progress_callback(total, total, "")

    audio_array = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    wav_io.write(tmp_wav, SAMPLE_RATE, audio_array)

    # Prefer MP3 output via pydub; fall back to WAV if not available
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(tmp_wav).export(output_path, format="mp3")
        os.remove(tmp_wav)
        return output_path
    except Exception:
        wav_path = os.path.splitext(output_path)[0] + ".wav"
        import shutil
        shutil.move(tmp_wav, wav_path)
        return wav_path