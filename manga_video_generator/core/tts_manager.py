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


def generate_tts(text, output_path, backend="gtts", voice=None, openai_key=None):
    """
    Generates a voiceover audio file.

    Args:
        text:        Narration text (any language/script).
        output_path: Desired output file path (.mp3 preferred).
        backend:     "gtts" | "openai" | "bark"
        voice:
            - gtts:   ignored (gTTS auto-handles the text encoding)
            - openai: voice name e.g. "onyx"
            - bark:   speaker preset e.g. "v2/en_speaker_6"
        openai_key: Required when backend == "openai".

    Returns:
        Actual path of the generated file (may be .wav if Bark is used without pydub).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if backend == "openai":
        _generate_openai_tts(text, output_path, voice or "onyx", openai_key)
        return output_path
    elif backend == "bark":
        return _generate_bark_tts(text, output_path, voice or "v2/en_speaker_6")
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


def _generate_bark_tts(text, output_path, voice_preset):
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

    preload_models()
    audio_array = generate_audio(text, history_prompt=voice_preset)

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