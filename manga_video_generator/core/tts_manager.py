"""
Unified TTS backend for voiceover generation.
Supports gTTS (free), OpenAI TTS (tts-1-hd), and Bark (local GPU, very expressive).
"""
import os
import tempfile

OPENAI_VOICES = ["onyx", "echo", "alloy", "fable", "nova", "shimmer"]
BARK_PRESETS = [
    "v2/en_speaker_6",
    "v2/en_speaker_0", 
    "v2/en_speaker_3",
    "v2/en_speaker_9",
]


def generate_tts(text, output_path, backend="gtts", voice=None, openai_key=None, use_streaming=False):
    """
    Generates voiceover audio.

    Args:
        text:           Narration text (any language for OpenAI; English for gTTS/Bark).
        output_path:    Desired output file path (.mp3 preferred).
        backend:        "gtts" | "openai" | "bark"
        voice:
            - gtts:     ignored
            - openai:   voice name e.g. "onyx" (deep), "shimmer" (high), "nova" (upbeat)
            - bark:     speaker preset e.g. "v2/en_speaker_6"
        openai_key:     Required when backend == "openai".
        use_streaming:  For Bark with very long texts, process in chunks (slower but more reliable).

    Returns:
        Actual path of generated file (may differ for Bark .wav fallback).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if backend == "openai":
        return _generate_openai_tts(text, output_path, voice or "onyx", openai_key)
    elif backend == "bark":
        return _generate_bark_tts(text, output_path, voice or "v2/en_speaker_6", use_streaming=use_streaming)
    else:  # gtts
        return _generate_gtts(text, output_path)


def _generate_gtts(text, output_path):
    """Generate using Google TTS (free, basic quality, English only)."""
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise ImportError("Install gtts: pip install gtts") from exc
    
    try:
        gTTS(text=text, lang="en", tld="co.in").save(output_path)
        return output_path
    except Exception as e:
        raise RuntimeError(f"gTTS failed: {e}") from e


def _generate_openai_tts(text, output_path, voice, api_key):
    """Generate using OpenAI TTS API (tts-1-hd, high quality, multilingual)."""
    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API key required for OpenAI TTS")
    
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai>=1.0") from exc
    
    try:
        client = OpenAI(api_key=api_key.strip())
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text
        )
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    except Exception as e:
        raise RuntimeError(f"OpenAI TTS failed: {e}") from e


def _generate_bark_tts(text, output_path, voice_preset, use_streaming=False):
    """
    Generate using Bark (local GPU, very expressive, ~30-60s per sentence).
    
    For long texts, use use_streaming=True to process in chunks (slower but prevents timeout).
    """
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
    except ImportError as exc:
        raise ImportError(
            "Install Bark: pip install git+https://github.com/suno-ai/bark.git scipy pydub"
        ) from exc
    
    try:
        import scipy.io.wavfile as wav_io
    except ImportError as exc:
        raise ImportError("Install scipy: pip install scipy") from exc
    
    try:
        preload_models()
        
        if use_streaming and len(text.split()) > 50:
            # Process long texts sentence-by-sentence to avoid timeouts
            sentences = text.split('. ')
            audio_arrays = []
            
            for i, sent in enumerate(sentences):
                if not sent.strip():
                    continue
                print(f"[Bark] Processing sentence {i+1}/{len(sentences)}: {sent[:50]}...")
                audio_array = generate_audio(sent.strip() + '.', history_prompt=voice_preset)
                audio_arrays.append(audio_array)
            
            # Concatenate all arrays
            if audio_arrays:
                import numpy as np
                audio_array = np.concatenate(audio_arrays)
            else:
                raise ValueError("No audio generated from text")
        else:
            # Single generation for shorter texts
            audio_array = generate_audio(text, history_prompt=voice_preset)
        
        # Write to temporary WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        wav_io.write(tmp_wav, SAMPLE_RATE, audio_array)
        
        # Try to export to MP3 via pydub; fall back to WAV
        try:
            from pydub import AudioSegment
            AudioSegment.from_wav(tmp_wav).export(output_path, format="mp3", bitrate="192k")
            os.remove(tmp_wav)
            return output_path
        except Exception as e:
            print(f"[Bark] MP3 export failed ({e}), using WAV fallback")
            wav_path = os.path.splitext(output_path)[0] + ".wav"
            import shutil
            shutil.move(tmp_wav, wav_path)
            return wav_path
    
    except Exception as e:
        raise RuntimeError(f"Bark TTS failed: {e}") from e