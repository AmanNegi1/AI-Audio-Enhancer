import whisper
import torch
import numpy as np
import re
from difflib import SequenceMatcher

def clean_word(w):
    """
    Normalizes a word for matching by removing punctuation and converting to lowercase.
    """
    return re.sub(r'[^\w\s]', '', w).lower().strip()

def align_audio_segments(audio_path, scenes):
    """
    Transcribes the audio using a local Whisper model on GPU (if CUDA available).
    Aligns the timeline of Whisper segments with the input script scenes monotonically
    using difflib.SequenceMatcher.
    
    Returns the scenes list updated with correct 'start' and 'end' timestamp keys.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper base model
    model = whisper.load_model("base", device=device)
    
    # Transcribe audio file with word-level timestamps for accurate per-word timing
    result = model.transcribe(audio_path, word_timestamps=True)
    whisper_segments = result.get("segments", [])
    
    # Calculate total duration of audio
    if whisper_segments:
        total_audio_duration = whisper_segments[-1]['end']
    else:
        total_audio_duration = 60.0 # Default fallback
        
    # 1. Generate baseline proportional timestamps (in case alignment fails)
    scene_word_counts = [len(s['text_segment'].split()) for s in scenes]
    total_scene_words = sum(scene_word_counts) if scene_word_counts else 1
    
    current_time = 0.0
    for i, scene in enumerate(scenes):
        w_count = scene_word_counts[i]
        duration = (w_count / total_scene_words) * total_audio_duration
        scene['start'] = round(current_time, 2)
        scene['end'] = round(current_time + duration, 2)
        current_time += duration

    if not whisper_segments:
        return scenes
        
    try:
        # 2. Extract words with timestamps from Whisper segments
        transcribed_words = []
        transcribed_word_times = []
        
        for seg in whisper_segments:
            words_data = seg.get('words', [])
            if words_data:
                # Use Whisper's DTW-aligned per-word timestamps (accurate)
                for wd in words_data:
                    cleaned = clean_word(wd.get('word', ''))
                    if cleaned:
                        transcribed_words.append(cleaned)
                        transcribed_word_times.append(wd['start'])
            else:
                # Fallback: distribute words evenly within segment
                seg_text = seg['text']
                seg_start = seg['start']
                seg_end = seg['end']
                words = seg_text.split()
                if not words:
                    continue
                duration = seg_end - seg_start
                step = duration / len(words)
                for j, w in enumerate(words):
                    cleaned = clean_word(w)
                    if cleaned:
                        transcribed_words.append(cleaned)
                        transcribed_word_times.append(seg_start + j * step)
                    
        # 3. Extract words with scene indices from scenes
        script_words = []
        script_word_scene_indices = []
        
        for idx, scene in enumerate(scenes):
            words = scene['text_segment'].split()
            for w in words:
                cleaned = clean_word(w)
                if cleaned:
                    script_words.append(cleaned)
                    script_word_scene_indices.append(idx)
                    
        if not transcribed_words or not script_words:
            return scenes
            
        # 4. Use SequenceMatcher to find matching blocks of words (retains relative order)
        matcher = SequenceMatcher(None, transcribed_words, script_words)
        matching_blocks = matcher.get_matching_blocks()
        
        # Collect matched timestamps for each scene
        scene_matched_times = {i: [] for i in range(len(scenes))}
        for block in matching_blocks:
            t_idx, s_idx, length = block.a, block.b, block.size
            for k in range(length):
                t_word_idx = t_idx + k
                s_word_idx = s_idx + k
                scene_idx = script_word_scene_indices[s_word_idx]
                word_time = transcribed_word_times[t_word_idx]
                scene_matched_times[scene_idx].append(word_time)
                
        # 5. Determine scene start/end boundaries based on matches
        # We define boundaries b_0, b_1, ..., b_M where Scene i spans [b_i, b_{i+1}]
        # b_0 = 0.0, b_M = total_audio_duration
        M = len(scenes)
        boundaries = [0.0] * (M + 1)
        boundaries[0] = 0.0
        boundaries[M] = total_audio_duration
        
        # Fill intermediate boundaries
        for i in range(1, M):
            times_prev = scene_matched_times[i-1]
            times_curr = scene_matched_times[i]
            default_b = scenes[i-1]['end'] # Proportional baseline
            
            if times_prev and times_curr:
                last_prev = max(times_prev)
                first_curr = min(times_curr)
                if last_prev < first_curr:
                    # Normal: split at midpoint between end of prev and start of current
                    boundaries[i] = (last_prev + first_curr) / 2.0
                else:
                    # Timestamps cross — a common word in scene i-1 was matched to a
                    # late transcript position, inflating scene i-1's duration.
                    # Fall back to the proportional baseline.
                    boundaries[i] = default_b
            elif times_prev:
                boundaries[i] = max(times_prev)
            elif times_curr:
                boundaries[i] = min(times_curr)
            else:
                boundaries[i] = default_b
                
        # Enforce strict monotonicity: b_0 <= b_1 <= b_2 <= ... <= b_M
        # Forward pass to enforce lower bounds
        for i in range(1, M):
            if boundaries[i] < boundaries[i-1]:
                boundaries[i] = boundaries[i-1]
                
        # Backward pass to enforce upper bounds
        for i in range(M-1, 0, -1):
            if boundaries[i] > boundaries[i+1]:
                boundaries[i] = boundaries[i+1]
                
        # Apply boundaries to scenes
        for i in range(M):
            scenes[i]['start'] = round(boundaries[i], 2)
            scenes[i]['end'] = round(boundaries[i+1], 2)
            
    except Exception as e:
        print(f"[ALIGNER ERROR] Sequence alignment failed: {e}. Falling back to baseline.")
        
    return scenes