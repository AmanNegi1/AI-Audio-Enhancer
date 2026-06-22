import whisper
import torch
import numpy as np

def align_audio_segments(audio_path, scenes):
    """
    Transcribes the audio using a local Whisper model on GPU (if CUDA available).
    Aligns the timeline of Whisper segments with the input script scenes.
    Returns the scenes list updated with 'start' and 'end' timestamp keys.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper base model
    model = whisper.load_model("base", device=device)
    
    # Transcribe audio file
    result = model.transcribe(audio_path)
    whisper_segments = result.get("segments", [])
    
    if not whisper_segments:
        # Fallback: distribute evenly across the total length of audio
        # If we can't load audio length, estimate 10 seconds per scene
        total_duration = 60.0
        step = total_duration / len(scenes)
        for i, s in enumerate(scenes):
            s['start'] = i * step
            s['end'] = (i + 1) * step
        return scenes

    # Map scenes to whisper segment timestamps based on text similarity
    # A simple and robust method is to match based on cumulative word index
    # We reconstruct the transcript from both sides and align them proportionally.
    total_audio_duration = whisper_segments[-1]['end'] if whisper_segments else 60.0
    
    # Calculate word counts for scenes
    scene_word_counts = [len(s['text_segment'].split()) for s in scenes]
    total_scene_words = sum(scene_word_counts) if scene_word_counts else 1
    
    # Distribute timestamps based on word counts as a baseline
    current_time = 0.0
    for i, scene in enumerate(scenes):
        w_count = scene_word_counts[i]
        duration = (w_count / total_scene_words) * total_audio_duration
        scene['start'] = round(current_time, 2)
        scene['end'] = round(current_time + duration, 2)
        current_time += duration
        
    # Attempt to refine timestamps using actual whisper segments
    # Match each whisper segment to the closest scene based on string overlap
    try:
        scene_transcripts = [s['text_segment'].lower() for s in scenes]
        
        # Initialize boundaries
        scene_starts = [None] * len(scenes)
        scene_ends = [None] * len(scenes)
        
        for seg in whisper_segments:
            seg_text = seg['text'].lower()
            # Find closest matching scene text segment
            best_idx = 0
            best_overlap = 0
            for idx, st in enumerate(scene_transcripts):
                # Calculate overlap score
                words_seg = set(seg_text.split())
                words_scene = set(st.split())
                overlap = len(words_seg.intersection(words_scene))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            
            # Record bounds
            if scene_starts[best_idx] is None or seg['start'] < scene_starts[best_idx]:
                scene_starts[best_idx] = seg['start']
            if scene_ends[best_idx] is None or seg['end'] > scene_ends[best_idx]:
                scene_ends[best_idx] = seg['end']
                
        # Fill in refined bounds, maintaining sequence order
        for idx in range(len(scenes)):
            if scene_starts[idx] is not None:
                scenes[idx]['start'] = round(scene_starts[idx], 2)
            if scene_ends[idx] is not None:
                scenes[idx]['end'] = round(scene_ends[idx], 2)
                
            # Guarantee start < end and sequence doesn't overlap backwards
            if idx > 0 and scenes[idx]['start'] < scenes[idx-1]['end']:
                scenes[idx]['start'] = scenes[idx-1]['end']
            if scenes[idx]['end'] <= scenes[idx]['start']:
                scenes[idx]['end'] = scenes[idx]['start'] + 2.0
                
    except Exception as e:
        # If refinement fails, keep the proportional baseline
        pass
        
    return scenes
