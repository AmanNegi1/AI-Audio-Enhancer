from moviepy import ImageClip, concatenate_videoclips, AudioFileClip, ColorClip, CompositeVideoClip, VideoClip
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_SIZE = (1920, 1080)

def ease_in_out(progress):
    progress = max(0.0, min(1.0, progress))
    return progress * progress * (3.0 - 2.0 * progress)

def create_keyframed_image_clip(image_path, duration, scene_index, motion="auto"):
    """
    Applies a keyframed Ken Burns move to every still image clip.
    Using a PIL transform function inside VideoClip to resolve the MoviePy absolute time bug.
    """
    # Load image using PIL
    img_base = Image.open(image_path).convert("RGB")
    
    # Resize base image to match VIDEO_SIZE
    img_base = img_base.resize(VIDEO_SIZE, Image.Resampling.LANCZOS)
    
    motion_paths = {
        "push_in": (1.04, 1.22, (0.50, 0.50), (0.50, 0.50)),
        "slow_push": (1.06, 1.16, (0.46, 0.44), (0.54, 0.52)),
        "pull_out": (1.22, 1.06, (0.50, 0.50), (0.50, 0.50)),
        "pan_left": (1.14, 1.18, (0.92, 0.45), (0.08, 0.55)),
        "pan_right": (1.14, 1.18, (0.08, 0.45), (0.92, 0.55)),
    }
    auto_paths = [
        (1.06, 1.22, (0.02, 0.08), (0.92, 0.78)),
        (1.22, 1.06, (0.96, 0.12), (0.10, 0.90)),
        (1.06, 1.22, (0.48, 0.02), (0.55, 0.96)),
        (1.22, 1.06, (0.05, 0.90), (0.95, 0.08)),
    ]
    start_zoom, end_zoom, start_focus, end_focus = motion_paths.get(
        motion,
        auto_paths[scene_index % len(auto_paths)]
    )

    def make_frame(t):
        progress = ease_in_out(t / max(duration, 0.001))
        zoom = start_zoom + ((end_zoom - start_zoom) * progress)
        
        # Calculate new size
        new_w = int(VIDEO_SIZE[0] * zoom)
        new_h = int(VIDEO_SIZE[1] * zoom)
        
        # Resize using PIL
        resized_img = img_base.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Calculate crop coordinates based on focus coordinates
        extra_w = new_w - VIDEO_SIZE[0]
        extra_h = new_h - VIDEO_SIZE[1]
        
        focus_x = start_focus[0] + ((end_focus[0] - start_focus[0]) * progress)
        focus_y = start_focus[1] + ((end_focus[1] - start_focus[1]) * progress)
        
        left = int(extra_w * focus_x)
        top = int(extra_h * focus_y)
        
        # Crop to VIDEO_SIZE
        cropped_img = resized_img.crop((left, top, left + VIDEO_SIZE[0], top + VIDEO_SIZE[1]))
        return np.array(cropped_img)

    return VideoClip(make_frame, duration=duration)

def create_caption_clip(text, duration):
    caption_height = 210
    image = Image.new("RGBA", (VIDEO_SIZE[0], caption_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arialbd.ttf", 54)
    except OSError:
        font = ImageFont.load_default()

    words = text.replace("\n", " ").split()
    wrapped_lines = []
    current_line = []
    for word in words:
        candidate = " ".join(current_line + [word])
        if draw.textlength(candidate, font=font) > VIDEO_SIZE[0] - 220 and current_line:
            wrapped_lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        wrapped_lines.append(" ".join(current_line))
    wrapped_lines = wrapped_lines[:2]

    line_height = 64
    block_height = line_height * len(wrapped_lines) + 38
    block_top = caption_height - block_height - 14
    draw.rounded_rectangle(
        [120, block_top, VIDEO_SIZE[0] - 120, caption_height - 12],
        radius=22,
        fill=(0, 0, 0, 178),
        outline=(255, 213, 74, 230),
        width=3,
    )
    y = block_top + 18
    for line in wrapped_lines:
        text_width = draw.textlength(line, font=font)
        x = (VIDEO_SIZE[0] - text_width) / 2
        draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0, 220))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height

    return ImageClip(np.array(image)).with_duration(duration).with_position((0, VIDEO_SIZE[1] - caption_height - 44))

def assemble_video(scenes, audio_path, output_path, add_captions=False):
    """
    Assembles the generated images into a single video, synchronized with the audio track.
    Applies a dynamic Ken Burns zoom-in animation to each clip, aligned to its exact start timestamp.
    """
    # 1. Determine active end time of the scenes to handle limited test runs gracefully
    active_ends = []
    found_count = 0
    missing_count = 0
    for scene in scenes:
        image_path = scene.get('image_path')
        if image_path and os.path.exists(image_path):
            active_ends.append(scene.get('end', 5.0))
            found_count += 1
        else:
            missing_count += 1
            
    max_scene_end = max(active_ends) if active_ends else 0.0
    
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration
    
    # Revert trimming logic: extend the last valid scene to fill the remaining duration of the video.
    last_valid_scene_idx = -1
    for i in range(len(scenes) - 1, -1, -1):
        scene = scenes[i]
        image_path = scene.get('image_path')
        if image_path and os.path.exists(image_path):
            last_valid_scene_idx = i
            break
            
    if last_valid_scene_idx != -1:
        orig_end = scenes[last_valid_scene_idx].get('end', 5.0)
        scenes[last_valid_scene_idx]['original_end'] = orig_end
        scenes[last_valid_scene_idx]['end'] = max(orig_end, total_duration)
        print(f"[ASSEMBLER - AUDIO FIRST] Extended last valid scene #{last_valid_scene_idx+1} from {orig_end:.2f}s to {total_duration:.2f}s")
        
    # Print status to terminal
    print(f"[ASSEMBLER - AUDIO FIRST] scenes count: {len(scenes)}, images found: {found_count}, missing: {missing_count}")
    print(f"[ASSEMBLER - AUDIO FIRST] max_scene_end before extension: {max_scene_end}s, audio.duration: {audio.duration}s -> total_duration: {total_duration}s")
    
    # Display status in Streamlit if running inside a Streamlit app
    try:
        import streamlit as st
        if st.runtime.exists():
            st.info(f"🎥 **Video Assembly Info:**\n"
                    f"- Total scenes: {len(scenes)} (Images found: {found_count}, missing/skipped: {missing_count})\n"
                    f"- Narration duration of scenes: {max_scene_end:.2f} seconds\n"
                    f"- Audio file duration: {audio.duration:.2f} seconds\n"
                    f"- **Final MP4 output duration:** {total_duration:.2f} seconds (extended last scene to match)")
    except ImportError:
        pass
        
    audio = audio.with_duration(total_duration)
    
    clips = []
    # Base background black clip
    background_clip = ColorClip(size=VIDEO_SIZE, color=(0, 0, 0)).with_duration(total_duration)
    clips.append(background_clip)
    
    for scene_index, scene in enumerate(scenes):
        image_path = scene.get('image_path')
        if not image_path or not os.path.exists(image_path):
            continue
            
        start = scene.get('start', 0.0)
        end = scene.get('end', 5.0)
        
        # If the scene starts after the total duration, skip it
        if start >= total_duration:
            continue
            
        # Limit end time to total duration
        end = min(end, total_duration)
        duration = max(0.1, end - start)
        
        animated_clip = create_keyframed_image_clip(image_path, duration, scene_index, scene.get('motion', 'auto'))
        if add_captions:
            caption_text = scene.get('caption') or scene.get('text_segment', '')
            if caption_text.strip():
                # The caption should only show for the original duration of the segment,
                # even if the visual image duration was extended.
                orig_end_val = scene.get('original_end', end)
                caption_duration = min(duration, max(0.1, orig_end_val - start))
                caption_clip = create_caption_clip(caption_text, caption_duration)
                animated_clip = CompositeVideoClip([animated_clip, caption_clip], size=VIDEO_SIZE).with_duration(duration)
        
        # Position the clip at its absolute start time
        positioned_clip = animated_clip.with_start(start)
        clips.append(positioned_clip)
        
    if len(clips) <= 1:
        audio.close()
        raise ValueError("No valid image clips to assemble.")
        
    # Composite all clips over the background
    final_video = CompositeVideoClip(clips, size=VIDEO_SIZE).with_duration(total_duration)
    final_video = final_video.with_audio(audio)
    
    # Write final MP4 video file
    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    # Close resources
    audio.close()
    final_video.close()
    
    return output_path


def transcribe_audio(audio_path):
    """
    Transcribes audio using local Whisper.
    Returns the Whisper result dict (containing text and segments).
    """
    import whisper
    import torch

    # Monkey patch Whisper Triton ops to prevent Windows import crashes
    try:
        import whisper.timing
        
        def patched_median_filter(x, filter_width):
            pad_width = filter_width // 2
            if x.shape[-1] <= pad_width:
                return x
            if x.ndim <= 2:
                x = x[None, None, :]
            x = torch.nn.functional.pad(x, (pad_width, pad_width, 0, 0), mode="reflect")
            result = x.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
            if x.ndim <= 2:
                result = result[0, 0]
            return result

        def patched_dtw(x):
            return whisper.timing.dtw_cpu(x.double().cpu().numpy())

        whisper.timing.median_filter = patched_median_filter
        whisper.timing.dtw = patched_dtw
    except Exception as e:
        print(f"[WHISPER PATCH WARNING] Failed to apply Triton bypass patch: {e}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    return model.transcribe(audio_path, word_timestamps=True)


def build_audio_first_beats(transcription_result, max_duration=4.5, max_words=30, content_type="anime"):
    """
    Chunks a Whisper transcription result into timed visual beats.
    """
    segments = transcription_result.get("segments", [])
    
    # Collect all words and their timestamps
    all_words = []
    for seg in segments:
        words_data = seg.get("words", [])
        if words_data:
            for wd in words_data:
                cleaned_word = wd.get("word", "")
                all_words.append({
                    "word": cleaned_word,
                    "start": wd["start"],
                    "end": wd.get("end", wd["start"])
                })
        else:
            seg_text = seg.get("text", "")
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            words = seg_text.split()
            if not words:
                continue
            step = (seg_end - seg_start) / len(words)
            for j, w in enumerate(words):
                all_words.append({
                    "word": w,
                    "start": seg_start + j * step,
                    "end": seg_start + (j + 1) * step
                })
                
    if not all_words:
        return []

    # Chunk words into timed scenes based on duration, word limit, and silences
    scenes = []
    current_chunk_words = []
    chunk_start = None
    
    for i, w in enumerate(all_words):
        w_text = w["word"]
        w_start = w["start"]
        w_end = w["end"]
        
        if chunk_start is None:
            chunk_start = w_start
            
        should_split = False
        if len(current_chunk_words) >= max_words:
            should_split = True
        elif w_end - chunk_start > max_duration and len(current_chunk_words) > 0:
            should_split = True
        elif i > 0 and w_start - all_words[i-1]["end"] > 1.2:
            should_split = True
            
        if should_split and current_chunk_words:
            scenes.append({
                "start": round(chunk_start, 2),
                "end": round(all_words[i-1]["end"], 2),
                "text_segment": " ".join(current_chunk_words).strip(),
                "motion": "auto",
                "image_prompt": ""
            })
            current_chunk_words = [w_text]
            chunk_start = w_start
        else:
            current_chunk_words.append(w_text)
            
    if current_chunk_words:
        scenes.append({
            "start": round(chunk_start, 2),
            "end": round(all_words[-1]["end"], 2),
            "text_segment": " ".join(current_chunk_words).strip(),
            "motion": "auto",
            "image_prompt": ""
        })
        
    return scenes


def create_audio_first_scenes(audio_path, api_key=None, max_duration=4.5, max_words=30, content_type="anime"):
    """
    Transcribes audio using local Whisper, chunks it into timed visual beats,
    and calls Gemini (if API key provided) to generate visual prompts and camera motions for each beat.
    Returns (scenes, transcript_text).
    """
    import google.generativeai as genai
    import json
    
    # 1. Transcribe audio
    result = transcribe_audio(audio_path)
    transcript_text = result.get("text", "").strip()
    
    # 2. Build beats
    scenes = build_audio_first_beats(result, max_duration=max_duration, max_words=max_words, content_type=content_type)
    
    if not scenes:
        return [], transcript_text

    # 3. Generate visual prompts and motions (using Gemini or fallback)
    if api_key and api_key.strip():
        from core.gemini_manager import run_with_rotation
        
        # Prepare segments for Gemini analysis
        prompt_input = [{"index": idx, "text_segment": scene["text_segment"]} for idx, scene in enumerate(scenes)]
        
        system_instruction = (
            f"You are a storyboard director making a video in the category: '{content_type}'.\n"
            "You will receive a JSON list of narrative beats. Your task is to output a JSON array of the exact same length "
            "where each item contains:\n"
            "- 'index': The beat index.\n"
            "- 'image_prompt': A highly detailed visual prompt suitable for AI image generation (e.g. SDXL) "
            "representing that beat's narration. Keep character descriptions and setting consistent across indices. "
            "Include descriptions of characters, actions, camera angles, color palette, and 'anime style' if appropriate.\n"
            "- 'motion': Choose the best matching Ken Burns camera motion from: ['push_in', 'slow_push', 'pull_out', 'pan_left', 'pan_right', 'auto'].\n"
            "Format output as a raw JSON array. Do not wrap in markdown or include extra text."
        )
        
        def _call_gemini(key):
            genai.configure(api_key=key.strip())
            models_to_try = [
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-flash-latest"
            ]
            last_err = None
            for model_name in models_to_try:
                try:
                    gemini_model = genai.GenerativeModel(model_name)
                    return gemini_model.generate_content(
                        f"Generate visual prompts for these beats:\n\n{json.dumps(prompt_input)}",
                        generation_config={"response_mime_type": "application/json"},
                        system_instruction=system_instruction
                    )
                except Exception as error:
                    last_err = error
            raise last_err
            
        try:
            response = run_with_rotation(_call_gemini)
            # Parse Gemini's output
            gemini_scenes = json.loads(response.text)
            for item in gemini_scenes:
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(scenes):
                    scenes[idx]["image_prompt"] = item.get("image_prompt", "")
                    scenes[idx]["motion"] = item.get("motion", "auto")
        except Exception as e:
            print(f"[GEMINI AUDIO-FIRST ERROR] {e}. Falling back to default prompts.")

    # 4. Local Fallback for prompts/motions
    motion_options = ["push_in", "slow_push", "pull_out", "pan_left", "pan_right"]
    for idx, scene in enumerate(scenes):
        if not scene.get("image_prompt"):
            scene["image_prompt"] = f"anime style digital painting, illustrating: {scene['text_segment']}, highly detailed, composition"
        if scene.get("motion") not in motion_options:
            scene["motion"] = motion_options[idx % len(motion_options)]

    return scenes, transcript_text