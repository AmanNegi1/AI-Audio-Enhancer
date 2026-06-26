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
        print(f"[ASSEMBLER] Extended last valid scene #{last_valid_scene_idx+1} from {orig_end:.2f}s to {total_duration:.2f}s")
        
    # Print status to terminal
    print(f"[ASSEMBLER] scenes count: {len(scenes)}, images found: {found_count}, missing: {missing_count}")
    print(f"[ASSEMBLER] max_scene_end before extension: {max_scene_end}s, audio.duration: {audio.duration}s -> total_duration: {total_duration}s")
    
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