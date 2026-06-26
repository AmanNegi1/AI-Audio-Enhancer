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
    # Collect valid scenes (images that were successfully generated)
    valid_scene_items = []
    found_count = 0
    missing_count = 0
    for i, scene in enumerate(scenes):
        image_path = scene.get('image_path')
        if image_path and os.path.exists(image_path):
            valid_scene_items.append((i, scene))
            found_count += 1
        else:
            missing_count += 1

    if not valid_scene_items:
        raise ValueError("No valid image clips to assemble.")

    max_scene_end = max(s.get('end', 5.0) for _, s in valid_scene_items)

    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    # Print status to terminal
    print(f"[ASSEMBLER] scenes count: {len(scenes)}, images found: {found_count}, missing: {missing_count}")
    print(f"[ASSEMBLER] max_scene_end: {max_scene_end:.2f}s, audio duration: {total_duration:.2f}s")
    if max_scene_end < total_duration * 0.98:
        print(f"[ASSEMBLER] Scenes cover {max_scene_end:.2f}s of {total_duration:.2f}s audio — will cycle images to fill remaining {total_duration - max_scene_end:.2f}s")

    # Display status in Streamlit if running inside a Streamlit app
    try:
        import streamlit as st
        if st.runtime.exists():
            fill_note = ""
            if max_scene_end < total_duration * 0.98:
                fill_note = f" The remaining {total_duration - max_scene_end:.1f}s will be covered by cycling the {found_count} generated images."
            st.info(
                f"🎥 **Video Assembly Info:**\n"
                f"- Total scenes: {len(scenes)} (Images found: {found_count}, missing/skipped: {missing_count})\n"
                f"- Scenes cover: {max_scene_end:.2f} seconds\n"
                f"- Audio duration: {total_duration:.2f} seconds\n"
                f"- **Final MP4 output duration:** {total_duration:.2f} seconds" + fill_note
            )
    except ImportError:
        pass

    audio = audio.with_duration(total_duration)

    clips = []
    # Base background black clip
    background_clip = ColorClip(size=VIDEO_SIZE, color=(0, 0, 0)).with_duration(total_duration)
    clips.append(background_clip)

    # Step 1: Add all generated scenes at their proper timestamps
    for scene_index, scene in valid_scene_items:
        image_path = scene.get('image_path')
        start = scene.get('start', 0.0)
        end = scene.get('end', 5.0)

        if start >= total_duration:
            continue

        end = min(end, total_duration)
        duration = max(0.1, end - start)

        animated_clip = create_keyframed_image_clip(image_path, duration, scene_index, scene.get('motion', 'auto'))
        if add_captions:
            caption_text = scene.get('caption') or scene.get('text_segment', '')
            if caption_text.strip():
                caption_clip = create_caption_clip(caption_text, duration)
                animated_clip = CompositeVideoClip([animated_clip, caption_clip], size=VIDEO_SIZE).with_duration(duration)

        clips.append(animated_clip.with_start(start))

    # Step 2: If scenes don't cover the full audio, cycle through all generated images
    # to fill the remaining time instead of freezing on the last frame.
    if max_scene_end < total_duration * 0.98:
        avg_beat_duration = max_scene_end / max(found_count, 1)
        fill_seg_duration = min(max(avg_beat_duration, 2.0), 5.0)

        fill_start = max_scene_end
        fill_cycle_idx = 0
        while fill_start < total_duration - 0.05:
            _, fill_scene = valid_scene_items[fill_cycle_idx % len(valid_scene_items)]
            fill_end = min(fill_start + fill_seg_duration, total_duration)
            fill_duration = max(0.1, fill_end - fill_start)

            fill_clip = create_keyframed_image_clip(
                fill_scene['image_path'],
                fill_duration,
                fill_cycle_idx,
                fill_scene.get('motion', 'auto')
            )
            clips.append(fill_clip.with_start(fill_start))
            fill_start = fill_end
            fill_cycle_idx += 1
        
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