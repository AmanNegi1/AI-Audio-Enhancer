from moviepy import (
    ImageClip, concatenate_videoclips, concatenate_audioclips,
    AudioFileClip, AudioArrayClip, ColorClip, CompositeVideoClip, CompositeAudioClip, VideoClip,
)
import math
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_SIZE = (1920, 1080)
FADE_DURATION = 0.35  # seconds of cross-dissolve overlap between clips

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


def make_fade_in_mask(fade_duration, total_duration):
    """Returns a mask clip that fades opacity from 0 to 1 over fade_duration seconds."""
    def make_frame(t):
        opacity = min(1.0, t / max(fade_duration, 0.001))
        return np.full((VIDEO_SIZE[1], VIDEO_SIZE[0]), opacity)
    return VideoClip(make_frame, duration=total_duration, is_mask=True)


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


def create_channel_overlay_clip(channel_name, show_duration=3.0):
    """Animated channel name tag: slides in from the left, holds, then fades out."""
    slide_in   = 0.35
    fade_start = show_duration - 0.4

    try:
        font_main = ImageFont.truetype("arialbd.ttf", 40)
        font_icon = ImageFont.truetype("arialbd.ttf", 32)
    except OSError:
        font_main = ImageFont.load_default()
        font_icon = font_main

    # Pre-measure badge width
    _tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    icon_w  = int(_tmp.textlength("\u25b6 ", font=font_icon))
    text_w  = int(_tmp.textlength(channel_name, font=font_main))
    pad_x   = 26
    badge_w = icon_w + text_w + pad_x * 2 + 10
    badge_h = 72
    badge_x_final = 60
    badge_y = VIDEO_SIZE[1] - 170

    def make_frame(t):
        # Slide-in x position
        if t < slide_in:
            prog    = ease_in_out(t / slide_in)
            badge_x = int(badge_x_final - badge_w * (1.0 - prog))
        else:
            badge_x = badge_x_final

        # Fade-out alpha
        if t >= fade_start:
            alpha = max(0.0, 1.0 - (t - fade_start) / max(show_duration - fade_start, 0.001))
        else:
            alpha = 1.0

        frame = Image.new("RGBA", VIDEO_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(frame)
        a = lambda v: int(v * alpha)

        # Background pill
        d.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
            radius=16, fill=(10, 10, 10, a(200)), outline=(220, 30, 30, a(255)), width=3,
        )
        # Red accent bar on left
        d.rounded_rectangle(
            [badge_x, badge_y, badge_x + 8, badge_y + badge_h],
            radius=8, fill=(220, 30, 30, a(255)),
        )
        # Play icon
        d.text((badge_x + 18, badge_y + badge_h // 2 - 18), "\u25b6 ",
               font=font_icon, fill=(220, 30, 30, a(255)))
        # Channel name (shadow + white)
        tx = badge_x + 18 + icon_w
        ty = badge_y + badge_h // 2 - 22
        d.text((tx + 2, ty + 2), channel_name, font=font_main, fill=(0, 0, 0, a(180)))
        d.text((tx, ty),         channel_name, font=font_main, fill=(255, 255, 255, a(255)))
        return np.array(frame)

    return VideoClip(make_frame, duration=show_duration)


def create_lss_clip(duration=5.0):
    """Animated Like / Share / Subscribe boxes that slide up at the bottom-centre."""
    stagger    = 0.22
    slide_time = 0.28
    fade_start = duration - 0.5

    labels = [("LIKE", (66, 103, 212)), ("SHARE", (46, 160, 67)), ("SUBSCRIBE", (220, 30, 30))]

    try:
        font = ImageFont.truetype("arialbd.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    _tmp     = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    pad_x    = 32
    box_h    = 66
    gap      = 18
    widths   = [int(_tmp.textlength(lbl, font=font)) + pad_x * 2 for lbl, _ in labels]
    total_w  = sum(widths) + gap * (len(labels) - 1)
    start_x  = (VIDEO_SIZE[0] - total_w) // 2
    box_y_f  = VIDEO_SIZE[1] - 110

    def make_frame(t):
        global_alpha = max(0.0, 1.0 - (t - fade_start) / max(duration - fade_start, 0.001)) \
                       if t >= fade_start else 1.0
        frame = Image.new("RGBA", VIDEO_SIZE, (0, 0, 0, 0))
        d     = ImageDraw.Draw(frame)
        x     = start_x
        for i, (label, (r, g, b)) in enumerate(labels):
            local_t = t - i * stagger
            if local_t <= 0:
                x += widths[i] + gap
                continue
            prog       = ease_in_out(min(local_t / slide_time, 1.0))
            box_alpha  = prog * global_alpha
            slide_off  = int(36 * (1.0 - prog))
            box_y      = box_y_f + slide_off
            a          = lambda v: int(v * box_alpha)
            d.rounded_rectangle(
                [x, box_y, x + widths[i], box_y + box_h],
                radius=14, fill=(r, g, b, a(230)),
            )
            tw  = int(_tmp.textlength(label, font=font))
            tx  = x + (widths[i] - tw) // 2
            ty  = box_y + box_h // 2 - 22
            d.text((tx + 2, ty + 2), label, font=font, fill=(0, 0, 0, a(140)))
            d.text((tx,     ty),     label, font=font, fill=(255, 255, 255, a(255)))
            x += widths[i] + gap
        return np.array(frame)

    return VideoClip(make_frame, duration=duration)


def create_whoosh_sfx(duration=0.28, sample_rate=44100, volume=0.15):
    """
    Generates a synthetic whoosh sound effect as a stereo numpy array (N, 2).
    Uses a descending frequency sweep + noise burst — no external audio files needed.
    """
    N = int(sample_rate * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    # Exponential frequency sweep: 2800 Hz → 180 Hz (classic whoosh contour)
    f_start, f_end = 2800.0, 180.0
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t ** 2 / (2 * duration))
    sweep = np.sin(phase)
    # Layered noise for texture (band-limited feel)
    rng = np.random.default_rng(42)          # fixed seed → consistent sound
    noise = rng.standard_normal(N) * 0.30
    # Envelope: fast 8% attack, exponential tail
    attack_n = max(1, int(N * 0.08))
    env = np.ones(N, dtype=np.float32)
    env[:attack_n] = np.linspace(0.0, 1.0, attack_n)
    env *= np.exp(-t * 5.5 / max(duration, 0.001))
    signal = (sweep + noise) * env * volume
    signal = np.clip(signal, -1.0, 1.0).astype(np.float32)
    return np.column_stack([signal, signal])  # stereo (N, 2)


def assemble_video(scenes, audio_path, output_path, add_captions=False,
                   bgm_path=None, bgm_volume=0.15,
                   channel_name=None, show_lss=False, lss_duration=5.0,
                   add_transition_sfx=False, sfx_volume=0.12):
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

    # Mix background music if provided
    bgm_raw = None
    if bgm_path and os.path.exists(bgm_path):
        bgm_raw = AudioFileClip(bgm_path)
        if bgm_raw.duration < total_duration:
            n_loops = math.ceil(total_duration / bgm_raw.duration)
            bgm_raw = concatenate_audioclips([bgm_raw] * n_loops)
        bgm_clip = bgm_raw.subclipped(0, total_duration).multiply_volume(bgm_volume).audio_fadeout(2.0)
        mixed_audio = CompositeAudioClip([audio, bgm_clip])
    else:
        mixed_audio = audio

    # Mix transition whoosh SFX at each scene cut
    if add_transition_sfx:
        _sfx_array = create_whoosh_sfx(volume=sfx_volume)
        _sfx_clips = []
        for _, _sc in valid_scene_items:
            _t = _sc.get('start', 0.0)
            # Skip the very first frame and anything too close to the end
            if 0.5 < _t < total_duration - 0.5:
                _sfx_clips.append(
                    AudioArrayClip(_sfx_array, fps=44100).with_start(_t)
                )
        if _sfx_clips:
            mixed_audio = CompositeAudioClip([mixed_audio] + _sfx_clips)

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

    # Step 1: Add all generated scenes at their proper timestamps with cross-dissolve transitions.
    # Each clip starts FADE_DURATION seconds early and fades in from transparent, so clips overlap
    # and dissolve into each other rather than hard-cutting.
    for scene_index, scene in valid_scene_items:
        image_path = scene.get('image_path')
        start = scene.get('start', 0.0)
        end = scene.get('end', 5.0)

        if start >= total_duration:
            continue

        end = min(end, total_duration)
        duration = max(0.1, end - start)

        # Cross-dissolve: extend clip start backward by fade amount so it overlaps with previous
        fade_in = min(FADE_DURATION, start)  # can't start before t=0
        clip_start = max(0.0, start - fade_in)
        clip_duration = duration + fade_in

        animated_clip = create_keyframed_image_clip(image_path, clip_duration, scene_index, scene.get('motion', 'auto'))

        # Apply fade-in mask so clip dissolves in over the previous one
        if fade_in > 0.05:
            animated_clip = animated_clip.with_mask(make_fade_in_mask(fade_in, clip_duration))

        if add_captions:
            caption_text = scene.get('caption') or scene.get('text_segment', '')
            if caption_text.strip():
                caption_clip = create_caption_clip(caption_text, clip_duration)
                animated_clip = CompositeVideoClip([animated_clip, caption_clip], size=VIDEO_SIZE).with_duration(clip_duration)

        clips.append(animated_clip.with_start(clip_start))

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
            if fill_start > 0.05:
                fill_clip = fill_clip.with_mask(make_fade_in_mask(min(FADE_DURATION, fill_duration * 0.4), fill_duration))
            clips.append(fill_clip.with_start(fill_start))
            fill_start = fill_end
            fill_cycle_idx += 1
        
    # Channel name overlay (slides in at t=0)
    if channel_name:
        ch_clip = create_channel_overlay_clip(channel_name, show_duration=3.0)
        clips.append(ch_clip.with_start(0))

    # Like / Share / Subscribe overlay (animates in at the end)
    if show_lss and total_duration > lss_duration:
        lss_clip = create_lss_clip(duration=lss_duration)
        clips.append(lss_clip.with_start(total_duration - lss_duration))

    # Composite all clips over the background
    final_video = CompositeVideoClip(clips, size=VIDEO_SIZE).with_duration(total_duration)
    final_video = final_video.with_audio(mixed_audio)
    
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
    if bgm_raw is not None:
        try:
            bgm_raw.close()
        except Exception:
            pass
    final_video.close()
    
    return output_path