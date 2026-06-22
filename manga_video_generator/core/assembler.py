from moviepy import ImageClip, concatenate_videoclips, AudioFileClip
import os

def assemble_video(scenes, audio_path, output_path):
    """
    Assembles the generated images into a single video, synchronized with the audio track.
    Applies a dynamic Ken Burns zoom-in animation to each clip.
    """
    clips = []
    
    for scene in scenes:
        image_path = scene.get('image_path')
        if not image_path or not os.path.exists(image_path):
            continue
            
        start = scene.get('start', 0.0)
        end = scene.get('end', 5.0)
        duration = max(1.0, end - start)
        
        # Create an image clip with specified duration
        # Resize to standard 1080p (1920x1080)
        img_clip = ImageClip(image_path).with_duration(duration).resized((1920, 1080))
        
        # Apply slow, smooth Ken Burns zoom-in animation (1.0x -> 1.06x scale)
        animated_clip = img_clip.resized(lambda t, d=duration: 1.0 + 0.06 * (t / d))
        
        clips.append(animated_clip)
        
    if not clips:
        raise ValueError("No valid image clips to assemble.")
        
    # Concatenate all scenes sequentially
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Load and overlay the voiceover audio track
    audio = AudioFileClip(audio_path)
    final_video = final_video.with_audio(audio)
    
    # Write final MP4 video file
    # Using libx264 for compatibility and aac for audio encoding
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
