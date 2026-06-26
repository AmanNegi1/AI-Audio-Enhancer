import base64
import json
from io import BytesIO

from PIL import Image

from core.audio_first_recaper import build_audio_first_beats, transcribe_audio
from core.parser import split_text_into_beats


DETAIL_PROFILES = {
    "Medium": "Write 90-130 words with clear subject, action, setting, camera, lighting, and style.",
    "High": "Write 150-220 words with rich subject design, action, setting, emotion, camera, composition, lighting, color, props, atmosphere, and style.",
    "Ultra": "Write 220-320 words with highly specific character redesign, environment details, foreground/background layers, cinematic lighting, camera lens/framing, motion energy, texture, mood, and negative constraints.",
}

MOTION_SEQUENCE = ["push_in", "pan_left", "pan_right", "slow_push", "pull_out"]


def image_to_data_url(image):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_instruction(narration, detail_strength, art_style, originality_strength):
    detail_rule = DETAIL_PROFILES.get(detail_strength, DETAIL_PROFILES["High"])
    return (
        "Analyze the uploaded black-and-white manga panel as a private storyboard reference, then create a new original image-generation prompt. "
        "Do not copy exact panel composition, line art, character identity, costumes, logos, text, speech bubbles, or recognizable named characters. "
        f"Originality strength: {originality_strength}/100. The higher this is, the more you must redesign faces, hair, clothing, props, setting, and camera framing while preserving only the broad story function. "
        f"Narration beat to visualize: {narration or 'No narration supplied; infer a strong original scene from the panel action.'}\n\n"
        "Return raw JSON only with these keys: panel_description, image_prompt. "
        "Do not use unescaped double quotes inside the JSON string values. Use single quotes for any dialogue or inner quotes (e.g. 'hello' instead of \"hello\"). "
        "panel_description should describe the source panel's broad action, mood, pose, and setting without names. "
        "image_prompt must be a detailed prompt for a new 16:9 YouTube recap visual. "
        f"Use this art direction: {art_style}. {detail_rule} "
        "The image_prompt must include subject, redesigned character appearance, action, setting, emotion, camera angle, composition, lighting, colors, atmosphere, and constraints: no text, no watermark, no speech bubbles, no copied panel composition."
    )


def parse_json_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Robust fallback: attempt to extract and repair unescaped double quotes inside values
        try:
            import re
            
            # Extract panel_description value
            desc_match = re.search(r'"panel_description"\s*:\s*"(.*?)"\s*,\s*"image_prompt"', cleaned, re.DOTALL)
            if not desc_match:
                desc_match = re.search(r'"panel_description"\s*:\s*"(.*?)"\s*}', cleaned, re.DOTALL)
                
            # Extract image_prompt value
            prompt_match = re.search(r'"image_prompt"\s*:\s*"(.*?)"\s*}', cleaned, re.DOTALL)
            if not prompt_match:
                prompt_match = re.search(r'"image_prompt"\s*:\s*"(.*?)"\s*,\s*"panel_description"', cleaned, re.DOTALL)
                
            result = {}
            if desc_match:
                result["panel_description"] = desc_match.group(1).replace('\\"', '"')
            if prompt_match:
                result["image_prompt"] = prompt_match.group(1).replace('\\"', '"')
                
            if result:
                return json.loads(json.dumps(result))
        except Exception:
            pass
        raise e


def build_batch_instruction(scenes, n_panels, detail_strength, art_style, originality_strength):
    detail_rule = DETAIL_PROFILES.get(detail_strength, DETAIL_PROFILES["High"])
    n_scenes = len(scenes)
    beats_text = "\n".join(
        f"Beat {i + 1}: {scene.get('text_segment', 'No narration')}"
        for i, scene in enumerate(scenes)
    )
    return (
        f"You are analyzing {n_panels} black-and-white manga panels (uploaded below, in order) as private storyboard references. "
        f"Generate {n_scenes} original image-generation prompts for a YouTube anime recap video — one per narration beat.\n\n"
        f"Narration beats:\n{beats_text}\n\n"
        "Rules:\n"
        "- Do not copy exact panel composition, line art, character identity, costumes, logos, text, speech bubbles, or recognizable named characters.\n"
        f"- Originality strength: {originality_strength}/100. Redesign faces, hair, clothing, props, setting, and framing while preserving only the broad story function.\n"
        "- Map each beat to the most relevant panel based on narrative arc position. Record which panel you used in source_panel_index (0-based).\n"
        "- CRITICAL: Keep character appearance, setting style, and color palette CONSISTENT across ALL prompts. "
        "Use the same redesigned character descriptions (hair, outfit, build, weapon) every time that character appears.\n\n"
        f"Art direction for all images: {art_style}. {detail_rule}\n\n"
        "Each image_prompt must include: subject, redesigned character appearance, action, setting, emotion, camera angle, "
        "composition, lighting, colors, atmosphere. End every prompt with: no text, no watermark, no speech bubbles, no copied panel composition.\n\n"
        f"Return ONLY a valid JSON array of exactly {n_scenes} objects. Each object must have:\n"
        '- "panel_description": brief description of the source panel action/mood (no character names)\n'
        '- "image_prompt": the full image generation prompt\n'
        '- "source_panel_index": integer, 0-based index of the panel you referenced\n\n'
        "Do not use unescaped double quotes inside JSON string values. Use single quotes for any inner dialogue."
    )


def analyze_all_panels_with_gemini(panel_images, scenes, api_key, detail_strength, art_style, originality_strength):
    import google.generativeai as genai
    from core.gemini_manager import run_with_rotation

    instruction = build_batch_instruction(scenes, len(panel_images), detail_strength, art_style, originality_strength)

    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-flash-latest",
    ]

    def _call_gemini(key):
        genai.configure(api_key=key.strip())
        last_err = None
        for name in models_to_try:
            try:
                model = genai.GenerativeModel(name)
                content_parts = [instruction] + [img.convert("RGB") for img in panel_images]
                response = model.generate_content(
                    content_parts,
                    generation_config={"response_mime_type": "application/json"},
                    safety_settings=[],
                )
                return response
            except Exception as error:
                last_err = error
        raise last_err

    response = run_with_rotation(_call_gemini)

    cleaned = response.text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    result = json.loads(cleaned)
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                result = v
                break
    return result


def analyze_panel_with_gemini(image, api_key, narration, detail_strength, art_style, originality_strength, model_name="gemini-2.0-flash"):
    import google.generativeai as genai
    from core.gemini_manager import run_with_rotation

    instruction = build_instruction(narration, detail_strength, art_style, originality_strength)
    
    models_to_try = [model_name] + [
        name for name in [
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-flash-latest"
        ] if name != model_name
    ]
    
    def _call_gemini(key):
        genai.configure(api_key=key.strip())
        last_err = None
        for name in models_to_try:
            try:
                model = genai.GenerativeModel(name)
                return model.generate_content(
                    [instruction, image.convert("RGB")],
                    generation_config={"response_mime_type": "application/json"},
                    safety_settings=[],
                )
            except Exception as error:
                last_err = error
        raise last_err
        
    response = run_with_rotation(_call_gemini)
    return parse_json_response(response.text)


def analyze_panel_with_openai(image, api_key, narration, detail_strength, art_style, originality_strength, model_name="gpt-4o-mini"):
    from core.openai_manager import run_with_openai_rotation
    import requests

    instruction = build_instruction(narration, detail_strength, art_style, originality_strength)
    
    models_to_try = [model_name] + [name for name in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"] if name != model_name]
    
    def _call_openai(key):
        last_err = None
        for model in models_to_try:
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": instruction},
                                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                                ],
                            }
                        ],
                        "temperature": 0.7,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                err_msg = str(e).lower()
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        err_msg += " " + e.response.text.lower()
                    except:
                        pass
                if "does not exist" in err_msg or "invalid_value" in err_msg or "not found" in err_msg:
                    last_err = e
                    continue
                else:
                    raise e
        raise last_err

    text = run_with_openai_rotation(_call_openai, passed_key=api_key)
    return parse_json_response(text)


def fallback_panel_prompt(narration, art_style, detail_strength, originality_strength):
    detail_rule = DETAIL_PROFILES.get(detail_strength, DETAIL_PROFILES["High"])
    return {
        "panel_description": "Panel analysis unavailable; using narration-only original scene planning.",
        "image_prompt": (
            "Original 16:9 anime cinematic recap scene inspired by the narration, not copied from any source panel. "
            f"Narration beat: {narration}. Redesign all characters with new faces, hair, clothes, props, and setting details. "
            f"Originality strength {originality_strength}/100. {art_style}. {detail_rule} "
            "Include expressive character acting, clear focal subject, dynamic composition, dramatic lighting, rich background detail, no text, no watermark, no speech bubbles."
        ),
    }


def build_text_scenes(narration_text, seconds_per_scene=4.0, max_words=34):
    beats = split_text_into_beats(narration_text, max_words=max_words) if narration_text.strip() else []
    if not beats:
        beats = ["Create a new original anime recap visual inspired by this uploaded manga panel."]

    scenes = []
    for index, beat in enumerate(beats):
        start = round(index * seconds_per_scene, 2)
        end = round(start + seconds_per_scene, 2)
        scenes.append({
            "start": start,
            "end": end,
            "text_segment": beat,
            "motion": MOTION_SEQUENCE[index % len(MOTION_SEQUENCE)],
        })
    return scenes, narration_text.strip()


def create_panel_remix_scenes(
    panel_images,
    api_key,
    analyzer_backend="gemini",
    narration_text="",
    audio_path=None,
    detail_strength="High",
    art_style="anime style, highly detailed digital painting, cinematic lighting, 16:9 aspect ratio",
    originality_strength=85,
    max_duration=4.5,
    max_words=30,
    openai_model="gpt-4o-mini",
):
    if not panel_images:
        raise ValueError("Upload at least one manga panel image.")

    if audio_path:
        transcription = transcribe_audio(audio_path)
        scenes = build_audio_first_beats(transcription, max_duration=max_duration, max_words=max_words, content_type="Anime / Manga Recap")
        transcript_text = transcription.get("text", "").strip()
    else:
        scenes, transcript_text = build_text_scenes(narration_text, max_duration, max_words=max_words)

    if not scenes:
        scenes, transcript_text = build_text_scenes(narration_text, max_duration, max_words=max_words)

    # --- Batched Gemini analysis (one call sees all panels + all beats → consistent characters) ---
    batch_results = None
    if api_key.strip() and analyzer_backend == "gemini":
        try:
            batch_results = analyze_all_panels_with_gemini(
                panel_images, scenes, api_key, detail_strength, art_style, originality_strength
            )
            if not isinstance(batch_results, list) or len(batch_results) != len(scenes):
                batch_results = None  # unexpected shape → fall back to per-scene
        except Exception:
            batch_results = None  # any failure → fall back to per-scene

    for index, scene in enumerate(scenes):
        narration = scene.get("text_segment", "")

        if batch_results is not None:
            result = batch_results[index]
            scene["source_panel_index"] = min(
                int(result.get("source_panel_index", index % len(panel_images))),
                len(panel_images) - 1,
            )
            scene["panel_description"] = result.get("panel_description", "").strip()
            scene["image_prompt"] = result.get("image_prompt", "").strip() or fallback_panel_prompt(narration, art_style, detail_strength, originality_strength)["image_prompt"]
        else:
            # Per-scene fallback: narrative position mapping instead of naive cycling
            panel_idx = min(int(index / len(scenes) * len(panel_images)), len(panel_images) - 1)
            image = panel_images[panel_idx]
            if not api_key.strip():
                analysis = fallback_panel_prompt(narration, art_style, detail_strength, originality_strength)
            elif analyzer_backend == "openai":
                analysis = analyze_panel_with_openai(image, api_key, narration, detail_strength, art_style, originality_strength, model_name=openai_model)
            else:
                analysis = analyze_panel_with_gemini(image, api_key, narration, detail_strength, art_style, originality_strength)

            scene["source_panel_index"] = panel_idx
            scene["panel_description"] = analysis.get("panel_description", "").strip()
            scene["image_prompt"] = analysis.get("image_prompt", "").strip() or fallback_panel_prompt(narration, art_style, detail_strength, originality_strength)["image_prompt"]

    # --- Group consecutive beats that reference the same panel → one image per panel segment ---
    # This reduces 67 beats (for 5-min audio) down to ~15-20 groups matching the panel count,
    # dramatically cutting image generation cost while keeping audio sync intact.
    grouped = []
    for scene in scenes:
        if grouped and grouped[-1]["source_panel_index"] == scene["source_panel_index"]:
            grouped[-1]["end"] = scene["end"]
            grouped[-1]["text_segment"] = grouped[-1]["text_segment"] + " " + scene["text_segment"]
        else:
            grouped.append(dict(scene))
    scenes = grouped

    return scenes, transcript_text