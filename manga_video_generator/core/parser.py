import json
import re
import google.generativeai as genai

GEMINI_TEXT_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]

def split_text_into_beats(text, max_words=38):
    sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text.strip()) if sentence.strip()]
    if not sentences:
        words = text.split()
        return [" ".join(words[index:index + max_words]) for index in range(0, len(words), max_words)]

    beats = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(sentence_words) > max_words:
            if current_sentences:
                beats.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0
            beats.extend(" ".join(sentence_words[index:index + max_words]) for index in range(0, len(sentence_words), max_words))
            continue

        if current_sentences and current_word_count + len(sentence_words) > max_words:
            beats.append(" ".join(current_sentences))
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += len(sentence_words)

    if current_sentences:
        beats.append(" ".join(current_sentences))

    return [beat for beat in beats if beat.strip()]

def refine_scene_pacing(scenes, max_words=38):
    refined_scenes = []
    for scene in scenes:
        text_segment = scene.get("text_segment", "").strip()
        image_prompt = scene.get("image_prompt", "").strip()
        if not text_segment:
            continue

        beats = split_text_into_beats(text_segment, max_words=max_words)
        if len(beats) <= 1:
            refined_scenes.append({
                "text_segment": text_segment,
                "image_prompt": image_prompt,
            })
            continue

        for beat in beats:
            refined_scenes.append({
                "text_segment": beat,
                "image_prompt": (
                    f"{image_prompt}. Focus this frame specifically on this narration: {beat}. "
                    "Keep the same character and location continuity, but make this beat's action and emotion the main visual subject."
                ),
            })

    return refined_scenes

def parse_script_to_prompts(script_text, api_key):
    """
    Splits the script into logical scene beats and writes descriptive text-to-image prompts.
    Returns a list of dicts: [{'text_segment': str, 'image_prompt': str}]
    """
    from core.gemini_manager import run_with_rotation
    
    system_instruction = (
        "You are an expert anime storyboard director. Your task is to analyze a video script, "
        "split it into a sequence of logical narrative beats (scenes), and write detailed image generation prompts "
        "for each beat.\n"
        "Ensure the output is a valid JSON array. Each element in the array must be an object with exactly "
        "two fields:\n"
        "- 'text_segment': The subset of script text spoken during this beat.\n"
        "- 'image_prompt': A highly descriptive text-to-image prompt to generate a 16:9 widescreen illustration "
        "representing the exact narration in text_segment, while using the previous and next beats only for continuity. "
        "The prompt must identify the main subject, action, setting, emotion, camera framing, lighting, and color palette "
        "from the spoken context so the image feels tightly matched to the voiceover. Preserve recurring characters and locations "
        "with consistent generic descriptions across beats. Use descriptors like 'anime style', 'dramatic lighting', 'high detail', "
        "and mention composition and coloring. Avoid vague recap posters, title cards, symbolic filler, or unrelated generic action. "
        "Avoid using trademarked character names; use generic physical descriptions "
        "(e.g., 'a boy with spiky black hair and a scar on his face, wearing a dark coat').\n"
        "Do not include any markdown format tags like ```json or other text in your response, output raw JSON only."
    )
    
    prompt = f"Analyze the following script and break it down into sequential scenes:\n\n{script_text}"
    
    def _call_gemini(key):
        genai.configure(api_key=key)
        response = None
        last_error = None
        for model_name in GEMINI_TEXT_MODELS:
            try:
                model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                    safety_settings=[]
                )
                return response
            except Exception as error:
                last_error = error
        if response is None:
            raise last_error

    try:
        response = run_with_rotation(_call_gemini)
        
        # Parse JSON output
        scenes = json.loads(response.text)
        if not isinstance(scenes, list):
            raise ValueError("Expected a JSON array of scenes.")
        return refine_scene_pacing(scenes)
        
    except Exception as e:
        # Fallback in case of parsing errors or API limits: split by paragraph
        paragraphs = [p.strip() for p in script_text.split('\n\n') if p.strip()]
        fallback_scenes = []
        for p in paragraphs:
            for beat in split_text_into_beats(p):
                fallback_scenes.append({
                    "text_segment": beat,
                    "image_prompt": f"Anime style 16:9 illustration matching this narration: {beat[:180]}..., main subject and action clearly visible, expressive emotion, cinematic composition, dramatic lighting, detailed digital art"
                })
        return fallback_scenes