import json
from google import genai
from google.genai import types

def parse_script_to_prompts(script_text, api_key):
    """
    Splits the script into logical scene beats and writes descriptive text-to-image prompts.
    Returns a list of dicts: [{'text_segment': str, 'image_prompt': str}]
    """
    client = genai.Client(api_key=api_key)

    system_instruction = (
        "You are an expert anime storyboard director. Your task is to analyze a video script, "
        "split it into a sequence of logical narrative beats (scenes), and write detailed image generation prompts "
        "for each beat.\n"
        "Ensure the output is a valid JSON array. Each element in the array must be an object with exactly "
        "two fields:\n"
        "- 'text_segment': The subset of script text spoken during this beat.\n"
        "- 'image_prompt': A highly descriptive text-to-image prompt to generate a 16:9 widescreen illustration "
        "representing that exact moment. Use descriptors like 'anime style', 'dramatic lighting', 'high detail', "
        "and mention composition and coloring. Avoid using trademarked character names; use generic physical descriptions "
        "(e.g., 'a boy with spiky black hair and a scar on his face, wearing a dark coat').\n"
        "Do not include any markdown format tags like ```json or other text in your response, output raw JSON only."
    )

    prompt = f"Analyze the following script and break it down into sequential scenes:\n\n{script_text}"

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )
        
        # Parse JSON output
        scenes = json.loads(response.text)
        if not isinstance(scenes, list):
            raise ValueError("Expected a JSON array of scenes.")
        return scenes
        
    except Exception as e:
        # Fallback in case of parsing errors or API limits: split by paragraph
        paragraphs = [p.strip() for p in script_text.split('\n\n') if p.strip()]
        fallback_scenes = []
        for p in paragraphs:
            fallback_scenes.append({
                "text_segment": p,
                "image_prompt": f"Anime style illustration showing: {p[:120]}..., detailed, digital art, 16:9 aspect ratio"
            })
        return fallback_scenes