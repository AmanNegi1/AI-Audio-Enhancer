from ytscript.llm.base import BaseLLMClient
from ytscript.models import ScriptRequest, Script
from ytscript.pipeline import render_template

def run_generate(llm: BaseLLMClient, request: ScriptRequest, outline: str) -> Script:
    """
    Stage 2: Script Generation.
    Expands the research outline into a complete word-for-word YouTube script.
    Outputs a structured Script object with hook, body sections, visual/audio cues, and CTAs.
    """
    user_prompt = render_template(
        "script.j2",
        topic=request.topic,
        niche=request.niche,
        duration_minutes=request.duration_minutes,
        tone=request.tone,
        audience=request.audience,
        outline=outline,
        voice_sample=request.voice_sample
    )
    system_prompt = (
        "You are an elite YouTube script writer. Write a natural, conversational script "
        "featuring strong hook transitions and clear structural segments. Generate "
        "comprehensive visual/audio instructions for every single section."
    )
    script = llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=Script
    )
    return script
