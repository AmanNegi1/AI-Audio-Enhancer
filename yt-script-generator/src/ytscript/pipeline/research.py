from ytscript.llm.base import BaseLLMClient
from ytscript.models import ScriptRequest
from ytscript.pipeline import render_template

def run_research(llm: BaseLLMClient, request: ScriptRequest) -> str:
    """
    Stage 1: Research & Outline.
    Analyzes the script request details (topic, niche, tone, duration, references)
    and uses the LLM to generate a search-optimized, structured outline.
    """
    user_prompt = render_template(
        "outline.j2",
        topic=request.topic,
        niche=request.niche,
        duration_minutes=request.duration_minutes,
        tone=request.tone,
        audience=request.audience,
        references=request.references
    )
    system_prompt = (
        "You are an expert YouTube content strategist. Generate research notes, target "
        "keywords, and a detailed chronological outline for the video. State the core "
        "hook angle and value proposition."
    )
    outline = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    return outline
