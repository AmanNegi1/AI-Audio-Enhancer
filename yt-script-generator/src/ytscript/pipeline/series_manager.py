from typing import Dict, Any, List, Callable, Tuple
from ytscript.llm.base import BaseLLMClient
from ytscript.models import SeriesBible, EpisodeOutline, Script, ComplianceReport, SEOData, PipelineOutput
from ytscript.pipeline import render_template

# Reuse existing validation pipelines
from ytscript.pipeline.monetization import run_monetization_check
from ytscript.pipeline.copyright_guard import run_copyright_check
from ytscript.pipeline.retention import run_retention_optimizer
from ytscript.pipeline.seo_report import run_seo_generator

def generate_series_bible(
    llm: BaseLLMClient,
    title: str,
    niche: str,
    tone: str,
    series_arc: str,
    num_episodes: int
) -> SeriesBible:
    """
    Stage 2.1: Generate Series Bible.
    Creates a structured course outline of episodes.
    """
    user_prompt = render_template(
        "series_outline.j2",
        title=title,
        niche=niche,
        tone=tone,
        series_arc=series_arc,
        num_episodes=num_episodes
    )
    system_prompt = (
        "You are an expert YouTube showrunner. Design a structured multi-episode "
        "video course outline based on the request details, outputting a SeriesBible."
    )
    
    bible = llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SeriesBible
    )
    return bible

def generate_series_episode(
    llm: BaseLLMClient,
    bible: SeriesBible,
    episode: EpisodeOutline,
    next_episode: EpisodeOutline,
    config: Dict[str, Any]
) -> PipelineOutput:
    """
    Stage 2.2: Generate Individual Episode Script with Continuity.
    """
    user_prompt = render_template(
        "series_script.j2",
        bible=bible,
        episode=episode,
        next_episode=next_episode
    )
    system_prompt = (
        "You are an elite YouTube script writer. Write a conversational script with visual "
        "and audio instructions. Ensure you align with the Series Bible, referencing "
        "previous episodes in the hook and cliffhanging to the next episode at the end."
    )
    
    # Generate structured script
    script = llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=Script
    )
    
    # Run monetization safety
    monetization_status, monetization_flags = run_monetization_check(
        llm, script, config.get("monetization_policy", {})
    )
    
    # Run copyright guard
    copyright_status, safe_assets_checklist, copyright_flags = run_copyright_check(
        script, config.get("copyright_policy", {})
    )
    
    # Run retention optimizer
    retention_score, retention_suggestions = run_retention_optimizer(llm, script)
    
    # Generate SEO package (reusing script-request wrapper parameters)
    from ytscript.models import ScriptRequest
    dummy_req = ScriptRequest(
        topic=episode.title,
        niche=bible.niche,
        tone=bible.tone,
        audience="general",
        duration_minutes=script.estimated_duration_seconds / 60.0
    )
    seo = run_seo_generator(llm, dummy_req, script)
    
    compliance = ComplianceReport(
        monetization_status=monetization_status,
        copyright_status=copyright_status,
        flags=monetization_flags + copyright_flags,
        safe_assets_checklist=safe_assets_checklist,
        retention_score=retention_score,
        retention_suggestions=retention_suggestions
    )
    
    return PipelineOutput(
        request=dummy_req,
        script=script,
        seo=seo,
        compliance=compliance
    )
