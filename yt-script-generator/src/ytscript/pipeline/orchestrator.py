from typing import Dict, Any, Callable
from ytscript.llm.base import BaseLLMClient
from ytscript.models import ScriptRequest, ComplianceReport, PipelineOutput
from ytscript.pipeline.research import run_research
from ytscript.pipeline.generate import run_generate
from ytscript.pipeline.monetization import run_monetization_check
from ytscript.pipeline.copyright_guard import run_copyright_check
from ytscript.pipeline.retention import run_retention_optimizer
from ytscript.pipeline.seo_report import run_seo_generator

def run_pipeline(
    llm: BaseLLMClient,
    request: ScriptRequest,
    config: Dict[str, Any],
    status_callback: Callable[[int, str], None] = None
) -> PipelineOutput:
    """
    Main Orchestrator pipeline execution engine.
    Sequentially passes inputs and intermediate structures through Stages 1-6
    to produce the final script, SEO files, and safety reports.
    Supports status_callback for live tracking in UI.
    """
    def log_status(stage: int, message: str):
        if status_callback:
            status_callback(stage, message)

    # Stage 1: Research & Outline
    log_status(1, "Researching topic outlines, defining hook angles, and gathering target keywords...")
    outline = run_research(llm, request)
    
    # Stage 2: Script Generation
    log_status(2, "Expanding outlines into full script sections with visual and audio cues...")
    script = run_generate(llm, request, outline)
    
    # Stage 3: Monetization Policy Screening
    log_status(3, "Screening script narration for profanity and advertiser safety policy violations...")
    monetization_status, monetization_flags = run_monetization_check(
        llm, script, config.get("monetization_policy", {})
    )
    
    # Stage 4: Copyright & Originality Guard
    log_status(4, "Reviewing copyright rules, fair-use warnings, and constructing royalty-free checklist...")
    copyright_status, safe_assets_checklist, copyright_flags = run_copyright_check(
        script, config.get("copyright_policy", {})
    )
    
    # Stage 5: Retention Optimization Analysis
    log_status(5, "Evaluating opening hook strength score and designing custom pattern interrupts...")
    retention_score, retention_suggestions = run_retention_optimizer(llm, script)
    
    # Stage 6: SEO Metadata Package
    log_status(6, "Generating high-CTR title suggestions, description copy, search tags, and timestamps...")
    seo = run_seo_generator(llm, request, script)
    
    # Final assembly
    log_status(6, "Aggregating all stages into final package outputs...")
    compliance = ComplianceReport(
        monetization_status=monetization_status,
        copyright_status=copyright_status,
        flags=monetization_flags + copyright_flags,
        safe_assets_checklist=safe_assets_checklist,
        retention_score=retention_score,
        retention_suggestions=retention_suggestions
    )
    
    return PipelineOutput(
        request=request,
        script=script,
        seo=seo,
        compliance=compliance
    )
