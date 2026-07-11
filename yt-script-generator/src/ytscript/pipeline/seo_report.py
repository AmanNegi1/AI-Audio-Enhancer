from ytscript.llm.base import BaseLLMClient
from ytscript.models import ScriptRequest, Script, SEOData
from ytscript.pipeline import render_template

def run_seo_generator(
    llm: BaseLLMClient,
    request: ScriptRequest,
    script: Script
) -> SEOData:
    """
    Stage 6: SEO Package Generation.
    Uses the LLM and script details to produce a high-CTR, search-optimized
    package containing title options, keyword tag lists, description, and chapter marks.
    """
    user_prompt = render_template(
        "seo.j2",
        topic=request.topic,
        niche=request.niche,
        audience=request.audience,
        script=script
    )
    system_prompt = (
        "You are an expert YouTube SEO and digital marketing specialist. Generate click-worthy "
        "video title ideas, keyword tags, video description copy, and sequential chapters "
        "calculated based on individual section durations. Ensure output strictly matches the "
        "SEOData schema."
    )
    
    seo_data = llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=SEOData
    )
    return seo_data
