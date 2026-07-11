from typing import List, Tuple
from pydantic import BaseModel, Field
from ytscript.llm.base import BaseLLMClient
from ytscript.models import Script
from ytscript.pipeline import render_template

class LLMRetentionEval(BaseModel):
    retention_score: int = Field(..., description="Estimated audience retention score (1-100) based on hook, pacing, and CTAs")
    retention_suggestions: List[str] = Field(..., description="Actionable recommendations to improve pacing or keep viewers watching longer")

def run_retention_optimizer(
    llm: BaseLLMClient,
    script: Script
) -> Tuple[int, List[str]]:
    """
    Stage 5: Retention Optimizer.
    Uses LLM analysis to evaluate the script structure, score the hook strength,
    and suggest custom pattern interrupts.
    """
    user_prompt = render_template("retention.j2", script=script)
    system_prompt = (
        "You are an expert YouTube retention strategist. Evaluate hook strength, pacing, "
        "and visual transitions in this script. Grade hook strength (1-100) and suggest "
        "pattern interrupts (text overlays, B-roll transitions) to maximize viewer retention."
    )

    try:
        eval_res = llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LLMRetentionEval
        )
        return eval_res.retention_score, eval_res.retention_suggestions
    except Exception as e:
        # Robust fallback default retention analysis
        return 80, [
            "Keep the intro hook tight and under 15 seconds.",
            "Insert visual zooms or sound effect transitions every 15-20 seconds.",
            "Verify that the CTA occurs after offering value, rather than right at the start.",
            f"System Note: LLM retention check failed: {str(e)}"
        ]
