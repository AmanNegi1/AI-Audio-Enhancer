import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from ytscript.llm.base import BaseLLMClient
from ytscript.models import Script
from ytscript.pipeline import render_template

class LLMMonetizationEval(BaseModel):
    status: str = Field(..., description="Monetization status: 'Safe', 'Warning', or 'Flagged'")
    issues: List[str] = Field(..., description="List of specific violations, profanities, or advertiser-unfriendly references")

def run_monetization_check(
    llm: BaseLLMClient,
    script: Script,
    policy_config: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """
    Stage 3: Monetization Policy Screening.
    Scans the script for prohibited words and assesses general advertiser-friendliness.
    """
    blocked_words = policy_config.get("blocked_words", [])
    sensitive_topics = policy_config.get("sensitive_topics", [])
    flags = []

    # 1. Rule-based regex scan for quick safety
    text_to_scan = f"{script.hook} " + " ".join(s.text for s in script.sections) + f" {script.cta} {script.outro}"
    found_words = []
    for word in blocked_words:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        if pattern.search(text_to_scan):
            found_words.append(word)

    if found_words:
        flags.append(
            f"Blocked Word Match: Found advertiser-unfriendly terms: {', '.join(found_words)}. "
            "It is highly recommended to replace these with safe synonyms."
        )

    # 2. Contextual LLM policy review
    user_prompt = render_template(
        "monetization.j2",
        blocked_words=blocked_words,
        sensitive_topics=sensitive_topics,
        script=script
    )
    system_prompt = (
        "You are an advertiser-safety compliance inspector for digital media. Review "
        "the provided video script and determine if it complies with advertiser-friendly "
        "content policies. Flag graphic topics, hate, or self-harm."
    )

    try:
        eval_res = llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LLMMonetizationEval
        )
        llm_status = eval_res.status
        llm_issues = eval_res.issues
    except Exception as e:
        # Fallback to local rule status if LLM check encounters errors
        llm_status = "Safe"
        llm_issues = [f"System: Could not run LLM monetization check. Error: {str(e)}"]

    # Combine flags and status
    all_flags = flags + llm_issues
    status_priority = {"Flagged": 3, "Warning": 2, "Safe": 1}
    
    rule_status = "Safe"
    if len(found_words) > 3:
        rule_status = "Flagged"
    elif len(found_words) > 0:
        rule_status = "Warning"

    final_priority = max(status_priority.get(rule_status, 1), status_priority.get(llm_status, 1))
    final_status = [k for k, v in status_priority.items() if v == final_priority][0]

    return final_status, all_flags
