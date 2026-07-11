from typing import List, Dict, Any, Tuple
from ytscript.models import Script

def run_copyright_check(
    script: Script,
    copyright_config: Dict[str, Any]
) -> Tuple[str, List[str], List[str]]:
    """
    Stage 4: Copyright & Originality Guard.
    Ensures that script narration is transformative and provides a curated
    royalty-free asset checklist to prevent copyright strikes on the channel.
    
    Returns:
        copyright_status: 'Safe' or 'Warning'
        safe_assets_checklist: list of royalty-free links and suggestions
        copyright_flags: list of potential citation or fair-use warnings
    """
    rules = copyright_config.get("transformative_rules", [])
    resources = copyright_config.get("royalty_free_resources", [])
    
    # 1. Compile the checklist from rules and resources in config.yaml
    safe_assets_checklist = []
    for rule in rules:
        safe_assets_checklist.append(f"Fair Use Guide: {rule}")
        
    for res in resources:
        name = res.get("name")
        res_type = res.get("type")
        url = res.get("url")
        safe_assets_checklist.append(f"Royalty-Free Resource: Use {name} for {res_type} ({url})")
        
    # 2. Local heuristic checks for copyright risks
    copyright_status = "Safe"
    copyright_flags = []
    
    text_to_scan = f"{script.hook} " + " ".join(s.text for s in script.sections) + f" {script.cta} {script.outro}"
    text_lower = text_to_scan.lower()
    
    # Check for potential third-party clip references or lack of attribution
    keywords_of_risk = ["clip from", "as seen in", "song by", "movie clip", "video by"]
    for keyword in keywords_of_risk:
        if keyword in text_lower:
            copyright_status = "Warning"
            copyright_flags.append(
                f"Copyright Check: Script references third-party material ('{keyword}'). "
                "Ensure any external assets are under 5 seconds, heavily commented on, "
                "and fit fair-use guidelines."
            )
            break
            
    return copyright_status, safe_assets_checklist, copyright_flags
