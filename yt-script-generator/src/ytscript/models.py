from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ScriptRequest(BaseModel):
    topic: str = Field(..., description="The main topic of the video")
    niche: str = Field(..., description="The target niche (e.g., tech, finance, health, etc.)")
    duration_minutes: float = Field(default=5.0, description="Target duration of the script in minutes")
    tone: str = Field(default="educational", description="The tone of the video (e.g., educational, dramatic, energetic)")
    audience: str = Field(default="general", description="Target audience description")
    references: List[str] = Field(default_factory=list, description="Optional URLs or text references to ground facts")
    voice_sample: Optional[str] = Field(None, description="Optional text representing the creator's past voice style")

class ScriptSection(BaseModel):
    title: str = Field(..., description="Section title or heading")
    text: str = Field(..., description="Narration script text for the creator to speak")
    visual_cues: str = Field(..., description="Visual instructions, B-roll ideas, or text overlays")
    audio_cues: str = Field(..., description="Sound effects, background music instructions, or voice inflection tips")
    estimated_duration_seconds: float = Field(..., description="Estimated duration to read/present this section")

class Script(BaseModel):
    hook: str = Field(..., description="The high-retention opening line or hook (first 10-15 seconds)")
    sections: List[ScriptSection] = Field(..., description="The chronological body sections of the script")
    cta: str = Field(..., description="Call to action (subscribe, comment, check link)")
    outro: str = Field(..., description="Closing sign-off and recommendation for the next video")
    word_count: int = Field(..., description="Total word count of narration text")
    estimated_duration_seconds: float = Field(..., description="Total estimated speaking duration in seconds")

class SEOData(BaseModel):
    title_options: List[str] = Field(..., description="5 high-CTR title suggestions for the video")
    description: str = Field(..., description="Optimized video description incorporating keywords and chapters")
    tags: List[str] = Field(..., description="Relevant tags/keywords for YouTube metadata")
    chapters: List[str] = Field(..., description="Timestamps and headings for YouTube chapters (e.g. 00:00 - Introduction)")

class ComplianceReport(BaseModel):
    monetization_status: str = Field(..., description="Advertiser-friendliness status: 'Safe', 'Warning', or 'Flagged'")
    copyright_status: str = Field(..., description="Copyright/originality status: 'Safe', 'Warning', or 'Flagged'")
    flags: List[str] = Field(..., description="Specific policy violations, profanity occurrences, or plagiarized parts flagged")
    safe_assets_checklist: List[str] = Field(..., description="List of safe royalty-free assets recommendations (music, B-roll)")
    retention_score: int = Field(..., description="Estimated audience retention rating (1-100) based on hook, pacing, and CTAs")
    retention_suggestions: List[str] = Field(..., description="Actionable ideas to improve pacing or keep viewers watching longer")

class PipelineOutput(BaseModel):
    request: ScriptRequest
    script: Script
    seo: SEOData
    compliance: ComplianceReport

class EpisodeOutline(BaseModel):
    episode_number: int = Field(..., description="Chronological episode number (e.g. 1)")
    title: str = Field(..., description="Episode title")
    focus_topic: str = Field(..., description="Key concept or focus topic of the episode")
    key_takeaways: List[str] = Field(..., description="List of core points or takeaways to cover")

class SeriesBible(BaseModel):
    series_id: str = Field(..., description="Unique slug or ID for this series")
    title: str = Field(..., description="Series title")
    niche: str = Field(..., description="General niche")
    tone: str = Field(..., description="Tone for the series")
    series_arc: str = Field(..., description="Overall narrative or learning arc description")
    episodes: List[EpisodeOutline] = Field(..., description="List of planned episodes")
    continuity_notes: List[str] = Field(default_factory=list, description="Log of facts/lore introduced in previous episodes")

class VideoAudit(BaseModel):
    video_id: str = Field(..., description="The YouTube video ID")
    title: str = Field(..., description="The audited video title")
    delivery_breakdown: str = Field(..., description="Detailed breakdown of how the content is delivered (pacing, style)")
    content_delivered: str = Field(..., description="Summary of the main concepts/value delivered")
    hooks_used: List[str] = Field(..., description="Specific hook techniques identified")

class CalendarEvent(BaseModel):
    day: int = Field(..., description="Day number (e.g. 1 to 30)")
    title: str = Field(..., description="Proposed video title")
    format: str = Field(..., description="Video format: 'Long-form Video' or 'Shorts / Reel'")
    angle: str = Field(..., description="Hook angle or theme (e.g., explainer, reaction, gap-exploit)")
    niche_gap_reason: str = Field(..., description="Why this topic was selected and why competitors lack it")

class ChannelAnalysisReport(BaseModel):
    channel_handle: str = Field(..., description="Audited YouTube channel handle")
    niche: str = Field(..., description="General channel niche")
    audited_videos: List[VideoAudit] = Field(default_factory=list, description="Audit reports for the transcribed competitor videos")
    niche_gaps: List[str] = Field(..., description="List of unmet needs and topics in this niche")
    competitor_trends: List[str] = Field(..., description="Core strategies and patterns observed in competitor videos")
    opportunities: List[str] = Field(..., description="Actionable growth opportunities")
    plan_days: int = Field(..., description="Number of days in the content plan (30 or 60)")
    content_calendar: List[CalendarEvent] = Field(..., description="Day-by-day content calendar outline")
