import time
from typing import Type, TypeVar
from pydantic import BaseModel
from ytscript.llm.base import BaseLLMClient
from ytscript.models import Script, ScriptSection, SEOData, ComplianceReport, SeriesBible, EpisodeOutline, VideoAudit, CalendarEvent, ChannelAnalysisReport

T = TypeVar("T", bound=BaseModel)

class MockLLMClient(BaseLLMClient):
    """
    Mock LLM Client that generates deterministic, topic-related placeholder data
    without making external API requests. Useful for testing the orchestration pipeline.
    """
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        time.sleep(0.2)  # Simulate small latency
        return f"Mock response for topic. Prompts received:\nSystem: {system_prompt[:60]}...\nUser: {user_prompt[:60]}..."

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.7
    ) -> T:
        time.sleep(0.5)  # Simulate small latency
        class_name = response_model.__name__
        
        # Extract topic from user prompt if possible
        topic = "YouTube Content Creation"
        for line in user_prompt.split("\n"):
            if "topic:" in line.lower() or "topic =" in line.lower():
                parts = line.split(":", 1) if ":" in line else line.split("=", 1)
                topic = parts[1].strip().strip('"').strip("'")
                break

        if class_name == "Script":
            sections = [
                ScriptSection(
                    title="The Core Problem",
                    text=f"Have you ever wondered why generating scripts for {topic} is so hard? Most creators fail because they skip research and jump straight into writing. Today, we break down the exact solution.",
                    visual_cues="Close up of creator looking frustrated, transitioning to sleek motion graphics showing statistics.",
                    audio_cues="Soft background synth music starts fading in, setting an intriguing tone.",
                    estimated_duration_seconds=30.0
                ),
                ScriptSection(
                    title="Step-by-Step Breakdown",
                    text=f"The secret to mastering {topic} lies in three simple steps. First, research deeply to find the angle. Second, write with intense focus on the first 10 seconds. Third, verify for copyright and advertiser friendliness.",
                    visual_cues="Title cards showing '1. Research', '2. Hook Strength', '3. Safe Checklist'. Overlay bullet points.",
                    audio_cues="Whoosh sound effect on transitions. Upbeat synth music continues.",
                    estimated_duration_seconds=60.0
                ),
                ScriptSection(
                    title="Actionable Takeaways",
                    text=f"If you apply these rules, your next video on {topic} will instantly see higher click-through-rates and viewer retention. Don't skip the validation phase.",
                    visual_cues="Screenshot of YouTube Analytics showing retention spikes. Camera returns to talking head.",
                    audio_cues="Upbeat background music swells slightly.",
                    estimated_duration_seconds=45.0
                )
            ]
            return response_model(
                hook=f"Wait, before you scroll, did you know that {topic} is changing forever in 2026? Most channels will ignore this, but here's what they aren't telling you.",
                sections=sections,
                cta="If you found this breakdown helpful, hit that subscribe button and leave a comment with your thoughts!",
                outro=f"Thanks for watching this guide on {topic}, make sure to check out our other playlists, and I will see you in the next video right here.",
                word_count=250,
                estimated_duration_seconds=155.0
            )
        elif class_name == "SEOData":
            return response_model(
                title_options=[
                    f"The Ultimate Guide to {topic} (2026)",
                    f"Why You're Failing at {topic} (And How to Fix It)",
                    f"I Tried {topic} for 30 Days (Crazy Results)",
                    f"5 Secrets of {topic} Nobody Tells You",
                    f"How to Master {topic} in Under 5 Minutes"
                ],
                description=f"This video is the complete guide to {topic}. Learn the top tips, tricks, and strategies to stand out today.\n\nTimestamps:\n00:00 - Introduction\n00:30 - The Core Problem\n01:00 - Step-by-Step Breakdown\n02:00 - Actionable Takeaways\n02:35 - Outro",
                tags=[topic.lower(), "youtube growth", "content creator", "tutorial", "2026 guide"],
                chapters=[
                    "00:00 - Introduction",
                    "00:30 - The Core Problem",
                    "01:00 - Step-by-Step Breakdown",
                    "02:00 - Actionable Takeaways",
                    "02:35 - Outro"
                ]
            )
        elif class_name == "ComplianceReport":
            return response_model(
                monetization_status="Safe",
                copyright_status="Safe",
                flags=[],
                safe_assets_checklist=[
                    "Background Music: Use royalty-free track from YouTube Audio Library.",
                    "Footage: Add Pexels/Pixabay clips for the Step-by-Step breakdown.",
                    "Overlays: Use custom text graphics for key terms."
                ],
                retention_score=85,
                retention_suggestions=[
                    "Add a visual pattern interrupt at 0:15 to keep interest high.",
                    "Shorten the hook slightly if speaking rate is too slow."
                ]
            )
        elif class_name == "LLMMonetizationEval":
            return response_model(
                status="Safe",
                issues=[]
            )
        elif class_name == "LLMRetentionEval":
            return response_model(
                retention_score=85,
                retention_suggestions=[
                    "Add a visual pattern interrupt at 0:15 to keep interest high.",
                    "Shorten the hook slightly if speaking rate is too slow."
                ]
            )
        elif class_name == "SeriesBible":
            episodes = [
                EpisodeOutline(
                    episode_number=1,
                    title="Episode 1: The Core Fundamentals of " + topic,
                    focus_topic="Understanding the basic syntax, setup, and key concepts of " + topic + ".",
                    key_takeaways=["Basic concepts of " + topic, "Setting up workspace", "Running first code script"]
                ),
                EpisodeOutline(
                    episode_number=2,
                    title="Episode 2: Advanced Techniques in " + topic,
                    focus_topic="Digging deeper into performance optimization, B-roll pacing, and safety checks.",
                    key_takeaways=["Performance metrics", "Visual pattern interrupts", "Fair-use checks"]
                ),
                EpisodeOutline(
                    episode_number=3,
                    title="Episode 3: Scaling & Monetizing " + topic,
                    focus_topic="Deploying your knowledge to build real-world systems and earn passive income.",
                    key_takeaways=["Building FastAPI routes", "Stripe payment configs", "Publishing guides"]
                )
            ]
            return response_model(
                series_id="mock_series_" + "".join(c if c.isalnum() else "_" for c in topic[:15]).strip("_").lower(),
                title=f"Mastering {topic}",
                niche="tech",
                tone="educational",
                series_arc=f"A complete step-by-step masterclass on {topic}.",
                episodes=episodes,
                continuity_notes=["Initialized Series Bible outline."]
            )
        elif class_name == "ChannelAnalysisReport":
            # Determine days requested
            plan_days = 30
            if "60" in user_prompt:
                plan_days = 60
            
            audited_videos = [
                VideoAudit(
                    video_id="dQw4w9WgXcQ",
                    title="How to automate your entire life with AI in 2026",
                    delivery_breakdown="High energy, fast pacing, visual pattern interrupts every 3 seconds. Uses zoom-ins and screen recording walkthroughs.",
                    content_delivered="Introduces 5 specific low-code automations, tools like Make and Zapier, and time-saving metrics.",
                    hooks_used=["Curiosity gap: 'This tool does 90% of my work'", "Loss aversion: 'You are losing 10 hours a week'"]
                ),
                VideoAudit(
                    video_id="dQw4w9WgXcR",
                    title="The truth about YouTube automation channels (Honest review)",
                    delivery_breakdown="Direct camera speech, casual living room background, high authenticity, slower pacing with emotional pauses.",
                    content_delivered="Exposes fake profit screenshots of guru courses. Breaks down real margins (15-20%) and hiring costs.",
                    hooks_used=["Contrarian statement: 'YouTube automation is a lie'", "Empathy build: 'I lost $5000 so you don't have to'"]
                )
            ]
            
            calendar_events = []
            formats = ["Long-form Video", "Shorts / Reel"]
            for d in range(1, plan_days + 1):
                vid_format = formats[d % 2]
                if vid_format == "Long-form Video":
                    title = f"Day {d}: The Ultimate Guide to {topic} secrets"
                    angle = "In-depth explainer with step-by-step visuals"
                else:
                    title = f"Day {d}: Stop doing {topic} this way!"
                    angle = "High-energy reaction / contrarian hook"
                    
                calendar_events.append(
                    CalendarEvent(
                        day=d,
                        title=title,
                        format=vid_format,
                        angle=angle,
                        niche_gap_reason=f"Competitors cover basic {topic} but omit this specific angle, creating a search traffic gap."
                    )
                )
                
            return response_model(
                channel_handle="@creator_handle",
                niche="tech",
                audited_videos=audited_videos,
                niche_gaps=[
                    "Lack of honest, no-fluff case studies with real profit/loss figures.",
                    f"Over-saturation of beginner tutorials for {topic} but zero advanced workflow scaling guides."
                ],
                competitor_trends=[
                    "High usage of dopamine-heavy retention editing to mask shallow research.",
                    "Clickbait titles promising instant wealth without showing operational complexity."
                ],
                opportunities=[
                    "Create deep-dive technical tutorials showing exact API payloads.",
                    "Build a transparent review series showing your real dashboard numbers."
                ],
                plan_days=plan_days,
                content_calendar=calendar_events
            )
        else:
            raise ValueError(f"Unknown response model class: {class_name}")
