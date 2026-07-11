import os
import re
import uuid
import json
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from ytscript.config import load_app_config, get_llm_client, BASE_DIR
from ytscript.models import ScriptRequest, SeriesBible, EpisodeOutline, ChannelAnalysisReport
from ytscript.pipeline.orchestrator import run_pipeline
from ytscript.pipeline.series_manager import generate_series_bible, generate_series_episode
from ytscript.pipeline.planner import run_channel_planner

app = FastAPI(title="AI YouTube Script Generator API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores for background tasks
tasks_store: Dict[str, Dict[str, Any]] = {}
series_tasks_store: Dict[str, Dict[str, Any]] = {}
planner_tasks_store: Dict[str, Dict[str, Any]] = {}

# Ensure static files directory exists
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="<h3>Web UI index.html not found. Place static UI files under src/ytscript/static/</h3>",
            status_code=404
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# =====================================================================
# SINGLE SCRIPT ENDPOINTS
# =====================================================================

def execute_pipeline_task(task_id: str, request: ScriptRequest, provider: str, model: str):
    def progress_callback(stage: int, message: str):
        tasks_store[task_id]["stage"] = stage
        tasks_store[task_id]["log"] = message

    try:
        config = load_app_config()
        llm = get_llm_client(provider, model)
        output = run_pipeline(llm, request, config, status_callback=progress_callback)
        
        out_path = BASE_DIR / "output"
        out_path.mkdir(exist_ok=True)
        
        safe_topic = "".join(c if c.isalnum() else "_" for c in request.topic[:25]).strip("_")
        script_filename = f"script_{safe_topic}.md"
        report_filename = f"report_{safe_topic}.md"
        
        script_file = out_path / script_filename
        report_file = out_path / report_filename

        # Write Script file
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(f"# YouTube Script: {request.topic}\n\n")
            f.write(f"**Niche:** {request.niche} | **Target Duration:** {request.duration_minutes} mins | **Tone:** {request.tone}\n\n")
            f.write(f"## Hook\n\n> {output.script.hook}\n\n")
            f.write("## Body Narration & Cues\n\n")
            for i, section in enumerate(output.script.sections, 1):
                f.write(f"### Segment {i}: {section.title}\n\n")
                f.write(f"**Narration:**\n{section.text}\n\n")
                f.write(f"- 🎬 *Visual Cue:* {section.visual_cues}\n")
                f.write(f"- 🎵 *Audio Cue:* {section.audio_cues}\n")
                f.write(f"- ⏱️ *Duration:* {section.estimated_duration_seconds} seconds\n\n")
            f.write(f"## Call To Action (CTA)\n\n> {output.script.cta}\n\n")
            f.write(f"## Outro\n\n> {output.script.outro}\n")

        # Write Compliance & SEO Report file
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# Compliance & Growth Report: {request.topic}\n\n")
            f.write(f"## Monetization Status: **{output.compliance.monetization_status}**\n")
            f.write(f"## Copyright Status: **{output.compliance.copyright_status}**\n")
            f.write(f"## Retention Score: **{output.compliance.retention_score}/100**\n\n")
            
            f.write("### Safety & Compliance Flags\n")
            if output.compliance.flags:
                for flag in output.compliance.flags:
                    f.write(f"- ⚠️ {flag}\n")
            else:
                f.write("- ✅ No safety flags raised.\n")
            f.write("\n")
            
            f.write("### Royalty-Free Asset Suggestions\n")
            for item in output.compliance.safe_assets_checklist:
                f.write(f"- [ ] {item}\n")
            f.write("\n")
            
            f.write("### Engagement & Retention Tips\n")
            for tip in output.compliance.retention_suggestions:
                f.write(f"- 💡 {tip}\n")
            f.write("\n")
            
            f.write("### YouTube SEO Metadata Package\n\n")
            f.write("#### Title Suggestions:\n")
            for i, opt in enumerate(output.seo.title_options, 1):
                f.write(f"{i}. **{opt}**\n")
            f.write("\n")
            
            f.write(f"#### Tags/Keywords:\n`{', '.join(output.seo.tags)}`\n\n")
            
            f.write("#### Chapter Timestamps:\n")
            for chap in output.seo.chapters:
                f.write(f"- {chap}\n")
            f.write("\n")
            
            f.write("#### Optimized Video Description:\n")
            f.write("```text\n")
            f.write(f"{output.seo.description}\n")
            f.write("```\n")

        tasks_store[task_id].update({
            "status": "completed",
            "stage": 6,
            "log": "Generation complete. Saved script and compliance reports successfully.",
            "script_file": script_filename,
            "report_file": report_filename
        })

    except Exception as e:
        tasks_store[task_id].update({
            "status": "failed",
            "log": f"Failure occurred: {str(e)}",
            "error": str(e)
        })

@app.post("/api/generate")
def generate_script_endpoint(
    background_tasks: BackgroundTasks,
    payload: Dict[str, Any] = Body(...)
):
    topic = payload.get("topic")
    if not topic:
        raise HTTPException(status_code=400, detail="Missing required field 'topic'")
        
    niche = payload.get("niche", "tech")
    tone = payload.get("tone", "educational")
    duration = float(payload.get("duration", 5.0))
    audience = payload.get("audience", "general")
    provider = payload.get("provider", "mock")
    model = payload.get("model")
    voice_sample = payload.get("voice_sample")
    references = payload.get("references", [])

    request = ScriptRequest(
        topic=topic,
        niche=niche,
        duration_minutes=duration,
        tone=tone,
        audience=audience,
        references=references,
        voice_sample=voice_sample
    )

    task_id = str(uuid.uuid4())
    tasks_store[task_id] = {
        "status": "pending",
        "stage": 0,
        "log": "Task registered, awaiting execution...",
        "topic": topic,
        "provider": provider
    }

    background_tasks.add_task(execute_pipeline_task, task_id, request, provider, model)
    return {"task_id": task_id, "status": "pending"}

@app.get("/api/tasks/{task_id}")
def get_task_status(task_id: str):
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_store[task_id]

@app.get("/api/history")
def get_history():
    out_dir = BASE_DIR / "output"
    if not out_dir.exists():
        return []
        
    scripts = []
    for filepath in out_dir.glob("script_*.md"):
        safe_name = filepath.name.replace("script_", "").replace(".md", "")
        report_file = out_dir / f"report_{safe_name}.md"
        
        topic = safe_name.replace("_", " ")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                line = f.readline()
                if line.startswith("# YouTube Script:"):
                    topic = line.replace("# YouTube Script:", "").strip()
        except Exception:
            pass

        stat = filepath.stat()
        scripts.append({
            "id": safe_name,
            "topic": topic,
            "created_at": stat.st_mtime,
            "script_file": filepath.name,
            "report_file": report_file.name if report_file.exists() else None
        })
        
    scripts.sort(key=lambda x: x["created_at"], reverse=True)
    return scripts

@app.get("/api/history/files/{filename}")
def get_history_file(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    filepath = BASE_DIR / "output" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history/{id}")
def delete_history_item(id: str):
    if ".." in id or "/" in id or "\\" in id:
        raise HTTPException(status_code=400, detail="Invalid ID")
        
    out_dir = BASE_DIR / "output"
    script_file = out_dir / f"script_{id}.md"
    report_file = out_dir / f"report_{id}.md"
    
    deleted_any = False
    if script_file.exists():
        script_file.unlink()
        deleted_any = True
    if report_file.exists():
        report_file.unlink()
        deleted_any = True
        
    if not deleted_any:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"status": "deleted", "id": id}

# =====================================================================
# SERIES MODE ENDPOINTS
# =====================================================================

def execute_series_pipeline_task(
    task_id: str,
    title: str,
    niche: str,
    tone: str,
    series_arc: str,
    num_episodes: int,
    provider: str,
    model: str
):
    try:
        config = load_app_config()
        llm = get_llm_client(provider, model)
        
        # 1. Generate Series Bible
        series_tasks_store[task_id].update({
            "status": "processing",
            "stage": 0,
            "log": "Generating Series Bible outline and episode plan..."
        })
        
        bible = generate_series_bible(llm, title, niche, tone, series_arc, num_episodes)
        
        # Ensure output directories exist
        series_dir = BASE_DIR / "output" / f"series_{bible.series_id}"
        series_dir.mkdir(parents=True, exist_ok=True)
        
        # Save bible outline to disk
        bible_path = series_dir / "bible.json"
        with open(bible_path, "w", encoding="utf-8") as f:
            json.dump(bible.model_dump(), f, indent=4)
            
        series_tasks_store[task_id]["series_id"] = bible.series_id
        
        # 2. Sequentially write episode scripts
        episodes_completed = []
        
        for idx, ep in enumerate(bible.episodes):
            ep_num = ep.episode_number
            next_ep = bible.episodes[idx + 1] if idx + 1 < len(bible.episodes) else None
            
            series_tasks_store[task_id].update({
                "stage": ep_num,
                "log": f"Generating Episode {ep_num} of {num_episodes}: '{ep.title}'..."
            })
            
            # Execute generation pipeline
            output = generate_series_episode(llm, bible, ep, next_ep, config)
            
            # Save files
            script_filename = f"episode_{ep_num}_script.md"
            report_filename = f"episode_{ep_num}_report.md"
            
            # Write Script Markdown
            with open(series_dir / script_filename, "w", encoding="utf-8") as f:
                f.write(f"# {ep.title}\n\n")
                f.write(f"**Series:** {title} | **Episode:** {ep_num} of {num_episodes}\n")
                f.write(f"**Focus Topic:** {ep.focus_topic}\n\n")
                f.write(f"## Hook\n\n> {output.script.hook}\n\n")
                f.write("## Body Narration & Cues\n\n")
                for s_idx, sec in enumerate(output.script.sections, 1):
                    f.write(f"### Segment {s_idx}: {sec.title}\n\n")
                    f.write(f"**Narration:**\n{sec.text}\n\n")
                    f.write(f"- 🎬 *Visual Cue:* {sec.visual_cues}\n")
                    f.write(f"- 🎵 *Audio Cue:* {sec.audio_cues}\n")
                    f.write(f"- ⏱️ *Duration:* {sec.estimated_duration_seconds} seconds\n\n")
                f.write(f"## Call To Action (CTA)\n\n> {output.script.cta}\n\n")
                f.write(f"## Outro\n\n> {output.script.outro}\n")
                
            # Write Report Markdown
            with open(series_dir / report_filename, "w", encoding="utf-8") as f:
                f.write(f"# Compliance & Growth Report: {ep.title}\n\n")
                f.write(f"## Monetization Status: **{output.compliance.monetization_status}**\n")
                f.write(f"## Copyright Status: **{output.compliance.copyright_status}**\n")
                f.write(f"## Retention Score: **{output.compliance.retention_score}/100**\n\n")
                
                f.write("### Safety & Compliance Flags\n")
                if output.compliance.flags:
                    for flag in output.compliance.flags:
                        f.write(f"- ⚠️ {flag}\n")
                else:
                    f.write("- ✅ No safety flags raised.\n")
                f.write("\n")
                
                f.write("### Royalty-Free Asset Suggestions\n")
                for item in output.compliance.safe_assets_checklist:
                    f.write(f"- [ ] {item}\n")
                f.write("\n")
                
                f.write("### Engagement & Retention Tips\n")
                for tip in output.compliance.retention_suggestions:
                    f.write(f"- 💡 {tip}\n")
                f.write("\n")
                
                f.write("### YouTube SEO Metadata Package\n\n")
                f.write("#### Title Suggestions:\n")
                for i, opt in enumerate(output.seo.title_options, 1):
                    f.write(f"{i}. **{opt}**\n")
                f.write("\n")
                
                f.write(f"#### Tags/Keywords:\n`{', '.join(output.seo.tags)}`\n\n")
                
                f.write("#### Chapter Timestamps:\n")
                for chap in output.seo.chapters:
                    f.write(f"- {chap}\n")
                f.write("\n")
                
                f.write("#### Optimized Video Description:\n")
                f.write("```text\n")
                f.write(f"{output.seo.description}\n")
                f.write("```\n")

            # Update Bible Continuity notes for the next episode
            summary_note = f"Episode {ep_num} ('{ep.title}') established: {', '.join(ep.key_takeaways)}."
            bible.continuity_notes.append(summary_note)
            
            # Save updated bible back to disk
            with open(bible_path, "w", encoding="utf-8") as f:
                json.dump(bible.model_dump(), f, indent=4)
                
            episodes_completed.append({
                "episode_number": ep_num,
                "title": ep.title,
                "script_file": script_filename,
                "report_file": report_filename
            })

        series_tasks_store[task_id].update({
            "status": "completed",
            "stage": num_episodes,
            "log": f"Successfully completed all {num_episodes} episodes of series '{title}'!",
            "episodes": episodes_completed
        })

    except Exception as e:
        series_tasks_store[task_id].update({
            "status": "failed",
            "log": f"Series generation failure: {str(e)}",
            "error": str(e)
        })

@app.post("/api/series/generate")
def generate_series_endpoint(
    background_tasks: BackgroundTasks,
    payload: Dict[str, Any] = Body(...)
):
    title = payload.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="Missing required field 'title'")
        
    niche = payload.get("niche", "tech")
    tone = payload.get("tone", "educational")
    series_arc = payload.get("series_arc", "")
    num_episodes = int(payload.get("num_episodes", 3))
    provider = payload.get("provider", "mock")
    model = payload.get("model")

    task_id = str(uuid.uuid4())
    series_tasks_store[task_id] = {
        "status": "pending",
        "stage": 0,
        "log": "Series task registered, awaiting execution...",
        "title": title,
        "num_episodes": num_episodes,
        "provider": provider
    }

    background_tasks.add_task(
        execute_series_pipeline_task,
        task_id, title, niche, tone, series_arc, num_episodes, provider, model
    )
    return {"task_id": task_id, "status": "pending"}

@app.get("/api/series/tasks/{task_id}")
def get_series_task_status(task_id: str):
    if task_id not in series_tasks_store:
        raise HTTPException(status_code=404, detail="Series task not found")
    return series_tasks_store[task_id]

@app.get("/api/series/history")
def get_series_history():
    out_dir = BASE_DIR / "output"
    if not out_dir.exists():
        return []
        
    series_list = []
    for path in out_dir.iterdir():
        if path.is_dir() and path.name.startswith("series_"):
            bible_file = path / "bible.json"
            if bible_file.exists():
                try:
                    with open(bible_file, "r", encoding="utf-8") as f:
                        bible_data = json.load(f)
                    
                    ep_count = len(list(path.glob("episode_*_script.md")))
                    stat = bible_file.stat()
                    
                    series_list.append({
                        "series_id": bible_data.get("series_id"),
                        "title": bible_data.get("title"),
                        "niche": bible_data.get("niche"),
                        "tone": bible_data.get("tone"),
                        "total_episodes": len(bible_data.get("episodes", [])),
                        "completed_episodes": ep_count,
                        "created_at": stat.st_mtime
                    })
                except Exception:
                    pass
                    
    series_list.sort(key=lambda x: x["created_at"], reverse=True)
    return series_list

@app.get("/api/series/history/{series_id}")
def get_series_detail(series_id: str):
    if ".." in series_id or "/" in series_id or "\\" in series_id:
        raise HTTPException(status_code=400, detail="Invalid Series ID")
        
    series_dir = BASE_DIR / "output" / f"series_{series_id}"
    bible_file = series_dir / "bible.json"
    
    if not series_dir.exists() or not bible_file.exists():
        raise HTTPException(status_code=404, detail="Series not found")
        
    try:
        with open(bible_file, "r", encoding="utf-8") as f:
            bible_data = json.load(f)
            
        episodes = []
        for ep in bible_data.get("episodes", []):
            ep_num = ep.get("episode_number")
            script_file = f"episode_{ep_num}_script.md"
            report_file = f"episode_{ep_num}_report.md"
            
            episodes.append({
                "episode_number": ep_num,
                "title": ep.get("title"),
                "focus_topic": ep.get("focus_topic"),
                "key_takeaways": ep.get("key_takeaways"),
                "has_script": (series_dir / script_file).exists(),
                "script_file": script_file,
                "report_file": report_file
            })
            
        return {
            "bible": bible_data,
            "episodes": episodes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/series/history/{series_id}/files/{filename}")
def get_series_file(series_id: str, filename: str):
    if ".." in series_id or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid path inputs")
        
    filepath = BASE_DIR / "output" / f"series_{series_id}" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/series/history/{series_id}")
def delete_series(series_id: str):
    if ".." in series_id or "/" in series_id or "\\" in series_id:
        raise HTTPException(status_code=400, detail="Invalid Series ID")
        
    series_dir = BASE_DIR / "output" / f"series_{series_id}"
    if not series_dir.exists():
        raise HTTPException(status_code=404, detail="Series not found")
        
    try:
        shutil.rmtree(series_dir)
        return {"status": "deleted", "series_id": series_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# CHANNEL PLANNER & AUDITOR ENDPOINTS
# =====================================================================

def execute_planner_task(
    task_id: str,
    channel_url: str,
    niche: str,
    plan_days: int,
    video_urls: List[str],
    provider: str,
    model: str
):
    def progress_callback(stage: int, message: str):
        planner_tasks_store[task_id]["stage"] = stage
        planner_tasks_store[task_id]["log"] = message

    try:
        config = load_app_config()
        llm = get_llm_client(provider, model)
        
        report = run_channel_planner(
            llm=llm,
            channel_url=channel_url,
            niche=niche,
            plan_days=plan_days,
            video_urls=video_urls,
            status_callback=progress_callback
        )
        
        # Save output
        handle_match = re.search(r"@([a-zA-Z0-9_-]+)", channel_url)
        handle = handle_match.group(1) if handle_match else "channel_" + str(uuid.uuid4())[:8]
        planner_id = f"{handle}_{niche}_{plan_days}d"
        
        planner_dir = BASE_DIR / "output" / f"planner_{planner_id}"
        planner_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = planner_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=4)
            
        planner_tasks_store[task_id].update({
            "status": "completed",
            "stage": 5,
            "log": "Growth Plan & Competitor Audit completed successfully!",
            "planner_id": planner_id
        })
    except Exception as e:
        planner_tasks_store[task_id].update({
            "status": "failed",
            "log": f"Growth Plan failure: {str(e)}",
            "error": str(e)
        })

@app.get("/api/planner/trending-recommendations")
def get_trending_recommendations(niche: str = "tech"):
    try:
        from ytscript.pipeline.planner import search_trending_youtube_videos
        videos = search_trending_youtube_videos(niche)
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/planner/generate")
def generate_plan_endpoint(
    background_tasks: BackgroundTasks,
    payload: Dict[str, Any] = Body(...)
):
    channel_url = payload.get("channel_url", "").strip() or "General Niche Creator"
        
    niche = payload.get("niche", "tech")
    plan_days = int(payload.get("plan_days", 30))
    video_urls = payload.get("video_urls", [])
    provider = payload.get("provider", "mock")
    model = payload.get("model")

    task_id = str(uuid.uuid4())
    planner_tasks_store[task_id] = {
        "status": "pending",
        "stage": 0,
        "log": "Planner task registered, awaiting scheduling...",
        "channel_url": channel_url,
        "plan_days": plan_days,
        "provider": provider
    }

    background_tasks.add_task(
        execute_planner_task,
        task_id, channel_url, niche, plan_days, video_urls, provider, model
    )
    return {"task_id": task_id, "status": "pending"}

@app.get("/api/planner/tasks/{task_id}")
def get_planner_task_status(task_id: str):
    if task_id not in planner_tasks_store:
        raise HTTPException(status_code=404, detail="Planner task not found")
    return planner_tasks_store[task_id]

@app.get("/api/planner/history")
def get_planner_history():
    out_dir = BASE_DIR / "output"
    if not out_dir.exists():
        return []
        
    planners = []
    for path in out_dir.iterdir():
        if path.is_dir() and path.name.startswith("planner_"):
            report_file = path / "report.json"
            if report_file.exists():
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    stat = report_file.stat()
                    
                    planners.append({
                        "planner_id": path.name.replace("planner_", ""),
                        "channel_handle": data.get("channel_handle"),
                        "niche": data.get("niche"),
                        "plan_days": data.get("plan_days"),
                        "created_at": stat.st_mtime
                    })
                except Exception:
                    pass
    planners.sort(key=lambda x: x["created_at"], reverse=True)
    return planners

@app.get("/api/planner/history/{planner_id}")
def get_planner_detail(planner_id: str):
    if ".." in planner_id or "/" in planner_id or "\\" in planner_id:
        raise HTTPException(status_code=400, detail="Invalid planner ID")
        
    report_file = BASE_DIR / "output" / f"planner_{planner_id}" / "report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Planner not found")
        
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/planner/history/{planner_id}")
def delete_planner(planner_id: str):
    if ".." in planner_id or "/" in planner_id or "\\" in planner_id:
        raise HTTPException(status_code=400, detail="Invalid planner ID")
        
    planner_dir = BASE_DIR / "output" / f"planner_{planner_id}"
    if not planner_dir.exists():
        raise HTTPException(status_code=404, detail="Planner not found")
        
    try:
        shutil.rmtree(planner_dir)
        return {"status": "deleted", "planner_id": planner_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# CONFIG ENDPOINTS
# =====================================================================

@app.get("/api/config")
def get_api_config():
    gemini_keys = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
    openai_keys = os.getenv("OPENAI_API_KEYS", os.getenv("OPENAI_API_KEY", ""))
    
    def mask_keys(keys_str: str) -> List[str]:
        masked = []
        for key in keys_str.split(","):
            key = key.strip()
            if not key:
                continue
            if len(key) > 10:
                masked.append(f"{key[:5]}...{key[-5:]}")
            else:
                masked.append("...")
        return masked

    return {
        "gemini_keys_masked": mask_keys(gemini_keys),
        "openai_keys_masked": mask_keys(openai_keys),
        "gemini_key_count": len([k for k in gemini_keys.split(",") if k.strip()]),
        "openai_key_count": len([k for k in openai_keys.split(",") if k.strip()])
    }

@app.post("/api/config")
def save_api_config(payload: Dict[str, str] = Body(...)):
    gemini_keys = payload.get("gemini_keys", "").strip()
    openai_keys = payload.get("openai_keys", "").strip()
    
    env_path = BASE_DIR / ".env"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("# Gemini API Keys (comma-separated list for automatic rotation)\n")
        f.write(f"GEMINI_API_KEYS={gemini_keys}\n\n")
        f.write("# OpenAI API Keys (comma-separated list for automatic rotation)\n")
        f.write(f"OPENAI_API_KEYS={openai_keys}\n")
        
    if gemini_keys:
        os.environ["GEMINI_API_KEYS"] = gemini_keys
    if openai_keys:
        os.environ["OPENAI_API_KEYS"] = openai_keys
        
    return {"status": "saved", "message": "API keys updated successfully."}
