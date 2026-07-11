import os
import sys
from pathlib import Path

# Dynamically add the 'src' directory to Python path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from ytscript.config import load_app_config, get_llm_client, BASE_DIR
from ytscript.models import ScriptRequest
from ytscript.pipeline.orchestrator import run_pipeline

app = typer.Typer(
    name="ytscript",
    help="AI YouTube Script Generator - Create monetization-safe & copyright-safe scripts."
)
console = Console()

@app.command()
def generate_script(
    topic: str = typer.Option(..., "--topic", "-t", help="Video topic or outline summary"),
    niche: str = typer.Option("tech", "--niche", "-n", help="Niche for the video (e.g., finance, comedy)"),
    tone: str = typer.Option("educational", "--tone", "-s", help="Voice tone of the narrator"),
    duration: float = typer.Option(5.0, "--duration", "-d", help="Target duration in minutes"),
    audience: str = typer.Option("general", "--audience", "-a", help="Description of target audience"),
    provider: str = typer.Option("mock", "--provider", "-p", help="LLM Provider: openai, gemini, or mock"),
    model: str = typer.Option(None, "--model", "-m", help="Custom LLM model name (e.g. gpt-4o)"),
    voice_path: str = typer.Option(None, "--voice", "-v", help="Path to text file containing creator voice sample"),
    references: list[str] = typer.Option(None, "--ref", "-r", help="URLs or local text files for grounding facts"),
    output_dir: str = typer.Option("output", "--out", "-o", help="Directory where scripts and reports are saved")
):
    """
    Generate a full YouTube script and compliance report for a given topic.
    """
    console.print("[bold green]Starting AI YouTube Script Generator...[/bold green]")
    console.print(f"[dim]Topic:[/dim] {topic}")
    console.print(f"[dim]LLM Provider:[/dim] {provider}")
    
    # Resolve reference text inputs
    resolved_refs = []
    if references:
        for ref in references:
            if os.path.exists(ref):
                with open(ref, "r", encoding="utf-8") as f:
                    resolved_refs.append(f.read())
            else:
                resolved_refs.append(ref)
                
    # Resolve voice styling sample
    voice_sample = None
    if voice_path:
        if os.path.exists(voice_path):
            with open(voice_path, "r", encoding="utf-8") as f:
                voice_sample = f.read()
        else:
            voice_sample = voice_path

    # Build Pydantic request object
    request = ScriptRequest(
        topic=topic,
        niche=niche,
        duration_minutes=duration,
        tone=tone,
        audience=audience,
        references=resolved_refs,
        voice_sample=voice_sample
    )

    try:
        # Load app rules configuration
        config = load_app_config()
        
        # Instantiate correct LLM client
        llm = get_llm_client(provider, model)
        
        console.print("[yellow]Running multi-stage pipeline...[/yellow]")
        output = run_pipeline(llm, request, config)
        
        # Ensure output directory exists
        out_path = BASE_DIR / output_dir
        out_path.mkdir(exist_ok=True)
        
        # Format a clean filename from the topic
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic[:25]).strip("_")
        
        # Write Script Markdown
        script_file = out_path / f"script_{safe_topic}.md"
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(f"# YouTube Script: {topic}\n\n")
            f.write(f"**Niche:** {niche} | **Target Duration:** {duration} mins | **Tone:** {tone}\n\n")
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
            
        # Write Compliance & SEO Report Markdown
        report_file = out_path / f"report_{safe_topic}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# Compliance & Growth Report: {topic}\n\n")
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

        console.print("[bold green]Generation complete![/bold green]")
        console.print(f"Script saved to: [cyan]{script_file}[/cyan]")
        console.print(f"Compliance/SEO report saved to: [cyan]{report_file}[/cyan]")
        
    except Exception as e:
        console.print(f"[bold red]Pipeline Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
