# AI YouTube Script Generator

A monetization-safe, copyright-safe, audience-attracting AI YouTube script generator. This tool turns a topic (or series outline) into ready-to-record YouTube scripts delivered with a compliance and SEO growth report.

---

## 📋 Features

- **Single Script Generation**: Topic + niche + duration + tone → full script.
- **Retention-Optimized Structure**: Hook → intro → body → CTA → outro with detailed visual/audio cues.
- **Monetization Policy Screening**: Combines fast local word scans with LLM advertiser-policy safety screening.
- **Copyright & Originality Guard**: Enforces original expression, flags usage of third-party clips, and generates a royalty-free asset licensing checklist.
- **SEO Package Generator**: Click-worthy title options, optimized description, chapters/timestamps, and search tags.
- **Provider-Agnostic client**: Adapter pattern supporting OpenAI (`gpt-4o`/`gpt-4o-mini`), Google Gemini (`gemini-2.5-flash`), and an offline `mock` client for API-free testing.

---

## ⚙️ Installation & Setup

1. **Clone/Navigate** to the project directory:
   ```bash
   cd d:/Voice Enhancer/yt-script-generator
   ```

2. **Install dependencies** (virtual environment recommended):
   ```bash
   pip install -r requirements.txt
   ```
   Or install in editable mode:
   ```bash
   pip install -e .
   ```

3. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   Provide:
   - `OPENAI_API_KEY` (if using `--provider openai`)
   - `GEMINI_API_KEY` (if using `--provider gemini`)

---

## 🚀 How to Run

Generate a script using the offline mock provider:
```bash
python src/ytscript/main.py generate-script --topic "How to Code in 2026" --niche tech --provider mock
```

Generate a script using Google Gemini:
```bash
python src/ytscript/main.py generate-script --topic "5 Mistakes New Programmers Make" --niche tech --provider gemini
```

Generate a script using OpenAI:
```bash
python src/ytscript/main.py generate-script --topic "The Future of AI Video Editing" --niche tech --provider openai
```

---

## 📂 Output

All outputs are saved as Markdown files in the `output/` directory:
- `script_<topic_name>.md`: The word-for-word spoken narration along with visual and audio cues.
- `report_<topic_name>.md`: Safety checks (Monetization & Copyright status, safety flags, resource checklist) and the SEO package (titles, description, tags, chapters).
