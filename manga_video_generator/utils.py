import os
import streamlit as st
import json
import torch
import shutil
from PIL import Image

from core.custom_generator import generate_custom_image
from core.text_to_video import generate_text_to_video
from core.image_to_video import generate_image_to_video
def render_panel_remix(tab, TEMP_DIR, OUTPUT_DIR, gemini_key, openai_key):
    from pathlib import Path
    from core.generator import generate_scene_image
    from core.panel_remixer import create_panel_remix_scenes
    from core.assembler import assemble_video
    with tab:
    
        st.header("🧩 Panel Remix Studio")
        st.write("Use uploaded manga panels as storyboard references, then generate new original anime-style recap visuals with richer prompts.")
    
        col_panel_in, col_panel_out = st.columns([1, 1])
    
        with col_panel_in:
            st.markdown("<div class='status-card'>", unsafe_allow_html=True)
            st.subheader("📚 Source Panels & Narration")
    
            panel_files = st.file_uploader(
                "Upload 10-20 black-and-white manga panel images (.png, .jpg, .webp)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="panel_remix_uploads"
            )
            panel_voiceover = st.file_uploader(
                "Optional voiceover audio for timing (.mp3, .wav, .m4a, .mpeg)",
                type=["mp3", "wav", "m4a", "mpeg"],
                key="panel_remix_voiceover"
            )
            panel_bgm = st.file_uploader(
                "🎵 Optional background music (.mp3, .wav)",
                type=["mp3", "wav"],
                key="panel_remix_bgm",
                help="Loops automatically to fill the video. Fades out at the end.",
            )
            if panel_bgm:
                panel_bgm_volume = st.slider(
                    "BGM volume (relative to voiceover)", 0.0, 0.5, 0.12, step=0.01,
                    key="panel_remix_bgm_volume"
                )
            else:
                panel_bgm_volume = 0.12
            panel_narration_text = st.text_area(
                "Optional script / narration text (used when no voiceover is uploaded)",
                height=150,
                placeholder="Paste the recap narration here, or upload voiceover audio above.",
                key="panel_remix_narration"
            )

            # Auto-TTS controls — only shown when no voiceover is uploaded
            if panel_voiceover is None:
                from core.tts_manager import OPENAI_VOICES, BARK_PRESETS
                _panel_tts_backend = st.selectbox(
                    "🔊 Auto-generate voiceover engine (from narration text above)",
                    options=["gtts", "openai", "bark"],
                    format_func=lambda x: {
                        "gtts": "gTTS (free, basic quality)",
                        "openai": "OpenAI TTS (tts-1-hd, high quality)",
                        "bark": "Bark (local GPU, very expressive — slow)",
                    }[x],
                    key="panel_remix_tts_backend",
                )
                if _panel_tts_backend == "openai":
                    _panel_tts_voice = st.selectbox(
                        "Voice", OPENAI_VOICES, index=0, key="panel_remix_tts_voice_openai",
                        help="onyx/echo are deep narrative voices.",
                    )
                    st.caption("Uses your OpenAI key. Works with any language text.")
                elif _panel_tts_backend == "bark":
                    _panel_tts_voice = st.selectbox(
                        "Speaker preset", BARK_PRESETS, index=0, key="panel_remix_tts_voice_bark",
                        help="Runs locally on GPU. ~30-60s per sentence. Very expressive.",
                    )
                    st.caption("⚠️ Requires: `pip install git+https://github.com/suno-ai/bark.git scipy`")
                else:
                    _panel_tts_voice = None
            else:
                _panel_tts_backend = "gtts"
                _panel_tts_voice = None

            st.markdown("---")
            st.subheader("🧠 Panel Analysis")
            panel_analyzer_backend_choice = st.selectbox(
                "Panel understanding model",
                options=["Gemini API - Vision", "OpenAI API - Vision"],
                key="panel_remix_analyzer_backend"
            )
            panel_analyzer_backend = "openai" if "OpenAI" in panel_analyzer_backend_choice else "gemini"
            panel_analyzer_key = st.text_input(
                "Panel analysis API key",
                value=gemini_key if panel_analyzer_backend == "gemini" else openai_key,
                type="password",
                help="Gemini key for Gemini Vision, or OpenAI key for OpenAI Vision.",
                key="panel_remix_analyzer_key"
            )
            panel_openai_model = "gpt-4o-mini"
            if panel_analyzer_backend == "openai":
                panel_openai_model = st.text_input(
                    "OpenAI vision model",
                    value="gpt-4o-mini",
                    key="panel_remix_openai_model"
                )
    
            col_panel_detail_a, col_panel_detail_b = st.columns(2)
            with col_panel_detail_a:
                panel_detail_strength = st.selectbox(
                    "Prompt detail strength",
                    options=["Medium", "High", "Ultra"],
                    index=1,
                    key="panel_remix_detail_strength"
                )
            with col_panel_detail_b:
                panel_originality = st.slider(
                    "Originality strength",
                    min_value=60,
                    max_value=100,
                    value=88,
                    step=2,
                    help="Higher values push stronger redesign of faces, hair, outfits, props, background, and framing.",
                    key="panel_remix_originality"
                )
    
            panel_art_style = st.selectbox(
                "Generated art style",
                options=[
                    "anime style, highly detailed digital painting, cinematic lighting, expressive character acting, 16:9 aspect ratio",
                    "dark cinematic manga-inspired anime art, dramatic shadows, inked energy, rich background detail, 16:9 aspect ratio",
                    "premium YouTube anime recap visual, bold focal subject, dramatic color, high-retention composition, 16:9 aspect ratio",
                    "watercolor manga-inspired anime illustration, soft atmospheric lighting, hand-painted texture, 16:9 aspect ratio",
                ],
                index=0,
                key="panel_remix_art_style"
            )
    
            st.markdown("---")
            st.subheader("🎨 Image Generation")
            panel_image_backend_choice = st.selectbox(
                "Image generation model",
                options=[
                    "Local GPU - segmind/SSD-1B",
                    "Local GPU - Anything V5 (anime SD1.5)",
                    "Local GPU - CounterfeitXL (anime SDXL)",
                    "Gemini API - Imagen",
                    "OpenAI API - DALL-E 3",
                    "Local GPU - Custom Hugging Face model",
                ],
                key="panel_remix_image_backend_choice"
            )
            if panel_image_backend_choice == "Gemini API - Imagen":
                panel_image_backend = "gemini"
                panel_image_model_id = "imagen-4.0-generate-001"
                panel_image_api_key = st.text_input(
                    "Gemini image API key",
                    value=gemini_key,
                    type="password",
                    key="panel_remix_image_key"
                )
            elif panel_image_backend_choice == "OpenAI API - DALL-E 3":
                panel_image_backend = "openai"
                panel_image_model_id = "dall-e-3"
                panel_image_api_key = st.text_input(
                    "OpenAI image API key",
                    value=openai_key,
                    type="password",
                    key="panel_remix_image_key"
                )
            elif panel_image_backend_choice == "Local GPU - Anything V5 (anime SD1.5)":
                panel_image_backend = "local"
                panel_image_model_id = "stablediffusionapi/anything-v5"
                panel_image_api_key = ""
                st.caption("SD 1.5-based anime model. Great for stylized manga art. Lower VRAM than SDXL (~4GB).")
            elif panel_image_backend_choice == "Local GPU - CounterfeitXL (anime SDXL)":
                panel_image_backend = "local"
                panel_image_model_id = "gsdf/CounterfeitXL"
                panel_image_api_key = ""
                st.caption("SDXL-based anime model with high detail and consistent character rendering.")
            elif panel_image_backend_choice == "Local GPU - Custom Hugging Face model":
                panel_image_backend = "local"
                panel_image_model_id = st.text_input(
                    "Custom Hugging Face image model ID",
                    value="segmind/SSD-1B",
                    key="panel_remix_custom_image_model_id"
                )
                panel_image_api_key = ""
            else:
                panel_image_backend = "local"
                panel_image_model_id = "segmind/SSD-1B"
                panel_image_api_key = ""
    
            # ── Character LoRA (optional) ──────────────────────────────────────────
            st.markdown("---")
            _loras_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
            _available_loras = sorted([
                d.name for d in Path(_loras_dir).iterdir()
                if d.is_dir() and (d / "pytorch_lora_weights.safetensors").exists()
            ]) if os.path.isdir(_loras_dir) else []
            panel_lora_choice = st.selectbox(
                "🧬 Character LoRA — maintain consistent character appearance (optional)",
                ["None"] + _available_loras,
                key="panel_lora_choice",
                help="Train a LoRA in the LoRA Training Studio tab, then select it here for character consistency.",
            )
            if panel_lora_choice != "None":
                panel_lora_path = os.path.join(_loras_dir, panel_lora_choice)
                _col_ls, _col_lt = st.columns(2)
                with _col_ls:
                    panel_lora_scale = st.slider(
                        "LoRA strength", 0.0, 1.5, 0.8, step=0.05, key="panel_lora_scale"
                    )
                with _col_lt:
                    panel_lora_trigger = st.text_input(
                        "Trigger word",
                        key="panel_lora_trigger",
                        help="The instance token used during training, e.g. 'chihiro_kb'",
                    )
            else:
                panel_lora_path = None
                panel_lora_scale = 0.8
                panel_lora_trigger = ""
            st.markdown("---")
    
            # ── IP-Adapter ───────────────────────────────────────────────────────
            panel_ip_adapter = st.checkbox(
                "🎭 IP-Adapter — auto reference each scene's source panel for character consistency",
                value=False,
                key="panel_ip_adapter_enabled",
                help=(
                    "Each generated scene uses its matching manga panel as a visual reference. "
                    "No training required — works for all characters instantly. "
                    "Best combined with LoRA for maximum consistency (~80–85% alone, ~95% with LoRA)."
                ),
            )
            if panel_ip_adapter:
                panel_ip_scale = st.slider(
                    "IP-Adapter strength",
                    min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                    key="panel_ip_scale",
                    help="0.5–0.7 balances character likeness with prompt creativity. Go higher for strict character lock-in.",
                )
            else:
                panel_ip_scale = 0.6
            st.markdown("---")
    
            col_panel_limit_a, col_panel_limit_b = st.columns(2)
            with col_panel_limit_a:
                panel_scene_limit = st.slider(
                    "Limit scenes for test run",
                    min_value=0,
                    max_value=30,
                    value=0,
                    step=1,
                    help="Use 0 for all beats (recommended — generates an image for every audio beat). Set a small number only while quickly testing prompt quality or managing API cost.",
                    key="panel_remix_scene_limit"
                )
            with col_panel_limit_b:
                panel_captions = st.checkbox(
                    "Burn captions into MP4",
                    value=True,
                    key="panel_remix_captions"
                )
            panel_channel_name = st.text_input(
                "Channel name overlay (leave blank to skip)",
                value="", placeholder="e.g. MangaRecaps",
                key="panel_channel_name",
            )
            panel_show_lss = st.checkbox(
                "Show Like / Share / Subscribe animation at the end",
                value=True, key="panel_show_lss",
            )
            panel_sfx = st.checkbox(
                "🔊 Whoosh sound effects on every scene cut",
                value=True,
                key="panel_remix_sfx",
                help="Synthetic descending-sweep whoosh placed at each transition. No external files needed.",
            )
            if panel_sfx:
                panel_sfx_volume = st.slider(
                    "SFX volume", 0.0, 0.40, 0.12, step=0.01,
                    key="panel_remix_sfx_volume",
                    help="Relative to the voiceover. 0.10–0.15 is subtle; 0.25+ is noticeable.",
                )
            else:
                panel_sfx_volume = 0.12
            panel_draft = st.checkbox(
                "⚡ Fast Draft Mode (placeholder text cards, skips real image generation)",
                value=False,
                key="panel_remix_draft"
            )
    
            panel_generate = st.button("Generate Panel Remix Visuals 🎨", key="panel_remix_generate")
            st.markdown("</div>", unsafe_allow_html=True)
    
        with col_panel_out:
            # ── Single-scene regeneration handler ──────────────────────────────────
            _regen_idx = st.session_state.pop("regen_scene_idx", None)
            if _regen_idx is not None:
                _stored = st.session_state.get("panel_scenes", [])
                _cfg = st.session_state.get("panel_gen_config", {})
                if _stored and 0 <= _regen_idx < len(_stored):
                    _sc = _stored[_regen_idx]
                    with st.spinner(f"🔄 Regenerating scene #{_regen_idx + 1}..."):
                        _ri = generate_scene_image(
                            _sc["image_prompt"],
                            _cfg.get("art_style", "anime style, highly detailed"),
                            mock_mode=_cfg.get("draft", False),
                            image_backend=_cfg.get("image_backend", "local"),
                            model_id=_cfg.get("model_id", "segmind/SSD-1B"),
                            api_key=_cfg.get("api_key", ""),
                        )
                        _ri.save(_sc["image_path"])
                    st.success(f"✅ Scene #{_regen_idx + 1} regenerated!")
    
            if panel_generate:
                if not panel_files:
                    st.error("Please upload at least one manga panel image.")
                elif not panel_analyzer_key.strip():
                    st.error("Please provide an API key for panel analysis.")
                elif panel_voiceover is None and not panel_narration_text.strip():
                    st.error("Please upload a voiceover or paste narration text.")
                else:
                    st.subheader("⚡ Panel Remix Pipeline")
                    panel_images = []
                    for panel_file in panel_files:
                        try:
                            panel_images.append(Image.open(panel_file).convert("RGB"))
                        except Exception as e:
                            st.error(f"Could not read panel image {panel_file.name}: {e}")
                            st.stop()
    
                    panel_audio_path = None
                    if panel_voiceover is not None:
                        panel_audio_path = os.path.join(TEMP_DIR, f"panel_remix_{panel_voiceover.name}")
                        with open(panel_audio_path, "wb") as f:
                            f.write(panel_voiceover.getbuffer())
                    elif panel_narration_text.strip():
                        from core.tts_manager import generate_tts
                        _tts_out = os.path.join(TEMP_DIR, "panel_remix_tts.mp3")
                        _tts_label = {"gtts": "gTTS", "openai": "OpenAI TTS", "bark": "Bark"}.get(_panel_tts_backend, _panel_tts_backend)
                        try:
                            with st.spinner(f"🔊 Generating voiceover with {_tts_label}..."):
                                panel_audio_path = generate_tts(
                                    panel_narration_text, _tts_out,
                                    backend=_panel_tts_backend,
                                    voice=_panel_tts_voice,
                                    openai_key=openai_key,
                                )
                            st.success(f"✅ Voiceover generated ({_tts_label}).")
                        except Exception as _tts_err:
                            st.warning(f"⚠️ TTS failed: {_tts_err}. Will assemble without audio.")
    
                    try:
                        st.write("🧠 **Step 1:** Analyzing panels and building rich original prompts...")
                        panel_scenes, panel_transcript = create_panel_remix_scenes(
                            panel_images,
                            api_key=panel_analyzer_key,
                            analyzer_backend=panel_analyzer_backend,
                            narration_text=panel_narration_text,
                            audio_path=panel_audio_path,
                            detail_strength=panel_detail_strength,
                            art_style=panel_art_style,
                            originality_strength=panel_originality,
                            max_duration=4.5,
                            max_words=30,
                            openai_model=panel_openai_model,
                        )
                        if panel_scene_limit > 0:
                            panel_scenes = panel_scenes[:panel_scene_limit]
                        st.success(f"Created {len(panel_scenes)} panel-guided original scene prompts.")
    
                        with st.expander("Show generated prompt plan", expanded=True):
                            for idx, scene in enumerate(panel_scenes):
                                st.markdown(f"**Scene #{idx+1}** · Source panel #{scene.get('source_panel_index', 0) + 1}")
                                st.caption(scene.get("panel_description", ""))
                                scene["image_prompt"] = st.text_area(
                                    f"Editable image prompt #{idx+1}",
                                    value=scene.get("image_prompt", ""),
                                    height=170,
                                    key=f"panel_remix_prompt_{idx}"
                                )
                        if panel_transcript:
                            with st.expander("Show transcript / narration"):
                                st.write(panel_transcript)
                    except Exception as e:
                        st.error(f"Error creating panel remix prompts: {e}")
                        st.stop()
    
                    try:
                        selected_panel_model = "Gemini Imagen" if panel_image_backend == "gemini" else panel_image_model_id
                        st.write(f"🎨 **Step 2:** Generating remixed original images with {selected_panel_model}...")
                        progress_bar = st.progress(0.0)
                        for idx, scene in enumerate(panel_scenes):
                            image_path = os.path.join(TEMP_DIR, f"panel_remix_scene_{idx}_output.png")
                            st.write(f"🖌️ Scene #{idx+1}: `{scene['image_prompt'][:110]}...`")
                            # Resolve the source panel for IP-Adapter reference
                            _ip_ref = None
                            if panel_ip_adapter and panel_images:
                                _src_idx = min(scene.get("source_panel_index", 0), len(panel_images) - 1)
                                _ip_ref = panel_images[_src_idx]
                            img = generate_scene_image(
                                scene["image_prompt"],
                                panel_art_style,
                                mock_mode=panel_draft,
                                image_backend=panel_image_backend,
                                model_id=panel_image_model_id,
                                api_key=panel_image_api_key,
                                lora_path=panel_lora_path,
                                lora_scale=panel_lora_scale,
                                lora_trigger=panel_lora_trigger,
                                ip_adapter_image=_ip_ref,
                                ip_adapter_scale=panel_ip_scale,
                            )
                            img.save(image_path)
                            scene["image_path"] = image_path
                            progress_bar.progress((idx + 1) / len(panel_scenes))
                        st.success(f"Generated {len(panel_scenes)} remixed images.")
                        # Persist scenes + config so per-scene regeneration works across reruns
                        st.session_state["panel_scenes"] = panel_scenes
                        st.session_state["panel_gen_config"] = {
                            "art_style": panel_art_style,
                            "image_backend": panel_image_backend,
                            "model_id": panel_image_model_id,
                            "api_key": panel_image_api_key,
                            "draft": panel_draft,
                        }
                    except Exception as e:
                        st.error(f"Error generating panel remix images: {e}")
                        st.stop()
    
                    if panel_audio_path:
                        try:
                            st.write("🎬 **Step 3:** Assembling panel remix recap video...")
                            panel_output_path = os.path.join(OUTPUT_DIR, "panel_remix_recap.mp4")
                            panel_bgm_path = None
                            if panel_bgm is not None:
                                panel_bgm_path = os.path.join(TEMP_DIR, f"panel_bgm_{panel_bgm.name}")
                                with open(panel_bgm_path, "wb") as _bf:
                                    _bf.write(panel_bgm.getbuffer())
                            assemble_video(
                                panel_scenes, panel_audio_path, panel_output_path,
                                add_captions=panel_captions,
                                bgm_path=panel_bgm_path,
                                bgm_volume=panel_bgm_volume,
                                channel_name=panel_channel_name.strip() or None,
                                show_lss=panel_show_lss,
                                add_transition_sfx=panel_sfx,
                                sfx_volume=panel_sfx_volume,
                            )
                            st.success("Panel remix recap video compiled successfully!")
                            st.video(panel_output_path)
                            with open(panel_output_path, "rb") as f:
                                st.download_button(
                                    label="Download Panel Remix MP4 📥",
                                    data=f,
                                    file_name="panel_remix_recap.mp4",
                                    mime="video/mp4",
                                    key="panel_remix_download"
                                )
                        except Exception as e:
                            st.error(f"Error assembling panel remix video: {e}")
                            st.stop()
                    else:
                        st.info("Images generated. Upload voiceover audio next time if you also want an assembled MP4.")
            else:
                st.info("Upload manga panels, add voiceover or narration text, then generate original panel-guided visuals.")
    
        _display_scenes = st.session_state.get("panel_scenes") or (panel_scenes if 'panel_scenes' in locals() else [])
        if _display_scenes:
            st.markdown("---")
            st.subheader("🖼️ Panel Remix Scene Details")
            st.caption("Edit any prompt then click 🔄 to regenerate just that one scene — all others stay untouched.")
            for idx, scene in enumerate(_display_scenes):
                with st.container():
                    st.markdown("<div class='scene-preview-card'>", unsafe_allow_html=True)
                    col_scene_img, col_scene_info, col_scene_ctrl = st.columns([1, 2, 0.4])
                    with col_scene_img:
                        image_path = scene.get('image_path')
                        if image_path and os.path.exists(image_path):
                            st.image(image_path, use_container_width=True)
                        else:
                            st.caption("No image yet")
                    with col_scene_info:
                        st.markdown(
                            f"**Scene #{idx+1}** · `{scene.get('start', 0.0):.1f}s – {scene.get('end', 5.0):.1f}s`"
                            f" · Panel #{scene.get('source_panel_index', 0) + 1}"
                        )
                        st.caption(scene.get('text_segment', ''))
                        st.text_area(
                            "prompt",
                            value=scene.get("image_prompt", ""),
                            height=100,
                            key=f"scene_edit_prompt_{idx}",
                            label_visibility="collapsed",
                        )
                    with col_scene_ctrl:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        if st.button("🔄", key=f"regen_btn_{idx}", help=f"Regenerate scene #{idx+1} with current prompt"):
                            # Save any edited prompt back to session state before rerun
                            _edited = st.session_state.get(f"scene_edit_prompt_{idx}", scene["image_prompt"])
                            st.session_state["panel_scenes"][idx]["image_prompt"] = _edited
                            st.session_state["regen_scene_idx"] = idx
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
    


def render_lora(tab, TEMP_DIR):
    import sys
    import subprocess
    from pathlib import Path
    WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
    with tab:
    
        st.subheader("🧬 LoRA Training Studio")
        st.markdown(
            """
            Train a lightweight **DreamBooth LoRA** on your character's manga panels so every generated
            scene shows the same face / costume. Training runs in the background — you can continue
            using other tabs.
    
            **Recommended workflow:**
            1. Crop 10–20 clean panels of the character you want to be consistent.
            2. Enter a unique trigger word (e.g. `chihiro_kb`), pick the same base model you use in Panel Remix.
            3. Hit **Start Training** — takes ~10–30 min on an RTX GPU.
            4. In **Panel Remix**, select the trained LoRA and paste the same trigger word.
            """
        )
    
        # Check peft availability
        try:
            import peft  # noqa: F401
            _peft_ok = True
        except ImportError:
            _peft_ok = False
    
        if not _peft_ok:
            st.warning(
                "⚠️ The `peft` package is required for LoRA training. "
                "Install it with:\n```\npip install peft>=0.7.0 accelerate>=0.30.0\n```",
                icon="⚠️",
            )
    
        with st.container():
            lora_ref_images = st.file_uploader(
                "Upload 10–20 character reference images (PNG / JPG)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="lora_ref_images",
            )
            _effective_images = list(lora_ref_images) if lora_ref_images else []
            if _effective_images:
                st.success(f"📸 **{len(_effective_images)} image(s)** ready for training.")
    
            lora_col1, lora_col2 = st.columns(2)
            with lora_col1:
                lora_name = st.text_input(
                    "LoRA name (used as folder name in `loras/`)",
                    value="my_character",
                    key="lora_name",
                )
                lora_trigger = st.text_input(
                    "Instance / trigger word",
                    value="mychar",
                    key="lora_trigger_word",
                    help="Short unique token prepended to prompts during generation, e.g. 'chihiro_kb'",
                )
                lora_base_model_choice = st.selectbox(
                    "Base model (must match what you use in Panel Remix)",
                    [
                        "Local GPU - CounterfeitXL (anime SDXL)",
                        "Local GPU - Anything V5 (anime SD1.5)",
                        "Local GPU - SSD-1B (fast SDXL)",
                        "Local GPU - Custom Hugging Face model",
                    ],
                    key="lora_base_model_choice",
                )
                if lora_base_model_choice == "Local GPU - CounterfeitXL (anime SDXL)":
                    _lora_base_id = "gsdf/CounterfeitXL"
                    _lora_is_sdxl = True
                elif lora_base_model_choice == "Local GPU - Anything V5 (anime SD1.5)":
                    _lora_base_id = "stablediffusionapi/anything-v5"
                    _lora_is_sdxl = False
                elif lora_base_model_choice == "Local GPU - SSD-1B (fast SDXL)":
                    _lora_base_id = "segmind/SSD-1B"
                    _lora_is_sdxl = True
                else:
                    _lora_base_id = st.text_input(
                        "Custom HF model ID", value="segmind/SSD-1B", key="lora_custom_base_id"
                    )
                    _lora_is_sdxl = st.checkbox("Is SDXL-based?", value=True, key="lora_custom_is_sdxl")
    
            with lora_col2:
                lora_steps = st.slider(
                    "Training steps", min_value=100, max_value=1500, value=400, step=50,
                    key="lora_steps",
                    help="400–600 steps is a good default for 15 reference images.",
                )
                lora_lr = st.select_slider(
                    "Learning rate",
                    options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                    value=1e-4,
                    format_func=lambda v: f"{v:.0e}",
                    key="lora_lr",
                )
                lora_rank = st.select_slider(
                    "LoRA rank",
                    options=[2, 4, 8, 16],
                    value=4,
                    key="lora_rank",
                    help="Higher rank = more capacity but more VRAM. Rank 4 works well for most cases.",
                )
    
        # ── Status display ────────────────────────────────────────────────────────
        _lora_status_file = os.path.join(TEMP_DIR, f"lora_status_{lora_name}.json")
        _lora_images_dir = os.path.join(TEMP_DIR, f"lora_images_{lora_name}")
        _lora_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras", lora_name)
    
        _status_placeholder = st.empty()
    
        def _read_lora_status():
            try:
                with open(_lora_status_file) as _f:
                    return json.load(_f)
            except Exception:
                return None
    
        def _render_status(status_data):
            if not status_data:
                return
            s = status_data.get("status", "")
            msg = status_data.get("message", "")
            step = status_data.get("step", 0)
            total = status_data.get("total_steps", 1)
            loss = status_data.get("loss")
            if s == "completed":
                _status_placeholder.success(f"✅ {msg}")
            elif s == "error":
                _status_placeholder.error(f"❌ Training error: {msg}")
            elif s in ("training", "initializing"):
                pct = step / max(total, 1)
                _status_placeholder.info(f"⚙️ {msg}")
                st.progress(pct, text=f"Step {step}/{total}" + (f" | loss {loss:.4f}" if loss else ""))
    
        # Restore status from any ongoing run
        _current_status = _read_lora_status()
        _render_status(_current_status)
    
        # Check if a training process is already running
        _proc = st.session_state.get("lora_training_process")
        _is_running = _proc is not None and _proc.poll() is None
    
        lora_btn_col, lora_refresh_col, lora_stop_col = st.columns([3, 1, 1])
    
        with lora_btn_col:
            _start_disabled = _is_running or not _peft_ok
            _btn_label = "⏳ Training in progress..." if _is_running else "🚀 Start Training"
            lora_start = st.button(_btn_label, disabled=_start_disabled, key="lora_start")
    
        with lora_refresh_col:
            lora_refresh = st.button("🔄 Refresh", key="lora_refresh")
    
        with lora_stop_col:
            lora_stop = st.button("⏹ Stop", disabled=not _is_running, key="lora_stop")
    
        if lora_refresh:
            st.rerun()
    
        if lora_stop and _is_running:
            _proc.terminate()
            st.session_state["lora_training_process"] = None
            st.warning("Training stopped.")
            st.rerun()
    
        if lora_start:
            if not _effective_images:
                st.error("No images found. Upload panels here or use panels from the Panel Remix tab.")
            elif not lora_name.strip():
                st.error("Please enter a LoRA name.")
            elif not lora_trigger.strip():
                st.error("Please enter a trigger / instance word.")
            else:
                # Save all effective images to temp dir
                os.makedirs(_lora_images_dir, exist_ok=True)
                for _uf in _effective_images:
                    _dest = os.path.join(_lora_images_dir, _uf.name)
                    with open(_dest, "wb") as _fh:
                        _fh.write(_uf.getbuffer())
    
                # Build subprocess command
                _train_script = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "core", "train_lora.py"
                )
                _cmd = [
                    sys.executable, _train_script,
                    "--images_dir", _lora_images_dir,
                    "--output_dir", _lora_output_dir,
                    "--instance_prompt", f"{lora_trigger.strip()} person",
                    "--base_model", _lora_base_id,
                    "--max_steps", str(lora_steps),
                    "--lr", str(lora_lr),
                    "--lora_rank", str(lora_rank),
                    "--status_file", _lora_status_file,
                ]
                if _lora_is_sdxl:
                    _cmd.append("--is_sdxl")
    
                # Write initial status so the Refresh poll works immediately
                with open(_lora_status_file, "w") as _sf:
                    json.dump({"status": "initializing", "step": 0,
                               "total_steps": lora_steps, "message": "Launching training process..."}, _sf)
    
                _proc = subprocess.Popen(
                    _cmd,
                    stdout=open(os.path.join(TEMP_DIR, f"lora_log_{lora_name}.txt"), "w"),
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                st.session_state["lora_training_process"] = _proc
                st.success(
                    f"Training started in background (PID {_proc.pid}). "
                    "Click **Refresh** to poll progress."
                )
                st.rerun()
    
        # ── Trained LoRA library ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📚 Trained LoRA Library")
        _loras_dir_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
        _trained = [
            d for d in Path(_loras_dir_lib).iterdir()
            if d.is_dir() and (d / "pytorch_lora_weights.safetensors").exists()
        ] if os.path.isdir(_loras_dir_lib) else []
        if _trained:
            for _ld in sorted(_trained):
                _sz = (Path(_ld) / "pytorch_lora_weights.safetensors").stat().st_size / 1024 / 1024
                st.markdown(f"- **{_ld.name}** — `{_sz:.1f} MB`")
        else:
            st.info("No trained LoRAs yet. Upload reference images and start training above.")