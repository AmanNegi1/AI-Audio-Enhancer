import streamlit as st
import numpy as np
import json
import zipfile
import io
import os
import hashlib
import cv2
from PIL import Image

# Import core modules
from core.loader import load_image, pil_to_bytes
from core.bubble_eraser import detect_speech_bubbles, erase_regions
from core.colorizer import apply_color_zone, build_color_layer, COLORING_MODES
from core.blender import blend_layers
from core.auto import (
    auto_region_colors,
    auto_region_colors_by_panel,
    detect_panels,
    apply_hue_shift,
    apply_mirror,
    apply_chromatic_aberration,
)

# ─────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────

def _strip_contour(bubble_dict):
    return {k: v for k, v in bubble_dict.items() if k != 'contour'}


def _upload_page_key(filename, file_bytes, current_batch_keys=None):
    current_batch_keys = current_batch_keys or set()
    digest = hashlib.sha1(file_bytes).hexdigest()[:12]
    existing = st.session_state.manga_pages
    if filename not in existing and filename not in current_batch_keys:
        return filename, digest
    if filename in existing and existing[filename].get('upload_hash') in (None, digest):
        return filename, digest
    stem, ext = os.path.splitext(filename)
    base_key = f"{stem} [{digest[:8]}]{ext}"
    page_key = base_key
    suffix = 2
    while page_key in existing and existing[page_key].get('upload_hash') != digest:
        page_key = f"{stem} [{digest[:8]}-{suffix}]{ext}"
        suffix += 1
    return page_key, digest


def _page_display_name(page_key):
    page = st.session_state.manga_pages.get(page_key, {})
    display_name = page.get('original_name', page_key)
    if display_name != page_key:
        return f"{display_name} ({page_key})"
    return display_name


def _unique_export_name(filename, used_names):
    stem, ext = os.path.splitext(filename)
    export_name = filename
    suffix = 2
    while export_name in used_names:
        export_name = f"{stem}_{suffix}{ext}"
        suffix += 1
    used_names.add(export_name)
    return export_name


def _bubble_id(bubble):
    return (
        int(bubble.get('x', 0)),
        int(bubble.get('y', 0)),
        int(bubble.get('w', 0)),
        int(bubble.get('h', 0)),
    )


def _bubble_checkbox_key(active_name, idx, bubble):
    x, y, bw, bh = _bubble_id(bubble)
    return f"bub_{active_name}_{idx}_{x}_{y}_{bw}_{bh}"


def _sync_bubble_checkboxes(active_name, bubbles, selected):
    selected_ids = {_bubble_id(b) for b in selected}
    for idx, bubble in enumerate(bubbles):
        st.session_state[_bubble_checkbox_key(active_name, idx, bubble)] = (
            _bubble_id(bubble) in selected_ids
        )


# ─────────────────────────────────────────────
#  Coloring / pipeline helpers
# ─────────────────────────────────────────────

COLORING_STYLES = ["Palette (per panel)", "Palette (whole page)"] + [
    m for m in COLORING_MODES if m not in ("Palette",)
]


def _build_regions(page_data, palette, style, extra_args=None):
    extra_args = extra_args or {}
    h, w = page_data['gray'].shape
    if style == "Palette (per panel)":
        return auto_region_colors_by_panel(page_data['gray'], palette)
    if style == "Palette (whole page)":
        return auto_region_colors(page_data['rgb'].shape, palette)
    region = {'mode': style, 'label': style, 'rect': (0, 0, w, h)}
    region.update(extra_args)
    return [region]


def auto_stylize_page(page_data, palette, style="Palette (per panel)", extra_args=None):
    """Erase bubble text and auto-color regions for copyright bypass."""
    bubbles = detect_speech_bubbles(page_data['gray'])
    page_data['bubbles'] = bubbles
    page_data['erased_bubbles'] = list(bubbles)
    page_data['regions'] = _build_regions(page_data, palette, style, extra_args)
    page_data['geoms'] = []


def apply_text_watermark(img_rgb, watermark_cfg=None):
    if not watermark_cfg:
        return img_rgb
    text = str(watermark_cfg.get('text', '')).strip()
    if not text:
        return img_rgb

    height, width = img_rgb.shape[:2]
    opacity = float(max(0.05, min(1.0, watermark_cfg.get('opacity', 0.28))))
    scale = float(max(0.3, min(3.0, watermark_cfg.get('scale', 1.0))))
    position = watermark_cfg.get('position', 'Bottom Right')

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(1 + scale)))
    font_scale = max(0.5, min(width, height) / 900.0) * scale
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = max(10, int(min(width, height) * 0.025))

    if position == 'Top Left':
        x, y = margin, margin + text_h
    elif position == 'Top Right':
        x, y = width - text_w - margin, margin + text_h
    elif position == 'Bottom Left':
        x, y = margin, height - margin
    elif position == 'Center':
        x, y = (width - text_w) // 2, (height + text_h) // 2
    else:
        x, y = width - text_w - margin, height - margin

    overlay = img_rgb.copy()
    bg_pad = max(6, int(margin * 0.35))
    box_x1 = max(0, x - bg_pad)
    box_y1 = max(0, y - text_h - bg_pad)
    box_x2 = min(width, x + text_w + bg_pad)
    box_y2 = min(height, y + baseline + bg_pad)
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, opacity, img_rgb, 1.0 - opacity, 0)


def build_final_image(page_data, brightness=1.0, contrast=1.0,
                      hue_degrees=0.0, mirror=False, watermark_cfg=None,
                      blend_mode="Multiply", ca_shift=0, ca_direction='horizontal'):
    """Erase bubbles, apply auto-color layers, then post-process."""
    erased = erase_regions(page_data['rgb'], page_data['erased_bubbles'])
    erased_gray = cv2.cvtColor(erased, cv2.COLOR_RGB2GRAY)
    layers = []
    for r in page_data['regions']:
        mode = r.get('mode', 'Palette')
        layers.append(build_color_layer(mode, erased_gray, r))
    active_blend = blend_mode
    if any(r.get('mode') in ('Neon Glow', 'Gradient Map') for r in page_data['regions']):
        active_blend = 'Replace'
    blended = blend_layers(
        erased, layers,
        geometric_shapes=page_data.get('geoms', []),
        brightness=brightness, contrast=contrast,
        blend_mode=active_blend,
    )
    if hue_degrees:
        blended = apply_hue_shift(blended, hue_degrees)
    if mirror:
        blended = apply_mirror(blended)
    if ca_shift and ca_shift > 0:
        blended = apply_chromatic_aberration(blended, ca_shift, ca_direction)
    blended = apply_text_watermark(blended, watermark_cfg)
    return blended


# ─────────────────────────────────────────────
#  Streamlit app
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Manga Colorizer & Copyright Bypass Stylizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 20% 30%, #171822 0%, #0c0d12 100%);
        color: #e2e8f0;
    }
    .css-1d391kg {
        background-color: rgba(23, 24, 34, 0.9) !important;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stSidebar"] {
        background-color: rgba(18, 19, 28, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(79, 70, 229, 0.4);
        border: none;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        background-color: transparent;
        color: #a0aec0;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(79, 70, 229, 0.15) !important;
        color: #818cf8 !important;
        font-weight: bold;
    }
    .highlight-box {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'palettes' not in st.session_state:
    st.session_state.palettes = {}
    palette_path = os.path.join(os.path.dirname(__file__), "palettes", "kagurabachi.json")
    if os.path.exists(palette_path):
        try:
            with open(palette_path, "r") as f:
                data = json.load(f)
                st.session_state.palettes[data["series"]] = data
        except Exception as e:
            st.error(f"Failed to load palette: {e}")

if 'manga_pages' not in st.session_state:
    st.session_state.manga_pages = {}

if 'active_file' not in st.session_state:
    st.session_state.active_file = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎨 Manga Stylizer")
    st.write(
        "Upload pages → auto-erase speech bubbles → auto-color panels "
        "for copyright bypass."
    )

    uploaded_files = st.file_uploader(
        "Upload manga pages or a .cbz file",
        type=["png", "jpg", "jpeg", "webp", "cbz"],
        accept_multiple_files=True,
    )

    def _load_page_bytes(filename, file_bytes, batch_keys):
        page_key, upload_hash = _upload_page_key(filename, file_bytes, batch_keys)
        batch_keys.add(page_key)
        if page_key in st.session_state.manga_pages:
            return page_key
        try:
            rgb, gray, pil = load_image(file_bytes)
            detected_bubbles = detect_speech_bubbles(gray)
            st.session_state.manga_pages[page_key] = {
                'original_name': filename,
                'upload_hash': upload_hash,
                'rgb': rgb,
                'gray': gray,
                'pil': pil,
                'bubbles': detected_bubbles,
                'erased_bubbles': list(detected_bubbles),
                'regions': [],
                'geoms': [],
                'panels': None,
            }
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
        return page_key

    if uploaded_files:
        current_batch_keys = set()
        for file in uploaded_files:
            file_bytes = file.getvalue()
            if file.name.lower().endswith('.cbz'):
                try:
                    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                        img_entries = sorted([
                            n for n in zf.namelist()
                            if os.path.splitext(n)[1].lower() in
                               ('.png', '.jpg', '.jpeg', '.webp')
                            and not os.path.basename(n).startswith('.')
                        ])
                        if not img_entries:
                            st.warning(f"{file.name}: no supported image files found inside.")
                        for entry in img_entries:
                            _load_page_bytes(
                                os.path.basename(entry),
                                zf.read(entry),
                                current_batch_keys,
                            )
                except Exception as e:
                    st.error(f"Could not open {file.name} as CBZ: {e}")
            else:
                _load_page_bytes(file.name, file_bytes, current_batch_keys)

        if not st.session_state.active_file and st.session_state.manga_pages:
            st.session_state.active_file = list(st.session_state.manga_pages.keys())[0]

    if st.session_state.manga_pages:
        _all_page_keys = list(st.session_state.manga_pages.keys())
        _cur_idx = (
            _all_page_keys.index(st.session_state.active_file)
            if st.session_state.active_file in _all_page_keys
            else 0
        )

        st.session_state.active_file = st.selectbox(
            "Select Page to Edit",
            options=_all_page_keys,
            index=_cur_idx,
            format_func=_page_display_name,
        )

        _nav_prev, _nav_next = st.columns(2)
        with _nav_prev:
            if st.button("← Prev", disabled=_cur_idx == 0, use_container_width=True):
                st.session_state.active_file = _all_page_keys[_cur_idx - 1]
                st.rerun()
        with _nav_next:
            if st.button("Next →", disabled=_cur_idx >= len(_all_page_keys) - 1,
                         use_container_width=True):
                st.session_state.active_file = _all_page_keys[_cur_idx + 1]
                st.rerun()
        st.caption(f"Page {_cur_idx + 1} of {len(_all_page_keys)}")

        # Auto-Color settings
        st.markdown("---")
        st.subheader("Auto-Color Settings")

        palette_names = list(st.session_state.palettes.keys())
        auto_palette_name = st.selectbox(
            "Palette",
            options=palette_names if palette_names else ["(no palette loaded)"],
            key="auto_palette",
        )
        auto_style = st.selectbox(
            "Coloring Style",
            COLORING_STYLES,
            key="auto_style",
            help="'Palette (per panel)' auto-detects manga panels and tints each differently.",
        )

        # Style-specific extra args
        _extra_args = {}
        if palette_names:
            _pal_obj = st.session_state.palettes[auto_palette_name]
            _flat_colors = {}
            for _name, _c in _pal_obj.get("environment", {}).items():
                _flat_colors[f"Env: {_name}"] = _c
            for _ch_name, _ch_cols in _pal_obj.get("characters", {}).items():
                for _part, _c in _ch_cols.items():
                    _flat_colors[f"{_ch_name} - {_part}"] = _c
            _ck = list(_flat_colors.keys()) or ["(empty)"]

            if auto_style == "Duotone":
                d_k = st.selectbox("Dark color", _ck, key="s_duo_dark")
                b_k = st.selectbox("Bright color", _ck, index=min(1, len(_ck) - 1), key="s_duo_bright")
                _extra_args = {
                    'dark': _flat_colors.get(d_k, [20, 20, 60]),
                    'bright': _flat_colors.get(b_k, [240, 220, 180]),
                }
            elif auto_style == "Sunset Gradient":
                t_k = st.selectbox("Top color", _ck, key="s_grad_top")
                b_k = st.selectbox("Bottom color", _ck, index=min(1, len(_ck) - 1), key="s_grad_bot")
                dir_ = st.radio("Direction", ["vertical", "horizontal"], key="s_grad_dir")
                _extra_args = {
                    'top': _flat_colors.get(t_k, [255, 180, 90]),
                    'bottom': _flat_colors.get(b_k, [60, 30, 90]),
                    'direction': dir_,
                }
            elif auto_style == "Posterize":
                s_k = st.selectbox("Shadow", _ck, key="s_post_shad")
                m_k = st.selectbox("Midtone", _ck, index=min(1, len(_ck) - 1), key="s_post_mid")
                h_k = st.selectbox("Highlight", _ck, index=min(2, len(_ck) - 1), key="s_post_hi")
                _extra_args = {
                    'shadow': _flat_colors.get(s_k, [30, 30, 60]),
                    'mid': _flat_colors.get(m_k, [180, 80, 80]),
                    'highlight': _flat_colors.get(h_k, [250, 240, 220]),
                }
            elif auto_style == "Flat Color":
                fc_k = st.selectbox("Fill Color", _ck, key="s_fc_color")
                _extra_args = {'color': _flat_colors.get(fc_k, [100, 150, 220])}
            elif auto_style == "Halftone":
                ht_k = st.selectbox("Dot Color", _ck, key="s_ht_color")
                ht_sp = st.slider("Dot Spacing (px)", 4, 24, 10, 1, key="s_ht_spacing")
                _extra_args = {
                    'dot_color': _flat_colors.get(ht_k, [80, 120, 200]),
                    'dot_spacing': ht_sp,
                }
            elif auto_style == "Neon Glow":
                ng_k = st.selectbox("Glow Color", _ck, key="s_ng_color")
                ng_str = st.slider("Glow Strength", 0.2, 1.5, 0.8, 0.1, key="s_ng_strength")
                _extra_args = {
                    'glow_color': _flat_colors.get(ng_k, [0, 255, 180]),
                    'glow_strength': ng_str,
                }
                st.caption("Neon Glow uses a dark background; blend mode auto-switches to Replace.")
            elif auto_style == "Zone Block":
                zb_hex = st.color_picker(
                    "Block Color", "#FFB591", key="s_zb_hex",
                    help="Light orange-pink / peach works best for copyright-bypass aesthetics.",
                )
                zb_r = int(zb_hex[1:3], 16)
                zb_g = int(zb_hex[3:5], 16)
                zb_b = int(zb_hex[5:7], 16)
                zb_density = st.slider(
                    "Content Density Threshold", 0.02, 0.30, 0.08, 0.01,
                    key="s_zb_density",
                    help="Lower = paint more cells. Higher = only the densest content areas.",
                )
                zb_rows = st.slider("Grid Rows", 4, 16, 8, 1, key="s_zb_rows")
                zb_cols = st.slider("Grid Cols", 3, 12, 6, 1, key="s_zb_cols")
                _extra_args = {
                    'block_color': [zb_r, zb_g, zb_b],
                    'density_thresh': zb_density,
                    'grid_rows': zb_rows,
                    'grid_cols': zb_cols,
                }
                st.caption("Fills content-rich grid cells with coloured rectangles automatically.")
            elif auto_style == "Gradient Map":
                gm_n = st.slider("Number of colour stops", 2, 4, 3, 1, key="s_gm_stops")
                _LABELS   = ["Shadow (darkest)", "Midtone", "Quarter-high", "Highlight (brightest)"]
                _DEFAULTS_3 = ["#1D1160", "#C4622D", "#F5E6C8"]
                _DEFAULTS_2 = ["#1D1160", "#F5E6C8"]
                _DEFAULTS_4 = ["#1A0A3D", "#8B2252", "#E8751A", "#FFF5D6"]
                _def = {2: _DEFAULTS_2, 3: _DEFAULTS_3, 4: _DEFAULTS_4}[gm_n]
                _positions = [i / max(1, gm_n - 1) for i in range(gm_n)]
                _stops = []
                for _si in range(gm_n):
                    _hex = st.color_picker(
                        _LABELS[_si], _def[_si], key=f"s_gm_c{_si}"
                    )
                    _stops.append([
                        _positions[_si],
                        [int(_hex[1:3], 16), int(_hex[3:5], 16), int(_hex[5:7], 16)],
                    ])
                _extra_args = {'stops': _stops}
                st.caption("Maps every brightness value to a colour — strongest hash-breaker. Blend auto-switches to Replace.")

        _col1, _col2 = st.columns(2)
        with _col1:
            if st.button("✨ Stylize Current") and palette_names:
                auto_stylize_page(
                    st.session_state.manga_pages[st.session_state.active_file],
                    st.session_state.palettes[auto_palette_name],
                    style=auto_style, extra_args=_extra_args,
                )
                st.success("Done.")
                st.rerun()
        with _col2:
            if st.button("✨ Stylize All") and palette_names:
                for _p in st.session_state.manga_pages.values():
                    auto_stylize_page(
                        _p, st.session_state.palettes[auto_palette_name],
                        style=auto_style, extra_args=_extra_args,
                    )
                st.success(f"Stylized {len(st.session_state.manga_pages)} pages.")
                st.rerun()

        # Hash-Breaker
        st.markdown("---")
        st.subheader("Hash-Breaker (Export)")
        st.session_state['export_hue'] = st.slider(
            "Hue Shift (°)", -30, 30,
            st.session_state.get('export_hue', 6), 1,
            help="Small hue rotation defeats perceptual hash matching.",
        )
        st.session_state['export_mirror'] = st.checkbox(
            "Mirror horizontally on export",
            value=st.session_state.get('export_mirror', False),
        )
        st.session_state['export_ca_shift'] = st.slider(
            "Chromatic Aberration (px)", 0, 20,
            int(st.session_state.get('export_ca_shift', 0)), 1,
            help="Splits R/B channels apart. 0 = off, 2–5 = subtle fringe, 8+ = dramatic.",
        )
        if st.session_state['export_ca_shift'] > 0:
            st.session_state['export_ca_dir'] = st.radio(
                "CA Direction",
                ["horizontal", "vertical", "diagonal"],
                index=["horizontal", "vertical", "diagonal"].index(
                    st.session_state.get('export_ca_dir', 'horizontal')
                ),
                horizontal=True,
                key="ca_dir_radio",
            )

        # Watermark
        st.markdown("---")
        st.subheader("Watermark (Export)")
        st.session_state['export_watermark_text'] = st.text_input(
            "Watermark Text",
            value=st.session_state.get('export_watermark_text', ''),
        )
        st.session_state['export_watermark_position'] = st.selectbox(
            "Watermark Position",
            ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"],
            index=["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"].index(
                st.session_state.get('export_watermark_position', 'Bottom Right')
            ),
        )
        st.session_state['export_watermark_opacity'] = st.slider(
            "Watermark Opacity", 0.05, 1.0,
            float(st.session_state.get('export_watermark_opacity', 0.28)), 0.05,
        )
        st.session_state['export_watermark_scale'] = st.slider(
            "Watermark Size", 0.5, 2.5,
            float(st.session_state.get('export_watermark_scale', 1.0)), 0.1,
        )

        watermark_cfg = {
            'text': st.session_state.get('export_watermark_text', ''),
            'position': st.session_state.get('export_watermark_position', 'Bottom Right'),
            'opacity': st.session_state.get('export_watermark_opacity', 0.28),
            'scale': st.session_state.get('export_watermark_scale', 1.0),
        }

        # Bulk Export
        st.markdown("---")
        st.subheader("Bulk Export")
        zip_buffer = io.BytesIO()
        used_export_names = set()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for fname, p_data in st.session_state.manga_pages.items():
                blended = build_final_image(
                    p_data,
                    hue_degrees=st.session_state.get('export_hue', 0),
                    mirror=st.session_state.get('export_mirror', False),
                    watermark_cfg=watermark_cfg,
                    blend_mode=st.session_state.get('blend_mode', 'Multiply'),
                    ca_shift=st.session_state.get('export_ca_shift', 0),
                    ca_direction=st.session_state.get('export_ca_dir', 'horizontal'),
                )
                pil_to_save = Image.fromarray(blended)
                img_bytes = pil_to_bytes(pil_to_save, format="PNG")
                original_name = p_data.get('original_name', fname)
                export_name = _unique_export_name(f"stylized_{original_name}", used_export_names)
                zip_file.writestr(export_name, img_bytes)

        st.download_button(
            "Download All Pages as ZIP 📦",
            data=zip_buffer.getvalue(),
            file_name="manga_stylized_batch.zip",
            mime="application/zip",
        )
    else:
        st.info("Upload one or more manga pages to get started.")

# ── Main window ────────────────────────────────────────────────────────────────
if st.session_state.active_file:
    active_name = st.session_state.active_file
    page = st.session_state.manga_pages[active_name]
    active_display_name = page.get('original_name', active_name)
    h, w, _ = page['rgb'].shape

    # Filmstrip
    _all_keys = list(st.session_state.manga_pages.keys())
    if len(_all_keys) > 1:
        _active_idx = _all_keys.index(active_name) if active_name in _all_keys else 0
        _STRIP = 10
        _start = max(0, min(_active_idx - _STRIP // 2, len(_all_keys) - _STRIP))
        _visible = _all_keys[_start: _start + _STRIP]
        _cols = st.columns(len(_visible))
        for _col, _pk in zip(_cols, _visible):
            _pg = st.session_state.manga_pages[_pk]
            _t = _pg['rgb']
            _th = cv2.resize(_t, (80, max(1, int(_t.shape[0] * 80 / _t.shape[1]))),
                             interpolation=cv2.INTER_AREA)
            _is_active = _pk == active_name
            with _col:
                st.image(_th, use_container_width=True)
                _label = f"▶ {_all_keys.index(_pk)+1}" if _is_active else str(_all_keys.index(_pk)+1)
                if st.button(_label, key=f"strip_{_pk}", use_container_width=True,
                             type="primary" if _is_active else "secondary"):
                    st.session_state.active_file = _pk
                    st.rerun()
        if len(_all_keys) > _STRIP:
            st.caption(f"Showing pages {_start+1}–{_start+len(_visible)} of {len(_all_keys)}")
        st.markdown("---")

    st.title(f"Editing: `{active_display_name}` ({w}×{h} px)")

    tab1, tab2 = st.tabs(["💬 Bubble Eraser", "💾 Preview & Export"])

    # ── Tab 1: Bubble Eraser ───────────────────────────────────────────────────
    with tab1:
        st.subheader("Speech Bubble Eraser")
        st.write(
            "AI + CV detection automatically finds text bubbles and erases them. "
            "Adjust sliders and press Re-detect if needed."
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            w_thresh = st.slider(
                "Bubble Brightness Floor", 150, 255, 225, 5,
                key=f"w_thr_{active_name}",
                help="Lower if your bubbles sit on screentone / off-white.",
            )
        with col_s2:
            m_area = st.slider(
                "Minimum Bubble Area (px)", 100, 20000, 400, 100,
                key=f"m_ar_{active_name}",
            )
        with col_s3:
            min_chars = st.slider(
                "Min text characters in cluster", 3, 30, 3, 1,
                key=f"mc_{active_name}",
                help="Raise to reject artwork details that look like text.",
            )

        col_act_1, col_act_2 = st.columns(2)
        with col_act_1:
            if st.button("Re-detect bubbles on this page", key=f"redet_{active_name}"):
                page['bubbles'] = detect_speech_bubbles(
                    page['gray'],
                    min_area=m_area,
                    white_thresh=w_thresh,
                    min_text_components=min_chars,
                )
                page['erased_bubbles'] = list(page['bubbles'])
                st.rerun()
        with col_act_2:
            if st.button("Re-detect on ALL pages", key="redet_all"):
                for p in st.session_state.manga_pages.values():
                    p['bubbles'] = detect_speech_bubbles(
                        p['gray'],
                        min_area=m_area,
                        white_thresh=w_thresh,
                        min_text_components=min_chars,
                    )
                    p['erased_bubbles'] = list(p['bubbles'])
                st.rerun()

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Detected Bubbles List")
            if page['bubbles']:
                if st.button("Select All Bubbles"):
                    page['erased_bubbles'] = list(page['bubbles'])
                    _sync_bubble_checkboxes(active_name, page['bubbles'], page['erased_bubbles'])
                    st.rerun()
                if st.button("Deselect All"):
                    page['erased_bubbles'] = []
                    _sync_bubble_checkboxes(active_name, page['bubbles'], page['erased_bubbles'])
                    st.rerun()

                selected_bubbles = []
                selected_ids = {_bubble_id(b) for b in page['erased_bubbles']}
                for idx, bubble in enumerate(page['bubbles']):
                    label = f"Bubble #{idx+1} ({bubble['x']}, {bubble['y']}) — Area: {int(bubble['area'])}"
                    checked = st.checkbox(
                        label,
                        value=_bubble_id(bubble) in selected_ids,
                        key=_bubble_checkbox_key(active_name, idx, bubble),
                    )
                    if checked:
                        selected_bubbles.append(bubble)
                page['erased_bubbles'] = selected_bubbles
            else:
                st.info("No speech bubbles detected. Try lowering the Brightness Floor.")

            st.markdown("#### Manual Bubble Erase")
            manual_shape = st.selectbox(
                "Erase Shape",
                ["Bubble double pass", "Ellipse only", "Caption rectangle"],
                key=f"manual_shape_{active_name}",
            )
            mx1 = st.number_input("Top-left X", 0, w, 0, key=f"manual_x1_{active_name}")
            my1 = st.number_input("Top-left Y", 0, h, 0, key=f"manual_y1_{active_name}")
            mx2 = st.number_input("Bottom-right X", 0, w, min(w, 100), key=f"manual_x2_{active_name}")
            my2 = st.number_input("Bottom-right Y", 0, h, min(h, 80), key=f"manual_y2_{active_name}")
            inset_px = st.number_input(
                "Center rectangle inset", 0, 50, 2,
                key=f"manual_inset_{active_name}",
            )
            if st.button("Erase Manual Bubble", key=f"manual_erase_{active_name}"):
                x1, x2 = sorted((int(mx1), int(mx2)))
                y1, y2 = sorted((int(my1), int(my2)))
                if x2 <= x1 or y2 <= y1:
                    st.warning("Manual erase needs a non-empty bounding box.")
                else:
                    shape_map = {
                        "Bubble double pass": "double_pass",
                        "Ellipse only": "ellipse",
                        "Caption rectangle": "rectangle",
                    }
                    page['erased_bubbles'].append({
                        'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                        'shape': shape_map[manual_shape],
                        'inset_px': int(inset_px),
                    })
                    st.success("Manual bubble erase added!")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col1:
            highlighted = page['rgb'].copy()
            for idx, bubble in enumerate(page['bubbles']):
                is_erased = any(
                    b['x'] == bubble['x'] and b['y'] == bubble['y']
                    for b in page['erased_bubbles']
                )
                color = (0, 255, 0) if is_erased else (255, 0, 0)
                cv2.rectangle(
                    highlighted,
                    (bubble['x'], bubble['y']),
                    (bubble['x'] + bubble['w'], bubble['y'] + bubble['h']),
                    color, 4,
                )
                cv2.putText(
                    highlighted, str(idx + 1),
                    (bubble['x'] + 5, bubble['y'] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )

            for manual in page['erased_bubbles']:
                if manual.get('mask') is not None:
                    continue
                x1 = int(manual.get('x', 0)); y1 = int(manual.get('y', 0))
                x2 = x1 + int(manual.get('w', 0)); y2 = y1 + int(manual.get('h', 0))
                shape = str(manual.get('shape', 'rectangle')).lower()
                if shape in ('ellipse', 'bubble', 'double_pass'):
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
                    cv2.ellipse(highlighted, center, axes, 0, 0, 360, (0, 180, 255), 3)
                if shape in ('rectangle', 'caption', 'double_pass'):
                    inset = int(manual.get('inset_px', 0))
                    cv2.rectangle(
                        highlighted,
                        (x1 + inset, y1 + inset),
                        (max(x1 + inset, x2 - inset), max(y1 + inset, y2 - inset)),
                        (0, 180, 255), 3,
                    )

            st.image(
                highlighted,
                caption="Red: detected | Green: selected for erasure | Blue: manual",
                use_container_width=True,
            )

    # ── Tab 2: Preview & Export ────────────────────────────────────────────────
    with tab2:
        st.subheader("Preview & Export")
        st.write(
            "Click **✨ Stylize Current** or **✨ Stylize All** in the sidebar first, "
            "then fine-tune and download here."
        )

        col_ctrls, col_views = st.columns([1, 2])

        with col_ctrls:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Fine Tuning")
            brightness_val = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
            contrast_val = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
            blend_mode_val = st.selectbox(
                "Blend Mode",
                ["Multiply", "Overlay", "Screen", "Hard Light"],
                key=f"blend_mode_{active_name}",
                help=(
                    "How color layers mix with the B&W lines. "
                    "Neon Glow auto-switches to Replace regardless."
                ),
            )
            st.session_state['blend_mode'] = blend_mode_val
            st.caption("Hue shift & mirror are set in the sidebar.")
            st.markdown("</div>", unsafe_allow_html=True)

        final_blended = build_final_image(
            page,
            brightness=brightness_val,
            contrast=contrast_val,
            hue_degrees=st.session_state.get('export_hue', 0),
            mirror=st.session_state.get('export_mirror', False),
            watermark_cfg={
                'text': st.session_state.get('export_watermark_text', ''),
                'position': st.session_state.get('export_watermark_position', 'Bottom Right'),
                'opacity': st.session_state.get('export_watermark_opacity', 0.28),
                'scale': st.session_state.get('export_watermark_scale', 1.0),
            },
            blend_mode=blend_mode_val,
            ca_shift=st.session_state.get('export_ca_shift', 0),
            ca_direction=st.session_state.get('export_ca_dir', 'horizontal'),
        )

        with col_views:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.image(page['rgb'], caption="Original", use_container_width=True)
            with col_v2:
                st.image(final_blended, caption="Stylized Output", use_container_width=True)

            out_pil = Image.fromarray(final_blended)
            out_bytes = pil_to_bytes(out_pil, format="PNG")
            st.download_button(
                "Download Current Page 📥",
                data=out_bytes,
                file_name=f"stylized_{active_display_name}",
                mime="image/png",
            )
else:
    st.info("Please upload manga panels in the sidebar to start styling.")