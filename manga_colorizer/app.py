import streamlit as st
import numpy as np
import json
import zipfile
import io
import os
import cv2
from PIL import Image

# Import core modules
from core.loader import load_image, pil_to_bytes
from core.bubble_eraser import detect_speech_bubbles, erase_regions
from core.colorizer import apply_color_zone, build_color_layer, COLORING_MODES
from core.blender import blend_layers
from core.auto import (
    auto_geometric_overlays,
    auto_region_colors,
    auto_region_colors_by_panel,
    detect_panels,
    apply_hue_shift,
    apply_mirror,
    OVERLAY_TEMPLATES,
)
from core.splitters import apply_splitter, SPLITTER_TEMPLATES
from core.live_panel import (
    LIVE_PANEL_TEMPLATES,
    EFFECT_TEMPLATES,
    render_live_panel_frames,
    encode_gif_bytes,
)


def _strip_contour(bubble_dict):
    """Return a copy of a bubble dict without the per-image contour data."""
    return {k: v for k, v in bubble_dict.items() if k != 'contour'}


def auto_stylize_page(page_data, palette, template="Corner Accents",
                      intensity=0.5, color_by_panel=True):
    """Apply the full copyright-bypass pipeline to a single page in place."""
    bubbles = detect_speech_bubbles(page_data['gray'])
    page_data['bubbles'] = bubbles
    page_data['erased_bubbles'] = list(bubbles)

    panels = detect_panels(page_data['gray'])
    page_data['panels'] = panels

    page_data['geoms'] = auto_geometric_overlays(
        page_data['rgb'].shape, palette,
        template=template, intensity=intensity, panels=panels,
    )
    if color_by_panel:
        page_data['regions'] = auto_region_colors_by_panel(
            page_data['gray'], palette,
        )
    else:
        page_data['regions'] = auto_region_colors(page_data['rgb'].shape, palette)


def apply_text_watermark(img_rgb, watermark_cfg=None):
    """Draw a semi-transparent export watermark onto the final image."""
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
                      hue_degrees=0.0, mirror=False, watermark_cfg=None):
    """Run the erase + color + overlay + post-process pipeline."""
    erased = erase_regions(page_data['rgb'], page_data['erased_bubbles'])
    # Use a gray derived from the ERASED page so that text pixels don't
    # bake themselves into the colour layers (Duotone / Sepia / Posterize
    # were re-tinting the original text into the recoloured page).
    erased_gray = cv2.cvtColor(erased, cv2.COLOR_RGB2GRAY)
    layers = []
    for r in page_data['regions']:
        mode = r.get('mode', 'Palette')
        layers.append(build_color_layer(mode, erased_gray, r))
    blended = blend_layers(
        erased, layers,
        geometric_shapes=page_data['geoms'],
        brightness=brightness, contrast=contrast,
    )
    # Image splitter (physically slice/transform the rendered page)
    split_cfg = page_data.get('splitter') or {}
    split_tpl = split_cfg.get('template', 'None')
    if split_tpl and split_tpl != 'None':
        panels = page_data.get('panels') or detect_panels(page_data['gray'])
        page_data['panels'] = panels
        blended = apply_splitter(
            blended,
            template=split_tpl,
            intensity=split_cfg.get('intensity', 0.5),
            panels=panels,
            gap_color=split_cfg.get('gap_color', (255, 255, 255)),
        )
    if hue_degrees:
        blended = apply_hue_shift(blended, hue_degrees)
    if mirror:
        blended = apply_mirror(blended)
    blended = apply_text_watermark(blended, watermark_cfg)
    return blended

# Page settings
st.set_page_config(
    page_title="Manga Colorizer & Copyright Bypass Stylizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Dark Mode Glassmorphism Style */
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
    .stTab {
        background-color: transparent !important;
        border: none !important;
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

# ----------------- SESSION STATE INITS -----------------
if 'palettes' not in st.session_state:
    st.session_state.palettes = {}
    # Load Kagurabachi palette
    palette_path = os.path.join(os.path.dirname(__file__), "palettes", "kagurabachi.json")
    if os.path.exists(palette_path):
        try:
            with open(palette_path, "r") as f:
                data = json.load(f)
                st.session_state.palettes[data["series"]] = data
        except Exception as e:
            st.error(f"Failed to load palette: {e}")

if 'manga_pages' not in st.session_state:
    st.session_state.manga_pages = {}  # filename -> { 'rgb', 'gray', 'pil', 'bubbles', 'erased_bubbles', 'regions', 'geoms' }

if 'active_file' not in st.session_state:
    st.session_state.active_file = None

# Helper to copy configuration
def copy_configs_to_all(src_filename):
    if not src_filename or src_filename not in st.session_state.manga_pages:
        return
    src_data = st.session_state.manga_pages[src_filename]
    for filename in st.session_state.manga_pages:
        if filename != src_filename:
            # Copy regions and geoms
            st.session_state.manga_pages[filename]['regions'] = list(src_data['regions'])
            st.session_state.manga_pages[filename]['geoms'] = list(src_data['geoms'])
            # Strip contours - they reference the source image's pixel grid
            st.session_state.manga_pages[filename]['erased_bubbles'] = [
                _strip_contour(b) for b in src_data['erased_bubbles']
            ]
    st.success(f"Copied regions, geometric overlays, and bubble cleanups from '{src_filename}' to all other pages!")

# ----------------- MAIN SIDEBAR -----------------
with st.sidebar:
    st.title("🎨 Manga Stylizer")
    st.write("Clean text bubbles and apply copyright-bypass overlays to batches of manga panels.")
    
    # 1. Multi-upload files
    uploaded_files = st.file_uploader(
        "Upload Manga Pages (Multiple allowed)", 
        type=["png", "jpg", "jpeg", "webp"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.manga_pages:
                try:
                    file_bytes = file.read()
                    rgb, gray, pil = load_image(file_bytes)
                    
                    # Auto-detect speech bubbles (text-verified)
                    detected_bubbles = detect_speech_bubbles(gray)
                    
                    st.session_state.manga_pages[file.name] = {
                        'rgb': rgb,
                        'gray': gray,
                        'pil': pil,
                        'bubbles': detected_bubbles,
                        # Pre-select every detected bubble so the user
                        # gets a clean, text-free page out of the box.
                        'erased_bubbles': list(detected_bubbles),
                        'regions': [], # Colorized zones: { 'label', 'rect': (x1, y1, x2, y2), 'color' }
                        'geoms': [], # Geometric overlays: { 'rect': (x1, y1, x2, y2), 'color', 'opacity' }
                        'splitter': {'template': 'None', 'intensity': 0.5,
                                     'gap_color': (255, 255, 255)},
                        'live_panel': {
                            'template': 'None',
                            'strength': 0.45,
                            'fps': 12,
                            'frames': 18,
                            'feather': 20,
                            'subject_rect': (
                                int(rgb.shape[1] * 0.28),
                                int(rgb.shape[0] * 0.18),
                                int(rgb.shape[1] * 0.72),
                                int(rgb.shape[0] * 0.88),
                            ),
                        },
                        'panels': None,
                    }
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")
                    
        # Set active file if not set
        if not st.session_state.active_file and st.session_state.manga_pages:
            st.session_state.active_file = list(st.session_state.manga_pages.keys())[0]

    # Select Active Image
    if st.session_state.manga_pages:
        st.session_state.active_file = st.selectbox(
            "Select Page to Edit", 
            options=list(st.session_state.manga_pages.keys()),
            index=list(st.session_state.manga_pages.keys()).index(st.session_state.active_file) if st.session_state.active_file in st.session_state.manga_pages else 0
        )
        
        st.markdown("---")
        st.subheader("Smart Auto-Stylize")

        palette_names = list(st.session_state.palettes.keys())
        auto_palette_name = st.selectbox(
            "Palette",
            options=palette_names if palette_names else ["(no palette loaded)"],
            key="auto_palette",
        )
        auto_template = st.selectbox(
            "Overlay Template",
            options=OVERLAY_TEMPLATES,
            index=0,
            help="Layout the geometric overlays are placed in. No x/y needed.",
            key="auto_template",
        )
        auto_intensity = st.slider(
            "Style Intensity", 0.0, 1.0, 0.55, 0.05,
            help="Higher = larger / more opaque overlays.",
            key="auto_intensity",
        )
        auto_color_by_panel = st.checkbox(
            "Color each detected panel differently",
            value=True,
            help="Auto-detects manga panels and tints each one.",
            key="auto_color_by_panel",
        )

        col_auto_1, col_auto_2 = st.columns(2)
        with col_auto_1:
            if st.button("\u2728 Stylize Current") and palette_names:
                palette = st.session_state.palettes[auto_palette_name]
                auto_stylize_page(
                    st.session_state.manga_pages[st.session_state.active_file],
                    palette,
                    template=auto_template,
                    intensity=auto_intensity,
                    color_by_panel=auto_color_by_panel,
                )
                st.success("Stylized this page.")
                st.rerun()
        with col_auto_2:
            if st.button("\u2728 Stylize All") and palette_names:
                palette = st.session_state.palettes[auto_palette_name]
                for p in st.session_state.manga_pages.values():
                    auto_stylize_page(
                        p, palette,
                        template=auto_template,
                        intensity=auto_intensity,
                        color_by_panel=auto_color_by_panel,
                    )
                st.success("Stylized every page.")
                st.rerun()

        st.markdown("---")
        st.subheader("Hash-Breaker (Export)")
        st.session_state['export_hue'] = st.slider(
            "Hue Shift (\u00b0)", -30, 30,
            st.session_state.get('export_hue', 6), 1,
            help="Small hue rotation. Near-invisible but defeats perceptual hashes.",
        )
        st.session_state['export_mirror'] = st.checkbox(
            "Mirror horizontally on export",
            value=st.session_state.get('export_mirror', False),
            help="Strong perceptual-hash breaker. Note: text in art will flip.",
        )

        st.markdown("---")
        st.subheader("Watermark (Export)")
        st.session_state['export_watermark_text'] = st.text_input(
            "Watermark Text",
            value=st.session_state.get('export_watermark_text', ''),
            help="Optional text watermark applied to final PNG export, ZIP export, and animated GIF preview/export.",
        )
        st.session_state['export_watermark_position'] = st.selectbox(
            "Watermark Position",
            ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"],
            index=["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"].index(
                st.session_state.get('export_watermark_position', 'Bottom Right')
            ) if st.session_state.get('export_watermark_position', 'Bottom Right') in ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"] else 0,
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

        st.markdown("---")
        if st.button("Copy Current Styling to All Pages"):
            copy_configs_to_all(st.session_state.active_file)
            
        # Export entire batch as ZIP
        st.subheader("Bulk Export")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for fname, p_data in st.session_state.manga_pages.items():
                blended = build_final_image(
                    p_data,
                    hue_degrees=st.session_state.get('export_hue', 0),
                    mirror=st.session_state.get('export_mirror', False),
                    watermark_cfg=watermark_cfg,
                )
                pil_to_save = Image.fromarray(blended)
                img_bytes = pil_to_bytes(pil_to_save, format="PNG")
                zip_file.writestr(f"stylized_{fname}", img_bytes)
                
        st.download_button(
            "Download All Pages as ZIP 📦",
            data=zip_buffer.getvalue(),
            file_name="manga_stylized_batch.zip",
            mime="application/zip"
        )
    else:
        st.info("Upload one or more manga pages to get started.")

# ----------------- MAIN WINDOW -----------------
if st.session_state.active_file:
    active_name = st.session_state.active_file
    page = st.session_state.manga_pages[active_name]
    
    h, w, _ = page['rgb'].shape
    st.title(f"Editing: `{active_name}` ({w}x{h} px)")
    
    # Render UI Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Bubble Eraser", 
        "🟩 Geometric Overlays (Copyright Bypass)", 
        "🖌️ Region Coloring", 
        "💾 Review & Export"
    ])
    
    # ---------------- TAB 1: BUBBLE ERASER ----------------
    with tab1:
        st.subheader("Speech Bubble Eraser")
        st.write(
            "Detection finds text pixels and erases only those pixels. "
            "Press 'Re-detect' after changing sliders."
        )

        # Parameter sliders (advanced)
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
                help="Raise this to reject face details, eyes, and other non-text artwork. Use manual erase for very short text.",
            )

        col_act_1, col_act_2 = st.columns(2)
        with col_act_1:
            redetect = st.button("Re-detect bubbles on this page", key=f"redet_{active_name}")
        with col_act_2:
            redetect_all = st.button("Re-detect on ALL pages", key="redet_all")

        if redetect:
            page['bubbles'] = detect_speech_bubbles(
                page['gray'],
                min_area=m_area,
                white_thresh=w_thresh,
                min_text_components=min_chars,
            )
            page['erased_bubbles'] = list(page['bubbles'])
            st.rerun()
        if redetect_all:
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
                    st.rerun()
                if st.button("Deselect All"):
                    page['erased_bubbles'] = []
                    st.rerun()
                    
                # We show checkbox matching index or coordinates
                for idx, bubble in enumerate(page['bubbles']):
                    label = f"Bubble #{idx+1} ({bubble['x']}, {bubble['y']}) - Area: {int(bubble['area'])}"
                    # Check if bubble coordinates are already in erased_bubbles
                    is_erased = any(b['x'] == bubble['x'] and b['y'] == bubble['y'] for b in page['erased_bubbles'])
                    
                    if st.checkbox(label, value=is_erased, key=f"bub_{active_name}_{idx}_{bubble['x']}_{bubble['y']}"):
                        if not is_erased:
                            page['erased_bubbles'].append(bubble)
                    else:
                        if is_erased:
                            # Remove it
                            page['erased_bubbles'] = [b for b in page['erased_bubbles'] if not (b['x'] == bubble['x'] and b['y'] == bubble['y'])]
            else:
                st.info("No speech bubbles automatically detected. Try lowering White Threshold or Solidity.")
                
            st.markdown("#### Manual Erase Coordinate")
            mx = st.number_input("X Coord", 0, w, 0)
            my = st.number_input("Y Coord", 0, h, 0)
            mw = st.number_input("Width", 1, w, 100)
            mh = st.number_input("Height", 1, h, 80)
            if st.button("Erase Manual Area"):
                page['erased_bubbles'].append({'x': mx, 'y': my, 'w': mw, 'h': mh})
                st.success("Manual eraser rectangle added!")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col1:
            # Draw highlight borders on the bubbles
            highlighted = page['rgb'].copy()
            for idx, bubble in enumerate(page['bubbles']):
                is_erased = any(
                    b['x'] == bubble['x'] and b['y'] == bubble['y']
                    for b in page['erased_bubbles']
                )
                color = (0, 255, 0) if is_erased else (255, 0, 0)
                # Draw rect
                cv2.rectangle(
                    highlighted, 
                    (bubble['x'], bubble['y']), 
                    (bubble['x'] + bubble['w'], bubble['y'] + bubble['h']), 
                    color, 
                    4
                )
                cv2.putText(
                    highlighted, 
                    str(idx+1), 
                    (bubble['x'] + 5, bubble['y'] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    color, 
                    2
                )
            
            # Show image
            st.image(highlighted, caption="Red: Detected Bubbles | Green: Selected for Erasure", use_container_width=True)
            
    # ---------------- TAB 2: GEOMETRIC OVERLAYS ----------------
    with tab2:
        st.subheader("Smart Geometric Overlays")
        st.write(
            "Pick a layout template and intensity; the app places blocks for you. "
            "No coordinates required."
        )

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Apply Overlay Template")

            tab2_palette_name = st.selectbox(
                "Palette",
                list(st.session_state.palettes.keys()),
                key="tab2_palette",
            )
            tab2_palette = st.session_state.palettes[tab2_palette_name]

            tab2_template = st.selectbox(
                "Template", OVERLAY_TEMPLATES, key="tab2_template",
            )
            tab2_intensity = st.slider(
                "Intensity", 0.0, 1.0, 0.55, 0.05, key="tab2_intensity",
            )

            replace_existing = st.checkbox(
                "Replace existing overlays", value=True, key="tab2_replace",
            )

            if st.button("Apply Template to This Page"):
                panels = page.get('panels') or detect_panels(page['gray'])
                page['panels'] = panels
                new_overlays = auto_geometric_overlays(
                    page['rgb'].shape, tab2_palette,
                    template=tab2_template, intensity=tab2_intensity,
                    panels=panels,
                )
                if replace_existing:
                    page['geoms'] = new_overlays
                else:
                    page['geoms'].extend(new_overlays)
                st.success(f"Applied '{tab2_template}'.")
                st.rerun()

            if st.button("Apply Template to ALL Pages"):
                for p in st.session_state.manga_pages.values():
                    panels = p.get('panels') or detect_panels(p['gray'])
                    p['panels'] = panels
                    new_overlays = auto_geometric_overlays(
                        p['rgb'].shape, tab2_palette,
                        template=tab2_template, intensity=tab2_intensity,
                        panels=panels,
                    )
                    if replace_existing:
                        p['geoms'] = new_overlays
                    else:
                        p['geoms'].extend(new_overlays)
                st.success(f"Applied '{tab2_template}' to all pages.")
                st.rerun()

            if st.button("Clear All Overlays on This Page"):
                page['geoms'] = []
                st.rerun()

            st.markdown("---")
            st.markdown("#### Image Splitter (slices the actual image)")

            # Ensure splitter dict exists (older session pages may lack it).
            page.setdefault('splitter', {
                'template': 'None', 'intensity': 0.5,
                'gap_color': (255, 255, 255),
            })

            split_tpl = st.selectbox(
                "Splitter Template",
                SPLITTER_TEMPLATES,
                index=SPLITTER_TEMPLATES.index(page['splitter'].get('template', 'None'))
                    if page['splitter'].get('template') in SPLITTER_TEMPLATES else 0,
                key="tab2_split_tpl",
                help=("Physically slices the rendered page and shifts the pieces. "
                      "'Panel Pop-out' uses the manga's own detected panels."),
            )
            split_int = st.slider(
                "Splitter Intensity", 0.0, 1.0,
                float(page['splitter'].get('intensity', 0.5)), 0.05,
                key="tab2_split_int",
            )
            gap_choice = st.selectbox(
                "Gap Color",
                ["White", "Black", "First Palette Color"],
                key="tab2_split_gap",
            )
            if gap_choice == "Black":
                gap_color = (10, 10, 10)
            elif gap_choice == "First Palette Color":
                pal_cols = []
                for c in tab2_palette.get("environment", {}).values():
                    pal_cols.append(c)
                for ch in tab2_palette.get("characters", {}).values():
                    for c in ch.values():
                        pal_cols.append(c)
                gap_color = tuple(pal_cols[0]) if pal_cols else (255, 255, 255)
            else:
                gap_color = (255, 255, 255)

            if st.button("Apply Splitter to This Page"):
                page['splitter'] = {
                    'template': split_tpl,
                    'intensity': split_int,
                    'gap_color': gap_color,
                }
                if page.get('panels') is None:
                    page['panels'] = detect_panels(page['gray'])
                st.success(f"Splitter set to '{split_tpl}'.")
                st.rerun()

            if st.button("Apply Splitter to ALL Pages"):
                for p in st.session_state.manga_pages.values():
                    p['splitter'] = {
                        'template': split_tpl,
                        'intensity': split_int,
                        'gap_color': gap_color,
                    }
                    if p.get('panels') is None:
                        p['panels'] = detect_panels(p['gray'])
                st.success(f"Splitter '{split_tpl}' set on all pages.")
                st.rerun()

            if st.button("Clear Splitter on This Page"):
                page['splitter'] = {
                    'template': 'None', 'intensity': 0.5,
                    'gap_color': (255, 255, 255),
                }
                st.rerun()

            current_split = page['splitter'].get('template', 'None')
            st.caption(f"Active splitter: **{current_split}**")

            st.markdown("---")
            st.markdown("#### Current Overlays")
            for idx, g in enumerate(page['geoms']):
                st.write(f"#{idx+1} - Color: {g['color']} - Opacity: {round(g['opacity'], 2)}")
                if st.button(f"Delete Block #{idx+1}", key=f"del_g_{active_name}_{idx}"):
                    page['geoms'].pop(idx)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col1:
            # Draw current overlays on a preview copy (handles both rect and mask shapes)
            preview_img = page['rgb'].copy()
            h_pv, w_pv = preview_img.shape[:2]
            for g in page['geoms']:
                color = tuple(int(c) for c in g.get('color', [255, 255, 255]))
                opacity = float(g.get('opacity', 0.5))
                if 'mask' in g and g['mask'] is not None:
                    m = g['mask']
                    if m.shape[:2] != (h_pv, w_pv):
                        m = cv2.resize(m, (w_pv, h_pv), interpolation=cv2.INTER_LINEAR)
                    alpha = (m.astype(np.float32) / 255.0) * opacity
                    alpha = alpha[..., None]
                    color_layer = np.zeros_like(preview_img, dtype=np.float32)
                    color_layer[:] = color
                    preview_img = (preview_img.astype(np.float32) * (1.0 - alpha) +
                                   color_layer * alpha).astype(np.uint8)
                elif 'rect' in g:
                    x1, y1, x2, y2 = g['rect']
                    overlay = preview_img.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, opacity, preview_img, 1.0 - opacity, 0, preview_img)
                    cv2.rectangle(preview_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # Apply splitter to preview so user sees the slice effect live
            split_cfg = page.get('splitter') or {}
            if split_cfg.get('template') and split_cfg['template'] != 'None':
                if page.get('panels') is None:
                    page['panels'] = detect_panels(page['gray'])
                preview_img = apply_splitter(
                    preview_img,
                    template=split_cfg['template'],
                    intensity=split_cfg.get('intensity', 0.5),
                    panels=page.get('panels'),
                    gap_color=split_cfg.get('gap_color', (255, 255, 255)),
                )

            st.image(preview_img, caption="Geometric Overlays Preview", use_container_width=True)

    # ---------------- TAB 3: REGION COLORING ----------------
    with tab3:
        st.subheader("Smart Region Coloring")
        st.write(
            "Pick a colouring style. Palette/Panel mode follows the manga's own brightness bands; "
            "Duotone, Sepia, Sunset Gradient and Posterize re-tone the whole page."
        )

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Apply Coloring")

            tab3_palette_name = st.selectbox(
                "Palette",
                list(st.session_state.palettes.keys()),
                key="tab3_palette",
            )
            tab3_palette = st.session_state.palettes[tab3_palette_name]

            # Build a flat color picker once for any mode that needs picks.
            flat_colors = {}
            for name, c in tab3_palette.get("environment", {}).items():
                flat_colors[f"Env: {name}"] = c
            for ch_name, ch_cols in tab3_palette.get("characters", {}).items():
                for part, c in ch_cols.items():
                    flat_colors[f"{ch_name} - {part}"] = c
            color_keys = list(flat_colors.keys()) or ["(empty)"]

            style = st.selectbox(
                "Style",
                ["Palette (per panel)", "Palette (whole page)"] + [
                    m for m in COLORING_MODES if m not in ("Palette",)
                ],
                key="tab3_style",
                help="Choose how the page should be re-coloured.",
            )

            extra_args = {}
            if style == "Duotone":
                d_key = st.selectbox("Dark color", color_keys, key="duo_dark")
                b_key = st.selectbox("Bright color", color_keys, index=min(1, len(color_keys) - 1), key="duo_bright")
                extra_args = {'dark': flat_colors.get(d_key, [20, 20, 60]),
                              'bright': flat_colors.get(b_key, [240, 220, 180])}
            elif style == "Sunset Gradient":
                t_key = st.selectbox("Top color", color_keys, key="grad_top")
                b_key = st.selectbox("Bottom color", color_keys, index=min(1, len(color_keys) - 1), key="grad_bot")
                direction = st.radio("Direction", ["vertical", "horizontal"], key="grad_dir")
                extra_args = {'top': flat_colors.get(t_key, [255, 180, 90]),
                              'bottom': flat_colors.get(b_key, [60, 30, 90]),
                              'direction': direction}
            elif style == "Posterize":
                s_key = st.selectbox("Shadow", color_keys, key="post_shad")
                m_key = st.selectbox("Midtone", color_keys, index=min(1, len(color_keys) - 1), key="post_mid")
                h_key = st.selectbox("Highlight", color_keys, index=min(2, len(color_keys) - 1), key="post_hi")
                extra_args = {'shadow': flat_colors.get(s_key, [30, 30, 60]),
                              'mid': flat_colors.get(m_key, [180, 80, 80]),
                              'highlight': flat_colors.get(h_key, [250, 240, 220])}

            def _build_regions(p):
                h_, w_ = p['gray'].shape
                if style == "Palette (per panel)":
                    return auto_region_colors_by_panel(p['gray'], tab3_palette)
                if style == "Palette (whole page)":
                    return auto_region_colors(p['rgb'].shape, tab3_palette)
                # Single full-page region in non-palette mode
                region = {'mode': style, 'label': style, 'rect': (0, 0, w_, h_)}
                region.update(extra_args)
                return [region]

            if st.button("Apply to This Page"):
                page['regions'] = _build_regions(page)
                st.success(f"Applied '{style}'.")
                st.rerun()

            if st.button("Apply to ALL Pages"):
                for p in st.session_state.manga_pages.values():
                    p['regions'] = _build_regions(p)
                st.success(f"Applied '{style}' to all pages.")
                st.rerun()

            if st.button("Clear All Regions on This Page"):
                page['regions'] = []
                st.rerun()

            st.markdown("---")
            st.markdown("#### Current Regions")
            for idx, r in enumerate(page['regions']):
                lbl = r.get('label', r.get('mode', 'region'))
                col_disp = r.get('color') or r.get('dark') or r.get('top') or r.get('shadow')
                st.write(f"#{idx+1} {lbl} - {col_disp if col_disp else ''}")
                if st.button(f"Delete Region #{idx+1}", key=f"del_r_{active_name}_{idx}"):
                    page['regions'].pop(idx)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col1:
            reg_preview = page['rgb'].copy()
            for r in page['regions']:
                rect = r.get('rect')
                if not rect:
                    continue
                x1, y1, x2, y2 = rect
                outline = r.get('color') or r.get('dark') or r.get('top') or r.get('shadow') or [255, 0, 0]
                cv2.rectangle(reg_preview, (x1, y1), (x2, y2), tuple(outline), 3)
                cv2.putText(
                    reg_preview, r.get('label', r.get('mode', '')),
                    (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
            st.image(reg_preview, caption="Color zones", use_container_width=True)

    # ---------------- TAB 4: REVIEW & EXPORT ----------------
    with tab4:
        st.subheader("Final Review & Tuning")

        page.setdefault('live_panel', {
            'template': 'None',
            'effect_template': 'None',
            'strength': 0.45,
            'effect_strength': 0.45,
            'fps': 12,
            'frames': 18,
        })
        
        # Blending settings
        col_ctrls, col_views = st.columns([1, 2])
        
        with col_ctrls:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Blend Fine Tuning")
            brightness_val = st.slider("Final Brightness Boost", 0.5, 2.0, 1.0, 0.05)
            contrast_val = st.slider("Final Contrast Boost", 0.5, 2.0, 1.0, 0.05)
            st.caption(
                "Hue shift & mirror are set in the sidebar and applied at export."
            )
            st.markdown("---")
            st.markdown("#### Smart Animation")

            live_cfg = page['live_panel']
            live_template = st.selectbox(
                "Animation Style",
                LIVE_PANEL_TEMPLATES,
                index=(LIVE_PANEL_TEMPLATES.index(live_cfg.get('template', 'None'))
                       if live_cfg.get('template') in LIVE_PANEL_TEMPLATES else 0),
                help="Whole-image animation templates. Smart Auto reuses the current splitter style when possible.",
                key=f"live_tpl_{active_name}",
            )
            live_strength = st.slider(
                "Motion Strength", 0.0, 1.0,
                float(live_cfg.get('strength', 0.45)), 0.05,
                key=f"live_strength_{active_name}",
            )
            effect_template = st.selectbox(
                "Effect Template",
                EFFECT_TEMPLATES,
                index=(EFFECT_TEMPLATES.index(live_cfg.get('effect_template', 'None'))
                       if live_cfg.get('effect_template') in EFFECT_TEMPLATES else 0),
                help="Premium visual finish. Can be used by itself or combined with motion.",
                key=f"live_effect_{active_name}",
            )
            effect_strength = st.slider(
                "Effect Strength", 0.0, 1.0,
                float(live_cfg.get('effect_strength', 0.45)), 0.05,
                key=f"live_effect_strength_{active_name}",
            )
            live_fps = st.slider(
                "GIF FPS", 6, 20,
                int(live_cfg.get('fps', 12)), 1,
                key=f"live_fps_{active_name}",
            )
            live_frames = st.slider(
                "Loop Frames", 8, 36,
                int(live_cfg.get('frames', 18)), 2,
                key=f"live_frames_{active_name}",
            )
            active_splitter = (page.get('splitter') or {}).get('template', 'None')
            st.caption(
                f"Current splitter base: **{active_splitter}**. You can use motion only, effect only, or both together. Splitter Drift and Smart Auto reuse the current splitter style."
            )

            if st.button("Apply Live Panel Settings", key=f"live_apply_{active_name}"):
                page['live_panel'] = {
                    'template': live_template,
                    'effect_template': effect_template,
                    'strength': live_strength,
                    'effect_strength': effect_strength,
                    'fps': live_fps,
                    'frames': live_frames,
                }
                st.success(f"Animation '{live_template}' + effect '{effect_template}' applied.")
                st.rerun()

            if st.button("Clear Live Panel", key=f"live_clear_{active_name}"):
                page['live_panel']['template'] = 'None'
                st.rerun()

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
        )
        
        with col_views:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.image(page['rgb'], caption="Original Image", use_container_width=True)
            with col_v2:
                st.image(final_blended, caption="Stylized Page Output", use_container_width=True)

            live_cfg = page.get('live_panel') or {}
            live_template = live_cfg.get('template', 'None')
            live_effect_template = live_cfg.get('effect_template', 'None')
            if live_template != 'None' or live_effect_template != 'None':
                panels = page.get('panels') or detect_panels(page['gray'])
                page['panels'] = panels

                live_frames = render_live_panel_frames(
                    final_blended,
                    template=live_template,
                    strength=live_cfg.get('strength', 0.45),
                    frame_count=live_cfg.get('frames', 18),
                    panels=panels,
                    splitter_template=(page.get('splitter') or {}).get('template', 'None'),
                    gap_color=(page.get('splitter') or {}).get('gap_color', (255, 255, 255)),
                    effect_template=live_cfg.get('effect_template', 'None'),
                    effect_strength=live_cfg.get('effect_strength', 0.45),
                )
                live_gif = encode_gif_bytes(live_frames, fps=live_cfg.get('fps', 12))
                st.image(live_gif, caption="Animated preview", use_container_width=True)
                st.download_button(
                    "Download Animated GIF 🎞️",
                    data=live_gif,
                    file_name=f"animated_{os.path.splitext(active_name)[0]}.gif",
                    mime="image/gif",
                )
                
            # Individual download button
            out_pil = Image.fromarray(final_blended)
            out_bytes = pil_to_bytes(out_pil, format="PNG")
            
            st.download_button(
                "Download Current Page 📥",
                data=out_bytes,
                file_name=f"stylized_{active_name}",
                mime="image/png"
            )
else:
    st.info("Please upload manga panels in the sidebar to start styling.")