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
from core.colorizer import apply_color_zone
from core.blender import blend_layers

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
            st.session_state.manga_pages[filename]['erased_bubbles'] = list(src_data['erased_bubbles'])
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
                    
                    # Auto-detect speech bubbles
                    detected_bubbles = detect_speech_bubbles(gray)
                    
                    st.session_state.manga_pages[file.name] = {
                        'rgb': rgb,
                        'gray': gray,
                        'pil': pil,
                        'bubbles': detected_bubbles,
                        'erased_bubbles': [], # Indicies of bubbles currently erased
                        'regions': [], # Colorized zones: { 'label', 'rect': (x1, y1, x2, y2), 'color' }
                        'geoms': [] # Geometric overlays: { 'rect': (x1, y1, x2, y2), 'color', 'opacity' }
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
        st.subheader("Batch Operations")
        if st.button("Copy Current Styling to All Pages"):
            copy_configs_to_all(st.session_state.active_file)
            
        # Export entire batch as ZIP
        st.subheader("Bulk Export")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for fname, p_data in st.session_state.manga_pages.items():
                # Generate final blended image for each page
                erased_img = erase_regions(p_data['rgb'], p_data['erased_bubbles'])
                
                # Apply color layers
                layers = []
                for reg in p_data['regions']:
                    layer = apply_color_zone(p_data['gray'], reg['color'], reg['rect'])
                    layers.append(layer)
                    
                blended = blend_layers(
                    erased_img, 
                    layers, 
                    geometric_shapes=p_data['geoms'],
                    brightness=1.0,
                    contrast=1.0
                )
                
                # Save to ZIP
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
        st.write("Tune detection parameters, click boxes to erase them, or draw manual areas.")
        
        # Parameter sliders
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            w_thresh = st.slider("White Brightness Threshold", 150, 255, 230, 5, key=f"w_thr_{active_name}")
        with col_s2:
            m_area = st.slider("Minimum Area Size (px)", 100, 20000, 1500, 100, key=f"m_ar_{active_name}")
        with col_s3:
            s_ratio = st.slider("Bubble Shape Solidity", 0.3, 1.0, 0.70, 0.05, key=f"s_ra_{active_name}")
            
        # Re-run detection dynamically based on sliders
        detected_bubbles = detect_speech_bubbles(
            page['gray'], 
            min_area=m_area, 
            solidity_thresh=s_ratio, 
            white_thresh=w_thresh
        )
        page['bubbles'] = detected_bubbles
        
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
                color = (0, 255, 0) if bubble in page['erased_bubbles'] else (255, 0, 0)
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
        st.subheader("Mondrian & Grid-style Color Blocks")
        st.write("Place colored rectangles to bypass automated copyright identification algorithms.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Add Color Block Overlay")
            
            # Palette selector
            series_choice = st.selectbox("Select Palette Theme", list(st.session_state.palettes.keys()))
            palette = st.session_state.palettes[series_choice]
            
            # Extract flat colors
            color_options = {}
            for name, color in palette.get("environment", {}).items():
                color_options[f"Env: {name}"] = color
            for char_name, char_cols in palette.get("characters", {}).items():
                for part, color in char_cols.items():
                    color_options[f"{char_name} - {part}"] = color
                    
            color_choice = st.selectbox("Select Color", list(color_options.keys()))
            custom_col = st.color_picker("Or Pick Custom Color", value="#4f46e5")
            
            # Convert chosen color to list
            if custom_col:
                # Hex to RGB
                rgb_custom = [int(custom_col[i:i+2], 16) for i in (1, 3, 5)]
            selected_rgb = rgb_custom if st.checkbox("Use Custom Color", value=False) else color_options[color_choice]
            
            gx = st.slider("X Position", 0, w, int(w * 0.1))
            gy = st.slider("Y Position", 0, h, int(h * 0.1))
            gw = st.slider("Block Width", 1, w, int(w * 0.3))
            gh = st.slider("Block Height", 1, h, int(h * 0.15))
            g_opacity = st.slider("Opacity", 0.0, 1.0, 0.45)
            
            if st.button("Add Geometric Overlay"):
                page['geoms'].append({
                    'rect': (gx, gy, gx + gw, gy + gh),
                    'color': selected_rgb,
                    'opacity': g_opacity
                })
                st.success("Overlay block added!")
                st.rerun()
                
            st.markdown("---")
            st.markdown("#### Current Overlays")
            for idx, g in enumerate(page['geoms']):
                st.write(f"#{idx+1} - Color: {g['color']} - Opacity: {g['opacity']}")
                if st.button(f"Delete Block #{idx+1}", key=f"del_g_{active_name}_{idx}"):
                    page['geoms'].pop(idx)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col1:
            # Draw current overlays on a preview copy
            preview_img = page['rgb'].copy()
            for g in page['geoms']:
                x1, y1, x2, y2 = g['rect']
                overlay = preview_img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(g['color']), -1)
                cv2.addWeighted(overlay, g['opacity'], preview_img, 1.0 - g['opacity'], 0, preview_img)
                cv2.rectangle(preview_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            st.image(preview_img, caption="Geometric Overlays Preview", use_container_width=True)

    # ---------------- TAB 3: REGION COLORING ----------------
    with tab3:
        st.subheader("Apply Flat Palette Colors to Character Regions")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Define Color Target Region")
            
            series_choice_col = st.selectbox("Select Palette Series", list(st.session_state.palettes.keys()), key="palette_col")
            palette_col = st.session_state.palettes[series_choice_col]
            
            # Choose specific character/environment coloring
            color_choices = {}
            for name, color in palette_col.get("environment", {}).items():
                color_choices[f"Env - {name}"] = color
            for char_name, char_cols in palette_col.get("characters", {}).items():
                for part, color in char_cols.items():
                    color_choices[f"{char_name} ({part})"] = color
                    
            color_key = st.selectbox("Character/Part Color", list(color_choices.keys()))
            region_color = color_choices[color_key]
            
            st.markdown("**Select Bounding Box Coordinates:**")
            rx = st.slider("X Start", 0, w, 0, key="rx")
            ry = st.slider("Y Start", 0, h, 0, key="ry")
            rw = st.slider("Width Slider", 1, w, int(w * 0.5), key="rw")
            rh = st.slider("Height Slider", 1, h, int(h * 0.5), key="rh")
            
            if st.button("Add Colored Region"):
                page['regions'].append({
                    'label': color_key,
                    'rect': (rx, ry, rx + rw, ry + rh),
                    'color': region_color
                })
                st.success(f"Colored region '{color_key}' added!")
                st.rerun()
                
            st.markdown("---")
            st.markdown("#### Current Regions")
            for idx, r in enumerate(page['regions']):
                st.write(f"#{idx+1} {r['label']} - Rect: {r['rect']}")
                if st.button(f"Delete Region #{idx+1}", key=f"del_r_{active_name}_{idx}"):
                    page['regions'].pop(idx)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col1:
            # Render a visualization of the region outlines
            reg_preview = page['rgb'].copy()
            for r in page['regions']:
                x1, y1, x2, y2 = r['rect']
                cv2.rectangle(reg_preview, (x1, y1), (x2, y2), tuple(r['color']), 3)
                cv2.putText(
                    reg_preview, 
                    r['label'], 
                    (x1 + 10, y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2,
                    cv2.LINE_AA
                )
            # Draw currently selected boundaries
            cv2.rectangle(reg_preview, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            
            st.image(reg_preview, caption="Defined Color Zones (Red: Current Slider bounds)", use_container_width=True)

    # ---------------- TAB 4: REVIEW & EXPORT ----------------
    with tab4:
        st.subheader("Final Review & Tuning")
        
        # Blending settings
        col_ctrls, col_views = st.columns([1, 2])
        
        with col_ctrls:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("#### Blend Fine Tuning")
            brightness_val = st.slider("Final Brightness Boost", 0.5, 2.0, 1.0, 0.05)
            contrast_val = st.slider("Final Contrast Boost", 0.5, 2.0, 1.0, 0.05)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Run final process
        erased_img = erase_regions(page['rgb'], page['erased_bubbles'])
        
        color_layers = []
        for reg in page['regions']:
            layer = apply_color_zone(page['gray'], reg['color'], reg['rect'])
            color_layers.append(layer)
            
        final_blended = blend_layers(
            erased_img,
            color_layers,
            geometric_shapes=page['geoms'],
            brightness=brightness_val,
            contrast=contrast_val
        )
        
        with col_views:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.image(page['rgb'], caption="Original Image", use_container_width=True)
            with col_v2:
                st.image(final_blended, caption="Stylized Page Output", use_container_width=True)
                
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
