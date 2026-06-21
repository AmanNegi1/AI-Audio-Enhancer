import cv2
import numpy as np


COLORING_MODES = [
    "Palette",
    "Duotone",
    "Sepia",
    "Sunset Gradient",
    "Posterize",
    "Flat Color",      # Otsu hard-fill — removes all screentone texture
    "Halftone",        # real coloured dot halftone pattern
    "Neon Glow",       # dark background + neon-lit ink lines
    "Zone Block",      # auto-detects content-rich cells, fills with warm rectangles
    "Gradient Map",    # maps every grayscale value to a user-defined colour ramp
]


def apply_color_zone(img_gray, color, rect=None):
    """
    Creates a colored mask layer for a specific region.
    color: list of RGB [R, G, B]
    rect: (x1, y1, x2, y2) bounds. If None, applies to entire image.
    """
    h, w = img_gray.shape
    color_layer = np.ones((h, w, 3), dtype=np.float32) * 255.0  # start with white
    
    # Slice the region we care about
    if rect:
        x1, y1, x2, y2 = rect
        # Clamp bounds
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        sub_gray = img_gray[y1:y2, x1:x2]
    else:
        sub_gray = img_gray
        x1, y1, x2, y2 = 0, 0, w, h
        
    # Convert base color to float 0-1
    base_color = np.array(color, dtype=np.float32) / 255.0

    # Paper (pure-white background) keeps almost white so the page background
    # doesn't turn solid-colored.  Only a whisper of tint is added so the
    # perceptual hash is still broken.
    paper_hint     = np.clip(base_color * 0.08 + 0.92, 0.0, 1.0)
    # Screentone (halftone gray) gets the full palette color — this is where
    # the coloring is most visible and most natural in manga.
    screentone_color = base_color
    hatching_color   = base_color * 0.70   # darker shade on heavy cross-hatching

    # Create the local color slice
    local_color = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32)

    # Create masks
    paper_mask      = sub_gray >= 220
    screentone_mask = (sub_gray < 220) & (sub_gray >= 130)
    hatching_mask   = (sub_gray < 130) & (sub_gray >= 70)
    ink_mask        = sub_gray < 70

    # Apply colors to region based on brightness zones
    local_color[paper_mask]      = paper_hint
    local_color[screentone_mask] = screentone_color
    local_color[hatching_mask]   = hatching_color
    local_color[ink_mask]        = [0.03, 0.03, 0.04]  # near black
    
    # Fill back into full color layer
    color_layer[y1:y2, x1:x2] = local_color * 255.0
    return color_layer


# ---------------------------------------------------------------------------
# Extra coloring modes
# ---------------------------------------------------------------------------

def _to01(color):
    return np.array(color, dtype=np.float32) / 255.0


def make_duotone_layer(img_gray, dark_color, bright_color):
    """
    Map grayscale brightness onto a smooth ramp between dark_color and
    bright_color, then convert back into the color-layer format the
    blender consumes (HxW float32 in 0..255).
    """
    h, w = img_gray.shape
    t = img_gray.astype(np.float32) / 255.0
    c0 = _to01(dark_color)
    c1 = _to01(bright_color)
    layer = c0[None, None, :] * (1.0 - t)[..., None] + c1[None, None, :] * t[..., None]
    return np.clip(layer * 255.0, 0, 255).astype(np.float32)


def make_sepia_layer(img_gray):
    """Classic warm sepia tone, useful for vintage manga vibes."""
    return make_duotone_layer(img_gray, [40, 22, 10], [255, 232, 200])


def make_gradient_layer(shape, top_color, bottom_color, direction="vertical"):
    """
    Build a full-page color layer that smoothly transitions between
    two palette colors. Direction can be 'vertical' or 'horizontal'.
    """
    h, w = shape[:2]
    if direction == "horizontal":
        t = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    else:
        t = np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w))
    c0 = _to01(top_color)
    c1 = _to01(bottom_color)
    layer = c0[None, None, :] * (1.0 - t)[..., None] + c1[None, None, :] * t[..., None]
    return np.clip(layer * 255.0, 0, 255).astype(np.float32)


def make_posterize_layer(img_gray, shadow_color, mid_color, highlight_color):
    """Hard 3-tone poster mapping."""
    h, w = img_gray.shape
    layer = np.ones((h, w, 3), dtype=np.float32)
    shadow = img_gray < 90
    mid = (img_gray >= 90) & (img_gray < 180)
    highlight = img_gray >= 180
    layer[shadow] = _to01(shadow_color)
    layer[mid] = _to01(mid_color)
    layer[highlight] = _to01(highlight_color)
    return np.clip(layer * 255.0, 0, 255).astype(np.float32)


def make_flat_color_layer(img_gray, base_color):
    """
    Hard flat fill: Otsu separates ink from paper; paper gets base_color,
    ink stays black.  All screentone texture is removed — the boldest visual
    change vs the original, closest to a proper comic re-colouring.
    """
    h, w = img_gray.shape
    ret, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    c = np.array(base_color, dtype=np.float32)
    layer = np.ones((h, w, 3), dtype=np.float32) * c
    layer[img_gray < ret] = [8.0, 8.0, 10.0]
    return np.clip(layer, 0, 255)


def make_halftone_layer(img_gray, dot_color, dot_spacing=10):
    """
    Replace screentones with actual coloured halftone dots.
    Dot radius at each grid point scales with local darkness
    (dark area → big dot, bright area → tiny/no dot).
    Fully vectorised — fast even on large pages.
    """
    h, w = img_gray.shape
    c = np.array(dot_color, dtype=np.float32)
    layer = np.ones((h, w, 3), dtype=np.float32) * 255.0

    spacing = max(4, int(dot_spacing))
    half = spacing // 2

    # Smooth so each grid cell has a stable local average brightness.
    smooth = cv2.GaussianBlur(img_gray, (spacing | 1, spacing | 1), 0)

    Y, X = np.mgrid[0:h, 0:w]
    cy = np.clip((Y // spacing) * spacing + half, 0, h - 1)
    cx = np.clip((X // spacing) * spacing + half, 0, w - 1)

    centre_brightness = smooth[cy, cx].astype(np.float32) / 255.0
    radius = (spacing * 0.50) * (1.0 - np.power(centre_brightness, 0.65))

    dist = np.sqrt(((Y - cy) ** 2 + (X - cx) ** 2).astype(np.float32))
    layer[dist <= radius] = c
    layer[img_gray < 45] = [8.0, 8.0, 10.0]   # hard ink always stays black
    return np.clip(layer, 0, 255)


def make_neon_glow_layer(img_gray, glow_color, glow_strength=0.8):
    """
    Dark-background neon effect: ink lines are relit as glowing neon colour,
    everything else is a deep dark background.
    build_final_image auto-applies blend_mode='Replace' when this mode is active.
    """
    h, w = img_gray.shape
    c = np.array(glow_color, dtype=np.float32)
    layer = np.full((h, w, 3), [10.0, 10.0, 22.0], dtype=np.float32)

    ink_f = (img_gray.astype(np.float32) < 70).astype(np.float32)
    sigma = 2.5 + float(glow_strength) * 5.0
    bloom = np.clip(cv2.GaussianBlur(ink_f, (0, 0), sigmaX=sigma), 0.0, 1.0)

    alpha = np.clip(ink_f + bloom * float(glow_strength) * 0.7, 0.0, 1.0)[..., None]
    return np.clip(layer * (1.0 - alpha) + c * alpha, 0, 255).astype(np.float32)


def make_gradient_map_layer(img_gray, stops):
    """
    Gradient Map: maps every grayscale brightness value to an interpolated
    colour across user-defined stops.  Each stop is [position_0_to_1, [R,G,B]].
    Equivalent to Photoshop's Gradient Map adjustment — every pixel changes
    non-linearly, making it one of the most effective perceptual-hash breakers.

    Uses Replace blend mode automatically so the full ramp is preserved
    (ink lines stay black via the blender's ink mask).
    """
    h, w = img_gray.shape
    t = img_gray.astype(np.float32) / 255.0   # brightness 0..1

    sorted_stops = sorted(stops, key=lambda s: s[0])
    positions = np.array([s[0] for s in sorted_stops], dtype=np.float32)
    colours   = np.array([s[1] for s in sorted_stops], dtype=np.float32) / 255.0

    result = np.zeros((h, w, 3), dtype=np.float32)

    # Below first stop → first colour
    result[t <= positions[0]] = colours[0]
    # Above last stop → last colour
    result[t >= positions[-1]] = colours[-1]

    # Interpolate between adjacent stops
    for i in range(len(sorted_stops) - 1):
        p0, p1 = float(positions[i]), float(positions[i + 1])
        if p1 <= p0:
            continue
        c0, c1 = colours[i], colours[i + 1]
        band = (t > p0) & (t < p1)
        local_t = (t[band] - p0) / (p1 - p0)
        result[band] = c0 * (1.0 - local_t[:, None]) + c1 * local_t[:, None]

    return np.clip(result * 255.0, 0, 255).astype(np.float32)


def make_zone_block_layer(img_gray, block_color, density_thresh=0.08,
                          grid_rows=8, grid_cols=6):
    """
    Auto-detects content-rich rectangular zones and fills them with block_color.
    The image is split into a grid; any cell whose screentone+ink density exceeds
    density_thresh gets painted.  Adjacent filled cells merge visually into larger
    blocks — ideal for the warm peach/salmon copyright-bypass aesthetic.
    """
    h, w = img_gray.shape
    layer = np.ones((h, w, 3), dtype=np.float32) * 255.0  # white = transparent to blender
    c = np.array(block_color, dtype=np.float32)

    rows = max(2, int(grid_rows))
    cols = max(2, int(grid_cols))
    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)

    for r in range(rows):
        for ci in range(cols):
            y1 = r * cell_h
            y2 = min(h, (r + 1) * cell_h)
            x1 = ci * cell_w
            x2 = min(w, (ci + 1) * cell_w)
            cell = img_gray[y1:y2, x1:x2]
            # Content = screentone / hatching (not pure white, not solid ink)
            content_frac = float(((cell > 20) & (cell < 235)).mean())
            if content_frac >= density_thresh:
                layer[y1:y2, x1:x2] = c

    return layer


def build_color_layer(mode, img_gray, region):
    """
    Dispatch a region dict to the right layer generator.
    `region` always carries 'mode' and 'rect'; the rest depends on mode.
    """
    if mode == "Palette":
        return apply_color_zone(img_gray, region['color'], region['rect'])
    if mode == "Duotone":
        return make_duotone_layer(img_gray, region['dark'], region['bright'])
    if mode == "Sepia":
        return make_sepia_layer(img_gray)
    if mode == "Sunset Gradient":
        return make_gradient_layer(
            img_gray.shape, region['top'], region['bottom'],
            direction=region.get('direction', 'vertical'),
        )
    if mode == "Posterize":
        return make_posterize_layer(
            img_gray, region['shadow'], region['mid'], region['highlight'],
        )
    if mode == "Flat Color":
        return make_flat_color_layer(img_gray, region['color'])
    if mode == "Halftone":
        return make_halftone_layer(img_gray, region['dot_color'],
                                   dot_spacing=region.get('dot_spacing', 10))
    if mode == "Neon Glow":
        return make_neon_glow_layer(img_gray, region['glow_color'],
                                    glow_strength=region.get('glow_strength', 0.8))
    if mode == "Zone Block":
        return make_zone_block_layer(
            img_gray,
            region.get('block_color', [255, 185, 145]),
            density_thresh=region.get('density_thresh', 0.08),
            grid_rows=region.get('grid_rows', 8),
            grid_cols=region.get('grid_cols', 6),
        )
    if mode == "Gradient Map":
        return make_gradient_map_layer(img_gray, region.get('stops', [
            [0.0, [29, 17, 96]],
            [0.5, [196, 98, 45]],
            [1.0, [245, 230, 163]],
        ]))
    return apply_color_zone(img_gray, region.get('color', [128, 128, 128]), region.get('rect'))