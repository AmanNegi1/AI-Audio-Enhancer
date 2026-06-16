import numpy as np


COLORING_MODES = [
    "Palette",        # current behaviour: paper/screentone/hatching/ink bands
    "Duotone",        # dark color -> bright color ramp
    "Sepia",          # warm monochrome
    "Sunset Gradient",  # vertical color gradient
    "Posterize",      # hard 3-tone mapping
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
    
    # Calculate colors for different brightness thresholds
    paper_color = base_color
    screentone_color = base_color * 1.15
    hatching_color = base_color * 0.75
    
    # Create the local color slice
    local_color = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32)
    
    # Create masks
    paper_mask = sub_gray >= 220
    screentone_mask = (sub_gray < 220) & (sub_gray >= 130)
    hatching_mask = (sub_gray < 130) & (sub_gray >= 70)
    ink_mask = sub_gray < 70
    
    # Apply colors to region based on brightness zones
    local_color[paper_mask] = paper_color
    local_color[screentone_mask] = screentone_color
    local_color[hatching_mask] = hatching_color
    local_color[ink_mask] = [0.03, 0.03, 0.04]  # near black
    
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
    return apply_color_zone(img_gray, region.get('color', [128, 128, 128]), region.get('rect'))