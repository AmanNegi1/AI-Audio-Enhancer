"""
Auto-pipeline helpers: derive geometric overlays and region coloring
from the active palette and the page's own panel layout, so the user
can stylize a whole batch with one click (copyright-bypass friendly).
"""

import math
from typing import List, Dict, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Palette utilities
# ---------------------------------------------------------------------------

def _palette_colors(palette: dict) -> List[List[int]]:
    """Flatten a palette dict into an ordered list of RGB colors."""
    colors: List[List[int]] = []
    for color in palette.get("environment", {}).values():
        colors.append(list(color))
    for char_cols in palette.get("characters", {}).values():
        for color in char_cols.values():
            colors.append(list(color))
    if not colors:
        colors = [[80, 80, 200], [200, 100, 100], [100, 200, 100], [220, 200, 100]]
    return colors


# ---------------------------------------------------------------------------
# Panel detection (manga page layout)
# ---------------------------------------------------------------------------

def detect_panels(img_gray, min_panel_fraction: float = 0.03) -> List[Dict]:
    """
    Detect manga panels by finding bright regions bounded by the
    page's dark gutter/border lines. Returns bounding-box rects in
    image coordinates, sorted in approximate manga reading order.
    """
    h, w = img_gray.shape
    page_area = h * w

    # Dark pixels = inked panel borders. Dilate so any small gaps in
    # the inked frame close up and panels become fully enclosed.
    _, lines = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines = cv2.dilate(lines, line_kernel, iterations=2)

    panel_mask = cv2.bitwise_not(lines)

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(
        panel_mask, connectivity=4
    )

    panels: List[Dict] = []
    min_area = min_panel_fraction * page_area
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        # Drop the "outside the page" background blob, if any.
        if bw > 0.97 * w and bh > 0.97 * h:
            continue
        # Drop ribbons that are obviously gutters, not panels.
        if bw < 0.05 * w or bh < 0.05 * h:
            continue
        panels.append({
            'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh),
            'area': int(area),
        })

    # Top-to-bottom, then right-to-left (typical manga reading order).
    panels.sort(key=lambda p: (p['y'], -p['x']))
    return panels


# ---------------------------------------------------------------------------
# Region coloring presets
# ---------------------------------------------------------------------------

def auto_region_colors(image_shape: Tuple[int, int, int], palette: dict) -> List[Dict]:
    """Single full-page color zone using the palette's primary env color."""
    h, w = image_shape[:2]
    env = palette.get("environment", {})
    if env:
        base_color = list(next(iter(env.values())))
    else:
        base_color = _palette_colors(palette)[0]
    return [{
        'label': 'Auto Page Tint',
        'rect': (0, 0, w, h),
        'color': base_color,
    }]


def auto_region_colors_by_panel(
    img_gray, palette: dict, fallback_full_page: bool = True
) -> List[Dict]:
    """
    Detect manga panels and assign a different palette color to each
    panel. Falls back to a full-page tint if no panels are found.
    """
    panels = detect_panels(img_gray)
    if not panels:
        if fallback_full_page:
            h, w = img_gray.shape
            return auto_region_colors((h, w, 3), palette)
        return []

    colors = _palette_colors(palette)
    regions: List[Dict] = []
    for idx, p in enumerate(panels):
        regions.append({
            'label': f'Panel {idx + 1}',
            'rect': (p['x'], p['y'], p['x'] + p['w'], p['y'] + p['h']),
            'color': colors[idx % len(colors)],
        })
    return regions


# ---------------------------------------------------------------------------
# Geometric overlay templates
# ---------------------------------------------------------------------------

OVERLAY_TEMPLATES = [
    "Corner Accents",
    "Mondrian Frame",
    "Barcode Strip",
    "Edge Ribbons",
    "Panel Corner Tags",
    "Scatter Specks",
    "Diagonal Stripes",
    "Halftone Dots",
    "Manga Speed Lines",
    "Sunburst Corner",
    "Polaroid Frame",
    "Color Vignette",
    "Comic Swash",
    "Sliced Strips H",
    "Sliced Strips V",
    "Cross Split",
    "Shattered Grid",
    "Quadrant Tint",
    "Torn Bands",
    "Tri Slice",
    "Film Strip",
    "Panel Gutter Cuts",
]


def _blank_mask(shape):
    h, w = shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def _corner_accents(shape, colors, intensity) -> List[Dict]:
    h, w = shape[:2]
    op = 0.40 + 0.45 * intensity
    size = 0.04 + 0.10 * intensity
    sw, sh = int(w * size), int(h * size * 0.5)
    return [
        {'rect': (0, 0, sw, sh),
         'color': colors[0 % len(colors)], 'opacity': op},
        {'rect': (w - sw, h - sh, w, h),
         'color': colors[1 % len(colors)], 'opacity': op},
        {'rect': (w - sw, 0, w, sh),
         'color': colors[2 % len(colors)], 'opacity': op * 0.85},
        {'rect': (0, h - sh, sw, h),
         'color': colors[3 % len(colors)], 'opacity': op * 0.85},
    ]


def _mondrian_frame(shape, colors, intensity) -> List[Dict]:
    h, w = shape[:2]
    op = 0.55 + 0.35 * intensity
    t = max(4, int(min(w, h) * (0.006 + 0.010 * intensity)))
    return [
        {'rect': (0, 0, w, t),
         'color': colors[0 % len(colors)], 'opacity': op},
        {'rect': (0, h - t, w, h),
         'color': colors[1 % len(colors)], 'opacity': op},
        {'rect': (0, 0, t, h),
         'color': colors[2 % len(colors)], 'opacity': op},
        {'rect': (w - t, 0, w, h),
         'color': colors[3 % len(colors)], 'opacity': op},
        {'rect': (0, 0, int(w * 0.12), int(h * 0.03)),
         'color': colors[4 % len(colors)], 'opacity': op},
    ]


def _barcode_strip(shape, colors, intensity) -> List[Dict]:
    h, w = shape[:2]
    op = 0.65 + 0.25 * intensity
    strip_h = max(6, int(h * (0.010 + 0.020 * intensity)))
    n_bars = 8 + int(intensity * 8)
    bar_w = max(1, w // n_bars)
    overlays: List[Dict] = []
    for i in range(n_bars):
        x1 = i * bar_w
        x2 = x1 + bar_w
        overlays.append({
            'rect': (x1, 0, x2, strip_h),
            'color': colors[i % len(colors)],
            'opacity': op,
        })
    return overlays


def _edge_ribbons(shape, colors, intensity) -> List[Dict]:
    h, w = shape[:2]
    op = 0.55 + 0.30 * intensity
    t = max(3, int(min(w, h) * (0.004 + 0.008 * intensity)))
    return [
        {'rect': (int(w * 0.55), 0, w, t),
         'color': colors[0 % len(colors)], 'opacity': op},
        {'rect': (0, h - t, int(w * 0.45), h),
         'color': colors[1 % len(colors)], 'opacity': op},
        {'rect': (0, int(h * 0.50), t, h),
         'color': colors[2 % len(colors)], 'opacity': op},
        {'rect': (w - t, 0, w, int(h * 0.40)),
         'color': colors[3 % len(colors)], 'opacity': op},
    ]


def _panel_corner_tags(shape, colors, intensity, panels) -> List[Dict]:
    """Tiny colored tags on each detected panel's top-left corner."""
    if not panels:
        return _corner_accents(shape, colors, intensity)
    op = 0.55 + 0.35 * intensity
    overlays: List[Dict] = []
    for idx, p in enumerate(panels):
        tag_w = max(8, int(p['w'] * (0.06 + 0.06 * intensity)))
        tag_h = max(6, int(p['h'] * (0.02 + 0.04 * intensity)))
        overlays.append({
            'rect': (p['x'], p['y'], p['x'] + tag_w, p['y'] + tag_h),
            'color': colors[idx % len(colors)],
            'opacity': op,
        })
    return overlays


def _scatter_specks(shape, colors, intensity, seed: int = 17) -> List[Dict]:
    """Pseudo-random small color specks scattered near edges only."""
    h, w = shape[:2]
    rng = np.random.default_rng(seed)
    op = 0.60 + 0.30 * intensity
    n = 6 + int(intensity * 10)
    specks: List[Dict] = []
    for i in range(n):
        edge = int(rng.integers(0, 4))
        sx = int(rng.uniform(0.02, 0.08) * w) + 1
        sy = int(rng.uniform(0.012, 0.030) * h) + 1
        if edge == 0:    # top edge
            x = int(rng.uniform(0, 1) * max(1, w - sx))
            y = int(rng.uniform(0, 0.05) * h)
        elif edge == 1:  # bottom edge
            x = int(rng.uniform(0, 1) * max(1, w - sx))
            y = int(rng.uniform(0.95, 1.0) * h) - sy
        elif edge == 2:  # left edge
            x = int(rng.uniform(0, 0.05) * w)
            y = int(rng.uniform(0, 1) * max(1, h - sy))
        else:            # right edge
            x = int(rng.uniform(0.95, 1.0) * w) - sx
            y = int(rng.uniform(0, 1) * max(1, h - sy))
        x = max(0, min(w - sx, x))
        y = max(0, min(h - sy, y))
        specks.append({
            'rect': (x, y, x + sx, y + sy),
            'color': colors[i % len(colors)],
            'opacity': op,
        })
    return specks


def _diagonal_stripes(shape, colors, intensity) -> List[Dict]:
    """Bold angled stripes across top-right and bottom-left corners."""
    h, w = shape[:2]
    op = 0.55 + 0.30 * intensity
    stripe_w = max(8, int(min(w, h) * (0.018 + 0.025 * intensity)))
    gap = stripe_w * 2
    overlays: List[Dict] = []
    # Top-right corner: 3 stripes
    for i in range(3):
        offset = i * gap
        # Stripe runs from top edge to right edge at 45 degrees
        # Defined as a quadrilateral
        cx_start = int(w * 0.65) + offset
        cy_end = int(h * 0.35) + offset
        pts = np.array([
            [cx_start, 0],
            [cx_start + stripe_w, 0],
            [w, cy_end + stripe_w],
            [w, cy_end],
        ], dtype=np.int32)
        mask = _blank_mask(shape)
        cv2.fillPoly(mask, [pts], 255)
        overlays.append({
            'mask': mask, 'color': colors[i % len(colors)], 'opacity': op,
        })
    # Bottom-left corner: 2 stripes
    for i in range(2):
        offset = i * gap
        cx_end = int(w * 0.35) - offset
        cy_start = int(h * 0.65) - offset
        pts = np.array([
            [0, cy_start],
            [0, cy_start + stripe_w],
            [cx_end + stripe_w, h],
            [cx_end, h],
        ], dtype=np.int32)
        mask = _blank_mask(shape)
        cv2.fillPoly(mask, [pts], 255)
        overlays.append({
            'mask': mask, 'color': colors[(i + 3) % len(colors)], 'opacity': op,
        })
    return overlays


def _halftone_dots(shape, colors, intensity) -> List[Dict]:
    """Comic-style dot grid that fades from one corner."""
    h, w = shape[:2]
    op = 0.45 + 0.45 * intensity
    spacing = max(8, int(min(w, h) * (0.022 - 0.010 * intensity)))
    max_radius = max(2, int(spacing * 0.42))
    color = colors[1 % len(colors)]
    mask = _blank_mask(shape)

    # Density fades from top-left to bottom-right
    for y in range(0, h, spacing):
        for x in range(0, w, spacing):
            # Linear falloff based on distance from origin (top-left)
            fade = 1.0 - ((x / w) * 0.7 + (y / h) * 0.7) * 0.9
            if fade <= 0.05:
                continue
            r = max(1, int(max_radius * fade))
            cv2.circle(mask, (x, y), r, 255, -1)
    return [{'mask': mask, 'color': color, 'opacity': op}]


def _manga_speed_lines(shape, colors, intensity) -> List[Dict]:
    """Radial action lines from the right edge (manga-style)."""
    h, w = shape[:2]
    op = 0.55 + 0.30 * intensity
    n_lines = 18 + int(intensity * 24)
    cx, cy = int(w * 1.05), int(h * 0.50)  # vanishing point just off-page
    color = colors[2 % len(colors)]
    mask = _blank_mask(shape)
    thickness = max(2, int(min(w, h) * (0.003 + 0.004 * intensity)))
    radius = int(math.hypot(w, h) * 1.2)

    for i in range(n_lines):
        angle = math.pi + (i / max(1, n_lines - 1)) * math.pi  # 180..360 deg
        x2 = int(cx + radius * math.cos(angle))
        y2 = int(cy + radius * math.sin(angle))
        cv2.line(mask, (cx, cy), (x2, y2), 255, thickness)

    # Fade towards the vanishing point so lines feel directional
    fade = np.linspace(0.0, 1.0, w, dtype=np.float32)
    fade = (fade * 255).astype(np.uint8)
    fade2d = np.broadcast_to(fade[None, :], (h, w))
    mask = np.minimum(mask, fade2d).astype(np.uint8)
    return [{'mask': mask, 'color': color, 'opacity': op}]


def _sunburst_corner(shape, colors, intensity) -> List[Dict]:
    """Radiating triangle wedges from the top-left corner."""
    h, w = shape[:2]
    op = 0.30 + 0.40 * intensity
    n_rays = 8 + int(intensity * 6)
    radius = int(math.hypot(w, h) * (0.55 + 0.30 * intensity))
    cx, cy = 0, 0  # corner origin
    overlays: List[Dict] = []
    for i in range(n_rays):
        # Alternate rays only (every other wedge) so we get the burst look
        if i % 2 == 1:
            continue
        a1 = (i / n_rays) * (math.pi / 2.0)
        a2 = ((i + 1) / n_rays) * (math.pi / 2.0)
        pts = np.array([
            [cx, cy],
            [int(radius * math.cos(a1)), int(radius * math.sin(a1))],
            [int(radius * math.cos(a2)), int(radius * math.sin(a2))],
        ], dtype=np.int32)
        mask = _blank_mask(shape)
        cv2.fillPoly(mask, [pts], 255)
        overlays.append({
            'mask': mask,
            'color': colors[i % len(colors)],
            'opacity': op,
        })
    return overlays


def _polaroid_frame(shape, colors, intensity) -> List[Dict]:
    """Wide solid frame with a thicker bottom title bar (instant-photo look)."""
    h, w = shape[:2]
    op = 0.92
    side = max(8, int(min(w, h) * (0.020 + 0.025 * intensity)))
    bottom = side * (3 + int(intensity * 2))
    frame_color = [248, 244, 232]  # warm off-white
    accent = colors[0 % len(colors)]
    return [
        {'rect': (0, 0, w, side), 'color': frame_color, 'opacity': op},
        {'rect': (0, h - bottom, w, h), 'color': frame_color, 'opacity': op},
        {'rect': (0, 0, side, h), 'color': frame_color, 'opacity': op},
        {'rect': (w - side, 0, w, h), 'color': frame_color, 'opacity': op},
        # Small accent strip near the bottom (palette colour pop)
        {'rect': (int(w * 0.06), h - int(bottom * 0.85),
                  int(w * 0.20), h - int(bottom * 0.70)),
         'color': accent, 'opacity': 0.95},
    ]


def _color_vignette(shape, colors, intensity) -> List[Dict]:
    """Soft radial colour vignette that darkens the edges."""
    h, w = shape[:2]
    op = 0.50 + 0.40 * intensity
    color = colors[0 % len(colors)]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = math.hypot(cx, cy)
    # Vignette ramps in over the outer ~40% of the radius
    ramp = np.clip((r / r_max - 0.55) / 0.45, 0.0, 1.0)
    # Ease for a softer falloff
    ramp = ramp ** 1.6
    mask = (ramp * 255).astype(np.uint8)
    return [{'mask': mask, 'color': color, 'opacity': op}]


def _comic_swash(shape, colors, intensity) -> List[Dict]:
    """Big diagonal translucent swash across the page."""
    h, w = shape[:2]
    op = 0.30 + 0.30 * intensity
    band_w = max(40, int(min(w, h) * (0.20 + 0.15 * intensity)))
    color = colors[1 % len(colors)]

    # Diagonal band from top-right to bottom-left
    pts = np.array([
        [int(w * 0.95) - band_w, 0],
        [int(w * 0.95), 0],
        [int(w * 0.05) + band_w, h],
        [int(w * 0.05), h],
    ], dtype=np.int32)
    mask = _blank_mask(shape)
    cv2.fillPoly(mask, [pts], 255)
    # Soften edges so it reads as a brushed swash
    mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, band_w * 0.08))
    return [{'mask': mask, 'color': color, 'opacity': op}]


# ---------------------------------------------------------------------------
# "Split" / collage style overlays
# ---------------------------------------------------------------------------

def _sliced_strips(shape, colors, intensity, horizontal: bool = True) -> List[Dict]:
    """Draws thick colored gap lines that visually cut the page into strips."""
    h, w = shape[:2]
    op = 0.95
    n_cuts = 3 + int(intensity * 4)          # 3..7 cuts
    gap_thickness = max(4, int(min(w, h) * (0.006 + 0.012 * intensity)))
    overlays: List[Dict] = []
    for i in range(n_cuts):
        # Slightly irregular spacing so it doesn't look like a perfect grid
        frac = (i + 1) / (n_cuts + 1)
        jitter = 0.04 * math.sin(i * 1.7)
        pos = max(gap_thickness, min((h if horizontal else w) - gap_thickness,
                                     int((h if horizontal else w) * (frac + jitter))))
        if horizontal:
            rect = (0, pos - gap_thickness // 2, w, pos + gap_thickness // 2)
        else:
            rect = (pos - gap_thickness // 2, 0, pos + gap_thickness // 2, h)
        overlays.append({
            'rect': rect,
            'color': colors[i % len(colors)],
            'opacity': op,
        })
    return overlays


def _cross_split(shape, colors, intensity) -> List[Dict]:
    """One bold horizontal + one bold vertical band crossing in the middle."""
    h, w = shape[:2]
    op = 0.92
    band_h = max(8, int(h * (0.020 + 0.030 * intensity)))
    band_v = max(8, int(w * (0.020 + 0.030 * intensity)))
    cy = h // 2
    cx = w // 2
    return [
        {'rect': (0, cy - band_h // 2, w, cy + band_h // 2),
         'color': colors[0 % len(colors)], 'opacity': op},
        {'rect': (cx - band_v // 2, 0, cx + band_v // 2, h),
         'color': colors[1 % len(colors)], 'opacity': op},
        # Center cap so the intersection reads cleanly as a different colour
        {'rect': (cx - band_v, cy - band_h, cx + band_v, cy + band_h),
         'color': colors[2 % len(colors)], 'opacity': op},
    ]


def _shattered_grid(shape, colors, intensity) -> List[Dict]:
    """Multi-line grid that splits the image like cracked panels."""
    h, w = shape[:2]
    op = 0.90
    thickness = max(3, int(min(w, h) * (0.004 + 0.008 * intensity)))
    n_h = 2 + int(intensity * 3)             # horizontal cuts
    n_v = 2 + int(intensity * 3)             # vertical cuts
    overlays: List[Dict] = []
    for i in range(n_h):
        frac = (i + 1) / (n_h + 1)
        jitter = 0.05 * math.sin(i * 2.3)
        y = max(thickness, min(h - thickness, int(h * (frac + jitter))))
        overlays.append({
            'rect': (0, y - thickness // 2, w, y + thickness // 2),
            'color': colors[i % len(colors)], 'opacity': op,
        })
    for j in range(n_v):
        frac = (j + 1) / (n_v + 1)
        jitter = 0.05 * math.cos(j * 1.9)
        x = max(thickness, min(w - thickness, int(w * (frac + jitter))))
        overlays.append({
            'rect': (x - thickness // 2, 0, x + thickness // 2, h),
            'color': colors[(j + 2) % len(colors)], 'opacity': op,
        })
    return overlays


def _quadrant_tint(shape, colors, intensity) -> List[Dict]:
    """Tints each of 4 quadrants a different palette colour, plus a cross divider."""
    h, w = shape[:2]
    op_tint = 0.18 + 0.22 * intensity
    op_div = 0.95
    cx, cy = w // 2, h // 2
    div = max(4, int(min(w, h) * (0.005 + 0.010 * intensity)))
    return [
        {'rect': (0, 0, cx, cy),
         'color': colors[0 % len(colors)], 'opacity': op_tint},
        {'rect': (cx, 0, w, cy),
         'color': colors[1 % len(colors)], 'opacity': op_tint},
        {'rect': (0, cy, cx, h),
         'color': colors[2 % len(colors)], 'opacity': op_tint},
        {'rect': (cx, cy, w, h),
         'color': colors[3 % len(colors)], 'opacity': op_tint},
        # Divider cross on top
        {'rect': (0, cy - div // 2, w, cy + div // 2),
         'color': [20, 20, 20], 'opacity': op_div},
        {'rect': (cx - div // 2, 0, cx + div // 2, h),
         'color': [20, 20, 20], 'opacity': op_div},
    ]


def _torn_bands(shape, colors, intensity) -> List[Dict]:
    """Jagged horizontal bands like torn paper strips with coloured gaps."""
    h, w = shape[:2]
    op = 0.95
    n_bands = 3 + int(intensity * 3)
    band_h = max(6, int(h * (0.010 + 0.015 * intensity)))
    overlays: List[Dict] = []
    rng = np.random.default_rng(42)
    for i in range(n_bands):
        cy = int(h * (i + 1) / (n_bands + 1))
        # Build a jagged polygon (top/bottom edges rough)
        n_pts = 24
        xs = np.linspace(0, w, n_pts).astype(np.int32)
        top_ys = cy - band_h // 2 + rng.integers(-band_h, band_h, size=n_pts)
        bot_ys = cy + band_h // 2 + rng.integers(-band_h, band_h, size=n_pts)
        top_pts = list(zip(xs.tolist(), top_ys.tolist()))
        bot_pts = list(zip(xs.tolist()[::-1], bot_ys.tolist()[::-1]))
        pts = np.array(top_pts + bot_pts, dtype=np.int32)
        mask = _blank_mask(shape)
        cv2.fillPoly(mask, [pts], 255)
        overlays.append({
            'mask': mask, 'color': colors[i % len(colors)], 'opacity': op,
        })
    return overlays


def _tri_slice(shape, colors, intensity) -> List[Dict]:
    """Three big diagonal cuts that split the page into wedge sections."""
    h, w = shape[:2]
    op = 0.92
    thickness = max(6, int(min(w, h) * (0.008 + 0.012 * intensity)))
    overlays: List[Dict] = []
    # Three diagonal lines at different angles & offsets
    line_specs = [
        ((0, int(h * 0.20)), (w, int(h * 0.55))),
        ((int(w * 0.15), 0), (int(w * 0.75), h)),
        ((0, int(h * 0.85)), (w, int(h * 0.55))),
    ]
    for i, (p1, p2) in enumerate(line_specs):
        mask = _blank_mask(shape)
        cv2.line(mask, p1, p2, 255, thickness)
        overlays.append({
            'mask': mask, 'color': colors[i % len(colors)], 'opacity': op,
        })
    return overlays


def _film_strip(shape, colors, intensity) -> List[Dict]:
    """Top + bottom black film strips with sprocket holes (palette accents)."""
    h, w = shape[:2]
    strip_h = max(20, int(h * (0.035 + 0.030 * intensity)))
    hole_size = max(6, int(strip_h * 0.45))
    hole_gap = max(hole_size * 2, int(w * 0.04))
    black = [12, 12, 12]
    accent = colors[0 % len(colors)]
    overlays: List[Dict] = [
        {'rect': (0, 0, w, strip_h), 'color': black, 'opacity': 0.97},
        {'rect': (0, h - strip_h, w, h), 'color': black, 'opacity': 0.97},
    ]
    # Sprocket holes (drawn with the page background colour via accent palette)
    y_top = strip_h // 2 - hole_size // 2
    y_bot = h - strip_h // 2 - hole_size // 2
    x = hole_gap // 2
    idx = 0
    while x + hole_size <= w:
        col = accent if idx % 4 == 0 else [240, 240, 240]
        overlays.append({
            'rect': (x, y_top, x + hole_size, y_top + hole_size),
            'color': col, 'opacity': 0.95,
        })
        overlays.append({
            'rect': (x, y_bot, x + hole_size, y_bot + hole_size),
            'color': col, 'opacity': 0.95,
        })
        x += hole_size + hole_gap
        idx += 1
    return overlays


def _panel_gutter_cuts(shape, colors, intensity, panels) -> List[Dict]:
    """Bold coloured gutters around each detected panel — splits page along its own panels."""
    if not panels:
        return _shattered_grid(shape, colors, intensity)
    h, w = shape[:2]
    op = 0.92
    thickness = max(4, int(min(w, h) * (0.005 + 0.010 * intensity)))
    overlays: List[Dict] = []
    for idx, p in enumerate(panels):
        col = colors[idx % len(colors)]
        x1, y1 = p['x'], p['y']
        x2, y2 = p['x'] + p['w'], p['y'] + p['h']
        # Top edge
        overlays.append({
            'rect': (x1, max(0, y1 - thickness // 2),
                     x2, min(h, y1 + thickness // 2)),
            'color': col, 'opacity': op,
        })
        # Bottom edge
        overlays.append({
            'rect': (x1, max(0, y2 - thickness // 2),
                     x2, min(h, y2 + thickness // 2)),
            'color': col, 'opacity': op,
        })
        # Left edge
        overlays.append({
            'rect': (max(0, x1 - thickness // 2), y1,
                     min(w, x1 + thickness // 2), y2),
            'color': col, 'opacity': op,
        })
        # Right edge
        overlays.append({
            'rect': (max(0, x2 - thickness // 2), y1,
                     min(w, x2 + thickness // 2), y2),
            'color': col, 'opacity': op,
        })
    return overlays


def auto_geometric_overlays(
    image_shape,
    palette: dict,
    template: str = "Corner Accents",
    intensity: float = 0.5,
    panels: List[Dict] = None,
) -> List[Dict]:
    """
    Build overlays from a named template + a 0..1 intensity dial.
    `image_shape` can be a numpy shape tuple or an (h, w) tuple.
    """
    colors = _palette_colors(palette)
    intensity = float(max(0.0, min(1.0, intensity)))

    if template == "Mondrian Frame":
        return _mondrian_frame(image_shape, colors, intensity)
    if template == "Barcode Strip":
        return _barcode_strip(image_shape, colors, intensity)
    if template == "Edge Ribbons":
        return _edge_ribbons(image_shape, colors, intensity)
    if template == "Panel Corner Tags":
        return _panel_corner_tags(image_shape, colors, intensity, panels or [])
    if template == "Scatter Specks":
        return _scatter_specks(image_shape, colors, intensity)
    if template == "Diagonal Stripes":
        return _diagonal_stripes(image_shape, colors, intensity)
    if template == "Halftone Dots":
        return _halftone_dots(image_shape, colors, intensity)
    if template == "Manga Speed Lines":
        return _manga_speed_lines(image_shape, colors, intensity)
    if template == "Sunburst Corner":
        return _sunburst_corner(image_shape, colors, intensity)
    if template == "Polaroid Frame":
        return _polaroid_frame(image_shape, colors, intensity)
    if template == "Color Vignette":
        return _color_vignette(image_shape, colors, intensity)
    if template == "Comic Swash":
        return _comic_swash(image_shape, colors, intensity)
    if template == "Sliced Strips H":
        return _sliced_strips(image_shape, colors, intensity, horizontal=True)
    if template == "Sliced Strips V":
        return _sliced_strips(image_shape, colors, intensity, horizontal=False)
    if template == "Cross Split":
        return _cross_split(image_shape, colors, intensity)
    if template == "Shattered Grid":
        return _shattered_grid(image_shape, colors, intensity)
    if template == "Quadrant Tint":
        return _quadrant_tint(image_shape, colors, intensity)
    if template == "Torn Bands":
        return _torn_bands(image_shape, colors, intensity)
    if template == "Tri Slice":
        return _tri_slice(image_shape, colors, intensity)
    if template == "Film Strip":
        return _film_strip(image_shape, colors, intensity)
    if template == "Panel Gutter Cuts":
        return _panel_gutter_cuts(image_shape, colors, intensity, panels or [])
    return _corner_accents(image_shape, colors, intensity)


# ---------------------------------------------------------------------------
# Post-process: invisible hash-breakers
# ---------------------------------------------------------------------------

def apply_hue_shift(img_rgb: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate hue uniformly; near-invisible at small angles, hash-busting."""
    if abs(degrees) < 0.5:
        return img_rgb
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    shift = int(round((degrees / 360.0) * 180))  # OpenCV H is 0..179
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_mirror(img_rgb: np.ndarray) -> np.ndarray:
    """Horizontal flip — strong perceptual-hash breaker."""
    return cv2.flip(img_rgb, 1)


def apply_chromatic_aberration(img_rgb: np.ndarray, shift_px: int,
                               direction: str = 'horizontal') -> np.ndarray:
    """
    Chromatic aberration: shifts the R channel one way and the B channel the
    opposite way, leaving G unchanged.  Creates vivid colour fringing around
    every hard edge — subtle at 2-4 px, dramatic at 8-15 px.  Wrap-around
    pixels at the edges are clamped to the original value to avoid seam artefacts.
    """
    s = max(1, int(shift_px))
    r = img_rgb[:, :, 0].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    result = img_rgb.astype(np.float32).copy()

    if direction == 'vertical':
        result[:, :, 0] = np.roll(r,  s, axis=0)
        result[:, :, 2] = np.roll(b, -s, axis=0)
        result[:s,  :, 0] = r[:s,  :]
        result[-s:, :, 2] = b[-s:, :]
    elif direction == 'diagonal':
        hs = max(1, s // 2)
        result[:, :, 0] = np.roll(np.roll(r,  hs, axis=0),  hs, axis=1)
        result[:, :, 2] = np.roll(np.roll(b, -hs, axis=0), -hs, axis=1)
        result[:hs,  :,   0] = r[:hs,  :  ]
        result[:,    :hs, 0] = r[:,    :hs]
        result[-hs:, :,   2] = b[-hs:, :  ]
        result[:,   -hs:, 2] = b[:,   -hs:]
    else:  # horizontal (default)
        result[:, :, 0] = np.roll(r,  s, axis=1)
        result[:, :, 2] = np.roll(b, -s, axis=1)
        result[:, :s,  0] = r[:, :s ]
        result[:, -s:, 2] = b[:, -s:]

    return np.clip(result, 0, 255).astype(np.uint8)