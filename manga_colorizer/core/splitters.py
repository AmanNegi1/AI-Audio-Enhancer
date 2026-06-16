"""
Image splitter transforms: physically slice the rendered page into
pieces and shift / rearrange them so the artwork itself looks split
(not just an overlay drawn on top).

Each function takes a uint8 RGB image and returns a uint8 RGB image
of the same shape. Gaps left by the transform are filled with `gap_color`.
"""

from typing import List, Dict, Tuple
import numpy as np
import cv2


SPLITTER_TEMPLATES = [
    "None",
    "Panel Pop-out",
    "Horizontal Slice Shift",
    "Vertical Slice Shift",
    "Shuffled Strips",
    "Grid Tile Shift",
    "Diagonal Split",
    "Glitch Slice",
    "Mirror Half",
    "Center Burst",
    "Inset Strips H",
    "Inset Strips V",
    "Inset Grid",
    "Cross Split Inset",
]


def _fill(shape: Tuple[int, int, int], color) -> np.ndarray:
    canvas = np.empty(shape, dtype=np.uint8)
    canvas[:] = np.array(color, dtype=np.uint8)
    return canvas


# ---------------------------------------------------------------------------
# Individual splitters
# ---------------------------------------------------------------------------

def _panel_popout(img: np.ndarray, intensity: float, panels: List[Dict],
                  gap_color) -> np.ndarray:
    """Each detected panel is rescaled slightly smaller and placed back into
    its bounding box, leaving a gap around it so panels look separated."""
    if not panels:
        return img.copy()
    h, w = img.shape[:2]
    out = _fill(img.shape, gap_color)
    pad_frac = 0.02 + 0.08 * intensity  # 2..10 % padding around each panel
    for p in panels:
        x, y, pw, ph = p['x'], p['y'], p['w'], p['h']
        if pw <= 4 or ph <= 4:
            continue
        pad_x = max(2, int(pw * pad_frac))
        pad_y = max(2, int(ph * pad_frac))
        new_w = max(1, pw - 2 * pad_x)
        new_h = max(1, ph - 2 * pad_y)
        panel_img = img[y:y + ph, x:x + pw]
        resized = cv2.resize(panel_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out[y + pad_y:y + pad_y + new_h, x + pad_x:x + pad_x + new_w] = resized
    return out


def _horizontal_slice_shift(img, intensity, gap_color) -> np.ndarray:
    h, w = img.shape[:2]
    n = 4 + int(intensity * 6)               # 4..10 strips
    strip_h = h // n
    max_shift = int(w * (0.04 + 0.12 * intensity))
    out = _fill(img.shape, gap_color)
    rng = np.random.default_rng(7)
    for i in range(n):
        y0 = i * strip_h
        y1 = h if i == n - 1 else (i + 1) * strip_h
        # Alternate direction and randomise magnitude
        sign = 1 if i % 2 == 0 else -1
        shift = sign * int(rng.integers(max_shift // 2, max_shift + 1))
        strip = img[y0:y1]
        if shift >= 0:
            out[y0:y1, shift:w] = strip[:, :w - shift]
        else:
            out[y0:y1, 0:w + shift] = strip[:, -shift:]
    return out


def _vertical_slice_shift(img, intensity, gap_color) -> np.ndarray:
    h, w = img.shape[:2]
    n = 4 + int(intensity * 6)
    strip_w = w // n
    max_shift = int(h * (0.04 + 0.12 * intensity))
    out = _fill(img.shape, gap_color)
    rng = np.random.default_rng(11)
    for i in range(n):
        x0 = i * strip_w
        x1 = w if i == n - 1 else (i + 1) * strip_w
        sign = 1 if i % 2 == 0 else -1
        shift = sign * int(rng.integers(max_shift // 2, max_shift + 1))
        strip = img[:, x0:x1]
        if shift >= 0:
            out[shift:h, x0:x1] = strip[:h - shift, :]
        else:
            out[0:h + shift, x0:x1] = strip[-shift:, :]
    return out


def _shuffled_strips(img, intensity, gap_color) -> np.ndarray:
    h, w = img.shape[:2]
    n = 4 + int(intensity * 4)  # 4..8 strips
    strip_h = h // n
    rng = np.random.default_rng(23)
    order = list(range(n))
    rng.shuffle(order)
    out = _fill(img.shape, gap_color)
    gap = max(2, int(strip_h * 0.05))
    for new_i, orig_i in enumerate(order):
        sy = orig_i * strip_h
        ey = h if orig_i == n - 1 else sy + strip_h
        src = img[sy:ey]
        dy = new_i * strip_h
        dy_end = min(h, dy + (ey - sy) - gap)
        h_dst = dy_end - dy
        if h_dst <= 0:
            continue
        out[dy:dy_end] = src[:h_dst]
    return out


def _grid_tile_shift(img, intensity, gap_color) -> np.ndarray:
    h, w = img.shape[:2]
    cols = 3 + int(intensity * 3)            # 3..6
    rows = 3 + int(intensity * 3)
    tile_w = w // cols
    tile_h = h // rows
    shift_x = int(tile_w * (0.06 + 0.10 * intensity))
    shift_y = int(tile_h * (0.06 + 0.10 * intensity))
    out = _fill(img.shape, gap_color)
    for r in range(rows):
        for c in range(cols):
            x0 = c * tile_w
            y0 = r * tile_h
            x1 = w if c == cols - 1 else x0 + tile_w
            y1 = h if r == rows - 1 else y0 + tile_h
            tile = img[y0:y1, x0:x1]
            # Alternate offset direction per tile in a checker pattern
            dx = shift_x if (r + c) % 2 == 0 else -shift_x
            dy = shift_y if (r + c) % 2 == 0 else -shift_y
            nx0 = max(0, min(w, x0 + dx))
            ny0 = max(0, min(h, y0 + dy))
            nx1 = min(w, nx0 + (x1 - x0))
            ny1 = min(h, ny0 + (y1 - y0))
            tw = nx1 - nx0
            th = ny1 - ny0
            if tw > 0 and th > 0:
                out[ny0:ny1, nx0:nx1] = tile[:th, :tw]
    return out


def _diagonal_split(img, intensity, gap_color) -> np.ndarray:
    """Cut along the main diagonal and push each triangle outward."""
    h, w = img.shape[:2]
    out = _fill(img.shape, gap_color)
    shift = int(min(w, h) * (0.03 + 0.07 * intensity))

    # Upper-left triangle mask
    yy, xx = np.mgrid[0:h, 0:w]
    upper_mask = (yy * w + xx * h) < (w * h)
    lower_mask = ~upper_mask

    # Shift the upper triangle up-left, lower triangle down-right
    upper_src = img.copy()
    upper_src[~upper_mask] = 0
    lower_src = img.copy()
    lower_src[~lower_mask] = 0

    M_up = np.float32([[1, 0, -shift], [0, 1, -shift]])
    M_dn = np.float32([[1, 0, shift], [0, 1, shift]])
    up_shifted = cv2.warpAffine(upper_src, M_up, (w, h),
                                borderValue=gap_color)
    dn_shifted = cv2.warpAffine(lower_src, M_dn, (w, h),
                                borderValue=gap_color)

    # Recompute masks after shift to know which pixels to keep from each
    up_mask_shift = cv2.warpAffine(upper_mask.astype(np.uint8) * 255, M_up,
                                   (w, h)) > 0
    dn_mask_shift = cv2.warpAffine(lower_mask.astype(np.uint8) * 255, M_dn,
                                   (w, h)) > 0
    out[up_mask_shift] = up_shifted[up_mask_shift]
    out[dn_mask_shift] = dn_shifted[dn_mask_shift]
    return out


def _glitch_slice(img, intensity, gap_color) -> np.ndarray:
    """Many thin strips shifted randomly + RGB-channel offset for chromatic
    aberration. Pure glitch / cyberpunk look."""
    h, w = img.shape[:2]
    n = 10 + int(intensity * 30)
    strip_h = max(2, h // n)
    rng = np.random.default_rng(31)
    out = img.copy()
    max_shift = int(w * (0.03 + 0.10 * intensity))
    for i in range(0, h, strip_h):
        # 50% chance of shifting any given strip
        if rng.random() < 0.5:
            continue
        end = min(h, i + strip_h)
        shift = int(rng.integers(-max_shift, max_shift + 1))
        out[i:end] = np.roll(img[i:end], shift, axis=1)

    # Chromatic aberration: shift red & blue channels horizontally
    ca = max(1, int(w * 0.004 * (1 + intensity)))
    r = np.roll(out[..., 0], ca, axis=1)
    b = np.roll(out[..., 2], -ca, axis=1)
    out = np.dstack([r, out[..., 1], b]).astype(np.uint8)
    return out


def _mirror_half(img, intensity, gap_color) -> np.ndarray:
    """Left half stays, right half is mirrored from the left (symmetry effect),
    with a coloured center divider whose width grows with intensity."""
    h, w = img.shape[:2]
    out = img.copy()
    mid = w // 2
    out[:, mid:] = cv2.flip(img[:, :w - mid], 1)
    div = max(2, int(w * (0.004 + 0.008 * intensity)))
    out[:, mid - div // 2:mid + div // 2] = np.array(gap_color, dtype=np.uint8)
    return out


def _center_burst(img, intensity, gap_color) -> np.ndarray:
    """Splits the image into 4 quadrants and pushes them outward from center."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    shift = int(min(w, h) * (0.03 + 0.08 * intensity))
    out = _fill(img.shape, gap_color)

    # Top-left quadrant -> shift up-left
    tl = img[0:cy, 0:cx]
    nx, ny = max(0, -shift), max(0, -shift)
    # Actually shift away from center -> top-left moves up-left
    out[max(0, -shift):cy - shift if cy - shift > 0 else 0,
        max(0, -shift):cx - shift if cx - shift > 0 else 0] = tl[
        : cy - max(0, shift) - max(0, -shift),
        : cx - max(0, shift) - max(0, -shift),
    ]
    # Simpler: use warpAffine on each quadrant copied into its own canvas
    out = _fill(img.shape, gap_color)

    def _shift_region(src_region, dy, dx, dst_y, dst_x):
        rh, rw = src_region.shape[:2]
        y0 = max(0, dst_y + dy)
        x0 = max(0, dst_x + dx)
        y1 = min(h, y0 + rh)
        x1 = min(w, x0 + rw)
        sy = y0 - (dst_y + dy)
        sx = x0 - (dst_x + dx)
        out[y0:y1, x0:x1] = src_region[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]

    _shift_region(img[0:cy, 0:cx], -shift, -shift, 0, 0)               # TL
    _shift_region(img[0:cy, cx:w], -shift,  shift, 0, cx)              # TR
    _shift_region(img[cy:h, 0:cx],  shift, -shift, cy, 0)              # BL
    _shift_region(img[cy:h, cx:w],  shift,  shift, cy, cx)             # BR
    return out


# ---------------------------------------------------------------------------
# Inset splitters: cut the image, show gaps, but keep ALL pieces inside the frame
# ---------------------------------------------------------------------------

def _inset_strips(img, intensity, gap_color, horizontal: bool = True) -> np.ndarray:
    """Cut into N equal strips. Compress each strip slightly along the cut axis
    so coloured gaps fit between them — full image is still visible, nothing
    leaves the frame."""
    h, w = img.shape[:2]
    n = 3 + int(intensity * 5)               # 3..8 strips
    gap = max(3, int((h if horizontal else w) * (0.010 + 0.025 * intensity)))
    out = _fill(img.shape, gap_color)

    total_gap = gap * (n - 1)
    axis_len = h if horizontal else w
    avail = max(n, axis_len - total_gap)
    base = avail // n
    extra = avail - base * n  # distribute remainder over the first `extra` strips

    if horizontal:
        src_strip = h // n
        dst_y = 0
        for i in range(n):
            sy = i * src_strip
            ey = h if i == n - 1 else sy + src_strip
            strip = img[sy:ey]
            new_h = base + (1 if i < extra else 0)
            resized = cv2.resize(strip, (w, new_h), interpolation=cv2.INTER_AREA)
            out[dst_y:dst_y + new_h] = resized
            dst_y += new_h + gap
    else:
        src_strip = w // n
        dst_x = 0
        for i in range(n):
            sx = i * src_strip
            ex = w if i == n - 1 else sx + src_strip
            strip = img[:, sx:ex]
            new_w = base + (1 if i < extra else 0)
            resized = cv2.resize(strip, (new_w, h), interpolation=cv2.INTER_AREA)
            out[:, dst_x:dst_x + new_w] = resized
            dst_x += new_w + gap
    return out


def _inset_grid(img, intensity, gap_color) -> np.ndarray:
    """Cut into a grid of cols×rows tiles and compress them so gaps appear
    between every tile, but the whole grid still fills the frame."""
    h, w = img.shape[:2]
    cols = 2 + int(intensity * 3)            # 2..5
    rows = 2 + int(intensity * 3)
    gap_x = max(3, int(w * (0.008 + 0.020 * intensity)))
    gap_y = max(3, int(h * (0.008 + 0.020 * intensity)))
    out = _fill(img.shape, gap_color)

    total_gap_w = gap_x * (cols - 1)
    total_gap_h = gap_y * (rows - 1)
    avail_w = max(cols, w - total_gap_w)
    avail_h = max(rows, h - total_gap_h)
    base_w, extra_w = avail_w // cols, avail_w - (avail_w // cols) * cols
    base_h, extra_h = avail_h // rows, avail_h - (avail_h // rows) * rows

    src_tw = w // cols
    src_th = h // rows
    dst_y = 0
    for r in range(rows):
        cell_h = base_h + (1 if r < extra_h else 0)
        sy = r * src_th
        ey = h if r == rows - 1 else sy + src_th
        dst_x = 0
        for c in range(cols):
            cell_w = base_w + (1 if c < extra_w else 0)
            sx = c * src_tw
            ex = w if c == cols - 1 else sx + src_tw
            tile = img[sy:ey, sx:ex]
            resized = cv2.resize(tile, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            out[dst_y:dst_y + cell_h, dst_x:dst_x + cell_w] = resized
            dst_x += cell_w + gap_x
        dst_y += cell_h + gap_y
    return out


def _cross_split_inset(img, intensity, gap_color) -> np.ndarray:
    """Splits the image into 4 quadrants and compresses them so a coloured
    cross-shaped gap appears in the middle — full image stays inside the frame."""
    h, w = img.shape[:2]
    gap_x = max(4, int(w * (0.012 + 0.028 * intensity)))
    gap_y = max(4, int(h * (0.012 + 0.028 * intensity)))
    new_w_left = (w - gap_x) // 2
    new_w_right = w - gap_x - new_w_left
    new_h_top = (h - gap_y) // 2
    new_h_bot = h - gap_y - new_h_top

    # Source quadrants (split at original midpoints)
    midx, midy = w // 2, h // 2
    tl = cv2.resize(img[0:midy, 0:midx], (new_w_left, new_h_top), interpolation=cv2.INTER_AREA)
    tr = cv2.resize(img[0:midy, midx:w], (new_w_right, new_h_top), interpolation=cv2.INTER_AREA)
    bl = cv2.resize(img[midy:h, 0:midx], (new_w_left, new_h_bot), interpolation=cv2.INTER_AREA)
    br = cv2.resize(img[midy:h, midx:w], (new_w_right, new_h_bot), interpolation=cv2.INTER_AREA)

    out = _fill(img.shape, gap_color)
    out[0:new_h_top, 0:new_w_left] = tl
    out[0:new_h_top, new_w_left + gap_x:new_w_left + gap_x + new_w_right] = tr
    out[new_h_top + gap_y:new_h_top + gap_y + new_h_bot, 0:new_w_left] = bl
    out[new_h_top + gap_y:new_h_top + gap_y + new_h_bot,
        new_w_left + gap_x:new_w_left + gap_x + new_w_right] = br
    return out


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def apply_splitter(img: np.ndarray,
                   template: str,
                   intensity: float = 0.5,
                   panels: List[Dict] = None,
                   gap_color=(255, 255, 255)) -> np.ndarray:
    """
    Apply a split/transform template to a finished RGB image.
    Returns img unchanged for template == 'None' or unknown templates.
    """
    if template is None or template == "None":
        return img
    intensity = float(max(0.0, min(1.0, intensity)))
    gap_color = tuple(int(c) for c in gap_color)
    if template == "Panel Pop-out":
        return _panel_popout(img, intensity, panels or [], gap_color)
    if template == "Horizontal Slice Shift":
        return _horizontal_slice_shift(img, intensity, gap_color)
    if template == "Vertical Slice Shift":
        return _vertical_slice_shift(img, intensity, gap_color)
    if template == "Shuffled Strips":
        return _shuffled_strips(img, intensity, gap_color)
    if template == "Grid Tile Shift":
        return _grid_tile_shift(img, intensity, gap_color)
    if template == "Diagonal Split":
        return _diagonal_split(img, intensity, gap_color)
    if template == "Glitch Slice":
        return _glitch_slice(img, intensity, gap_color)
    if template == "Mirror Half":
        return _mirror_half(img, intensity, gap_color)
    if template == "Center Burst":
        return _center_burst(img, intensity, gap_color)
    if template == "Inset Strips H":
        return _inset_strips(img, intensity, gap_color, horizontal=True)
    if template == "Inset Strips V":
        return _inset_strips(img, intensity, gap_color, horizontal=False)
    if template == "Inset Grid":
        return _inset_grid(img, intensity, gap_color)
    if template == "Cross Split Inset":
        return _cross_split_inset(img, intensity, gap_color)
    return img