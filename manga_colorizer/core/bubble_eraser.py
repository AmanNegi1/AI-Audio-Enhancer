import cv2
import numpy as np

# EasyOCR reader is lazily initialised once per process and reused.
_easyocr_reader = None


def _get_reader():
    """Return a cached EasyOCR Reader (Japanese + English, CPU)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # imported here so the module still loads without it
        _easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Text-pixel detector + eraser
# ---------------------------------------------------------------------------
#
# We do NOT try to find the speech bubble outline. We just find pixels
# that look like manga lettering and paint them white. This is exactly
# what the user wants: "just erase the text".
#
# Heuristic:
#   1. Adaptive threshold the page (dark-on-light -> binary).
#   2. Run connected components.
#   3. Keep components that look like characters:
#        - bounded character size
#        - moderate aspect ratio
#        - moderate fill density
#        - the area immediately AROUND the component is mostly bright
#          (so we don't erase art lines that sit on a dark background)
#   4. Group nearby characters into clusters so the UI can show them as
#      "bubbles" you can toggle on/off.
#

def detect_text_clusters(
    img_gray,
    min_char_area=10,
    max_char_area_frac=0.0015,
    bright_surround_thresh=0.55,
    dilate_x=22,
    dilate_y=10,
    min_chars_per_cluster=2,
):
    """
    Find clusters of text pixels on the page.

    Returns a dict:
        {
          'text_mask': uint8 HxW mask of pixels considered text,
          'clusters': [ { 'x','y','w','h','n_chars','mask' (cluster crop) }, ... ]
        }

    `mask` inside each cluster is a small per-cluster text mask the
    eraser uses to paint exactly those pixels white.
    """
    h, w = img_gray.shape
    page_area = float(h * w)

    # Smooth lightly to suppress JPEG noise before thresholding.
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        15, 9,
    )

    n, labels, stats, _cent = cv2.connectedComponentsWithStats(bw, connectivity=8)

    max_char_area = int(page_area * max_char_area_frac)
    text_mask = np.zeros_like(img_gray)

    # Pad for the "bright surround" check
    pad = 4

    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_char_area or area > max_char_area:
            continue
        if cw == 0 or ch == 0:
            continue

        # Reject long thin lines (panel borders / hatching / motion lines)
        aspect = cw / float(ch)
        if aspect > 6 or aspect < 0.16:
            continue
        if max(cw, ch) > min(h, w) * 0.10:
            continue

        density = area / float(cw * ch)
        if density < 0.18 or density > 0.95:
            continue

        # Check the area JUST OUTSIDE the component is mostly bright.
        # This is the key test that keeps art ink out of the text mask.
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad); y1 = min(h, y + ch + pad)
        surround = img_gray[y0:y1, x0:x1]

        # Exclude the component itself from the surround.
        comp_local = (labels[y0:y1, x0:x1] == i)
        outside = surround[~comp_local]
        if outside.size == 0:
            continue

        # Relative check: surround must be brighter than the text component
        # itself, not just brighter than an absolute 200 threshold.
        # This makes the detector work on gray/toned speech bubbles where
        # the background is e.g. 150 rather than 240.
        comp_vals = img_gray[y0:y1, x0:x1][comp_local]
        comp_mean = float(np.mean(comp_vals)) if comp_vals.size > 0 else 128.0
        bright_frac = float(np.mean(outside > comp_mean + 20))
        if bright_frac < bright_surround_thresh:
            continue

        text_mask[labels == i] = 255

    # Group characters into clusters for the UI (so users can deselect a
    # whole bubble in one click). Dilate along reading lines.
    if int(np.count_nonzero(text_mask)) == 0:
        return {'text_mask': text_mask, 'clusters': []}

    # Two kernels: horizontal (left-to-right text) and vertical (top-to-bottom,
    # common in manga). Take the union so both orientations cluster correctly.
    kernel_h = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, dilate_x), max(3, dilate_y)),
    )
    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, dilate_y), max(3, dilate_x)),
    )
    grouped = cv2.bitwise_or(
        cv2.dilate(text_mask, kernel_h, iterations=2),
        cv2.dilate(text_mask, kernel_v, iterations=2),
    )

    n_c, labels_c, stats_c, _cent_c = cv2.connectedComponentsWithStats(
        grouped, connectivity=8,
    )

    clusters = []
    for ci in range(1, n_c):
        cx, cy, cw, ch, _carea = stats_c[ci]
        cluster_label_mask = (labels_c[cy:cy + ch, cx:cx + cw] == ci)
        cluster_text = np.where(
            cluster_label_mask,
            text_mask[cy:cy + ch, cx:cx + cw],
            0,
        ).astype(np.uint8)
        if int(np.count_nonzero(cluster_text)) == 0:
            continue

        # Quick character count inside this cluster (sub-CC pass).
        sub_n, _sl, _ss, _sc = cv2.connectedComponentsWithStats(
            cluster_text, connectivity=8,
        )
        n_chars = max(0, sub_n - 1)
        if n_chars < min_chars_per_cluster:
            continue

        clusters.append({
            'x': int(cx),
            'y': int(cy),
            'w': int(cw),
            'h': int(ch),
            'n_chars': int(n_chars),
            'mask': cluster_text,  # local text mask, paint to white on erase
        })

    # Multi-pass fallback: if nothing was found, retry with looser thresholds.
    # This handles pages with unusual contrast, small fonts, or heavy toning.
    if not clusters and bright_surround_thresh > 0.25:
        return detect_text_clusters(
            img_gray,
            min_char_area=max(6, min_char_area - 2),
            max_char_area_frac=max_char_area_frac * 1.5,
            bright_surround_thresh=max(0.25, bright_surround_thresh - 0.2),
            dilate_x=dilate_x,
            dilate_y=dilate_y,
            min_chars_per_cluster=max(1, min_chars_per_cluster - 1),
        )

    return {'text_mask': text_mask, 'clusters': clusters}


# ---------------------------------------------------------------------------
# AI-based text detection (EasyOCR / CRAFT)
# ---------------------------------------------------------------------------

def detect_text_clusters_ai(img_gray):
    """
    Use EasyOCR's CRAFT neural network to detect text regions.
    Returns the same dict shape as detect_text_clusters() so both are
    interchangeable:
        {'text_mask': uint8 HxW, 'clusters': [{'x','y','w','h','n_chars','mask'}]}
    """
    h, w = img_gray.shape
    reader = _get_reader()

    # reader.detect() is detection-only (no OCR) — much faster than readtext.
    # It returns (horizontal_list, free_list), each a list with one entry per
    # input image.  We pass a single image so index [0] is always the result.
    horizontal_list, free_list = reader.detect(
        img_gray,
        slope_ths=0.15,    # tolerate slight tilts in speech bubbles
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.05,   # small margin around each detected word box
        min_size=8,
    )
    boxes_h = horizontal_list[0] if horizontal_list else []
    boxes_f = free_list[0] if free_list else []

    text_mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes_h:
        # box = [x_min, x_max, y_min, y_max]
        x1, x2, y1, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            text_mask[y1:y2, x1:x2] = 255

    for quad in boxes_f:
        # quad = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        pts = np.array([[int(p[0]), int(p[1])] for p in quad], dtype=np.int32)
        cv2.fillPoly(text_mask, [pts], 255)

    if int(np.count_nonzero(text_mask)) == 0:
        return {'text_mask': text_mask, 'clusters': []}

    # Group nearby word boxes into per-bubble clusters (same as CV path).
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 10))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 22))
    grouped = cv2.bitwise_or(
        cv2.dilate(text_mask, kernel_h, iterations=2),
        cv2.dilate(text_mask, kernel_v, iterations=2),
    )

    n_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(
        grouped, connectivity=8,
    )

    clusters = []
    for ci in range(1, n_c):
        cx, cy, cw, ch, _ = stats_c[ci]
        cluster_label_mask = (labels_c[cy:cy + ch, cx:cx + cw] == ci)
        cluster_text = np.where(
            cluster_label_mask,
            text_mask[cy:cy + ch, cx:cx + cw],
            0,
        ).astype(np.uint8)
        if int(np.count_nonzero(cluster_text)) == 0:
            continue
        sub_n, _, _, _ = cv2.connectedComponentsWithStats(
            cluster_text, connectivity=8,
        )
        clusters.append({
            'x': int(cx),
            'y': int(cy),
            'w': int(cw),
            'h': int(ch),
            'n_chars': max(0, sub_n - 1),
            'mask': cluster_text,
        })

    return {'text_mask': text_mask, 'clusters': clusters}


# ---------------------------------------------------------------------------
# Backwards-compatible bubble-detection API used by app.py
# ---------------------------------------------------------------------------
#
# The app code still calls `detect_speech_bubbles(...)` and treats the
# return values as "bubbles" to be erased. We now return text clusters
# under the same shape, plus 'mask' / 'area' so the rest of the code
# keeps working unchanged.

def detect_speech_bubbles(
    img_gray,
    min_area=400,             # kept for slider compatibility (ignored)
    max_area=200000,          # kept for slider compatibility (ignored)
    solidity_thresh=0.55,     # kept for slider compatibility (ignored)
    white_thresh=225,         # kept for slider compatibility (ignored)
    min_text_components=2,
    max_text_components=300,  # kept for slider compatibility (ignored)
):
    """
    Neural (EasyOCR CRAFT) text detector with CV heuristic fallback.
    Each returned 'bubble' is a text cluster; erasing it paints only the
    detected text pixels white (with a small margin).
    """
    # --- Primary path: EasyOCR neural detector ---
    clusters = []
    try:
        result = detect_text_clusters_ai(img_gray)
        clusters = result['clusters']
    except Exception:
        pass  # fall through to CV heuristics

    # --- Fallback: CV heuristics (no EasyOCR or empty result) ---
    if not clusters:
        result = detect_text_clusters(
            img_gray,
            min_chars_per_cluster=max(1, int(min_text_components)),
        )
        clusters = result['clusters']

    bubbles = []
    for c in clusters:
        area = int(np.count_nonzero(c['mask']))
        bubbles.append({
            'x': c['x'],
            'y': c['y'],
            'w': c['w'],
            'h': c['h'],
            'area': float(area),
            'n_chars': c['n_chars'],
            'mask': c['mask'],  # local text-only mask
        })
    return bubbles


# ---------------------------------------------------------------------------
# Eraser: paint only the text pixels (plus a small margin)
# ---------------------------------------------------------------------------

def erase_regions(img_rgb, regions, fill_color=(255, 255, 255), margin_px=2, inpaint=False):
    """
    Paint detected text pixels with `fill_color`.

    - If the region carries a per-cluster 'mask' (from the new detector),
      we dilate it by `margin_px` and paint exactly those pixels. This
      preserves the bubble outline and surrounding artwork.
        - If a region is a manual bubble, we can fill an ellipse, rectangle,
            or both for cleaner bubble coverage.
    - When `inpaint=True`, uses cv2.INPAINT_TELEA to reconstruct the
      background naturally. This produces much better results on gray,
      toned, or gradient-filled speech bubbles.
    """
    out = img_rgb.copy()
    if not regions:
        return out

    h, w = out.shape[:2]
    fill = np.array(fill_color, dtype=out.dtype)
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1),
    )

    if inpaint:
        # Build a single combined mask from all regions, then inpaint once.
        combined = np.zeros((h, w), dtype=np.uint8)
        for r in regions:
            local_mask = r.get('mask') if isinstance(r, dict) else None
            if local_mask is not None:
                rx = max(0, int(r['x'])); ry = max(0, int(r['y']))
                mh, mw = local_mask.shape[:2]
                rx2 = min(w, rx + mw); ry2 = min(h, ry + mh)
                if rx2 > rx and ry2 > ry:
                    combined[ry:ry2, rx:rx2] = np.maximum(
                        combined[ry:ry2, rx:rx2],
                        local_mask[:ry2 - ry, :rx2 - rx],
                    )
            else:
                rx = max(0, int(r['x'])); ry = max(0, int(r['y']))
                rx2 = min(w, rx + int(r['w'])); ry2 = min(h, ry + int(r['h']))
                if rx2 > rx and ry2 > ry:
                    combined[ry:ry2, rx:rx2] = 255
        if margin_px > 0:
            combined = cv2.dilate(combined, margin_kernel, iterations=1)
        out = cv2.inpaint(out, combined, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return out

    for r in regions:
        local_mask = r.get('mask') if isinstance(r, dict) else None
        if local_mask is not None:
            x = int(r['x']); y = int(r['y'])
            mh, mw = local_mask.shape[:2]
            x2 = min(w, x + mw); y2 = min(h, y + mh)
            x = max(0, x); y = max(0, y)
            if x2 <= x or y2 <= y:
                continue
            local = local_mask[: y2 - y, : x2 - x]
            if margin_px > 0:
                local = cv2.dilate(local, margin_kernel, iterations=1)
            out[y:y2, x:x2][local > 0] = fill
            continue

        # Manual shape fallback. Legacy manual regions have no shape and
        # are treated as rectangles.
        x = int(r['x']); y = int(r['y'])
        bw = int(r['w']); bh = int(r['h'])
        x2 = min(w, x + bw); y2 = min(h, y + bh)
        x = max(0, x); y = max(0, y)
        if x2 <= x or y2 <= y:
            continue

        shape = str(r.get('shape', 'rectangle')).lower() if isinstance(r, dict) else 'rectangle'
        fill_tuple = tuple(int(c) for c in fill_color)

        if shape in ('ellipse', 'bubble', 'double_pass'):
            center = ((x + x2) // 2, (y + y2) // 2)
            axes = (max(1, (x2 - x) // 2), max(1, (y2 - y) // 2))
            cv2.ellipse(out, center, axes, 0, 0, 360, fill_tuple, -1)

        if shape in ('rectangle', 'caption', 'double_pass'):
            inset = int(r.get('inset_px', 0)) if isinstance(r, dict) else 0
            rx1 = min(x2, max(x, x + inset))
            ry1 = min(y2, max(y, y + inset))
            rx2 = max(rx1, min(x2, x2 - inset))
            ry2 = max(ry1, min(y2, y2 - inset))
            if rx2 > rx1 and ry2 > ry1:
                out[ry1:ry2, rx1:rx2] = fill
        elif shape not in ('ellipse', 'bubble'):
            out[y:y2, x:x2] = fill

    return out