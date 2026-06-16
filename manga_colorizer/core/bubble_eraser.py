import cv2
import numpy as np


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
    max_char_area_frac=0.0009,
    bright_surround_thresh=0.72,
    bright_pixel_thresh=215,
    dilate_x=22,
    dilate_y=10,
    min_chars_per_cluster=3,
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

    max_char_area = max(500, int(page_area * max_char_area_frac))
    text_mask = np.zeros_like(img_gray)

    # Pad for the "bright surround" check
    pad = 8

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
        if max(cw, ch) > max(48, min(h, w) * 0.06):
            continue

        density = area / float(cw * ch)
        if density < 0.12 or density > 0.72:
            continue

        # Check the area JUST OUTSIDE the component is mostly bright.
        # This is the key test that keeps art ink out of the text mask.
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad); y1 = min(h, y + ch + pad)
        surround = img_gray[y0:y1, x0:x1]

        # Exclude nearby foreground pixels so adjacent letters do not make
        # the surround look like artwork.
        foreground_local = (bw[y0:y1, x0:x1] > 0)
        outside = surround[~foreground_local]
        if outside.size == 0:
            continue

        bright_frac = float(np.mean(outside > bright_pixel_thresh))
        if bright_frac < bright_surround_thresh:
            continue

        text_mask[labels == i] = 255

    # Group characters into clusters for the UI (so users can deselect a
    # whole bubble in one click). Dilate along reading lines.
    if int(np.count_nonzero(text_mask)) == 0:
        return {'text_mask': text_mask, 'clusters': []}

    cluster_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, dilate_x), max(3, dilate_y)),
    )
    grouped = cv2.dilate(text_mask, cluster_kernel, iterations=2)

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
    Text-pixel detector wrapped in the app's bubble API.
    Each returned 'bubble' is actually a text cluster; erasing it paints
    only the text pixels white (with a tiny margin).
    """
    result = detect_text_clusters(
        img_gray,
        bright_pixel_thresh=max(180, min(245, int(white_thresh) - 10)),
        min_chars_per_cluster=max(3, int(min_text_components)),
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

def erase_regions(img_rgb, regions, fill_color=(255, 255, 255), margin_px=2):
    """
    Paint detected text pixels with `fill_color`.

    - If the region carries a per-cluster 'mask' (from the new detector),
      we dilate it by `margin_px` and paint exactly those pixels. This
      preserves the bubble outline and surrounding artwork.
    - If a region is just a rectangle (manual erase), we fill the rect.
    """
    out = img_rgb.copy()
    if not regions:
        return out

    h, w = out.shape[:2]
    fill = np.array(fill_color, dtype=out.dtype)
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1),
    )

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

        # Manual rectangle fallback
        x = int(r['x']); y = int(r['y'])
        bw = int(r['w']); bh = int(r['h'])
        x2 = min(w, x + bw); y2 = min(h, y + bh)
        x = max(0, x); y = max(0, y)
        if x2 > x and y2 > y:
            out[y:y2, x:x2] = fill

    return out