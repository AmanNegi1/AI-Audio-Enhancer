import cv2
import numpy as np

def blend_layers(original_rgb, color_layers, geometric_shapes=None, brightness=1.0, contrast=1.0, blend_mode="Multiply"):
    """
    Blends the original black-and-white lines with color canvas layers
    and optional geometric block overlays.
    
    original_rgb: np.ndarray, shape (H, W, 3), range 0-255 (uint8)
    color_layers: list of np.ndarray, shape (H, W, 3), range 0-255 (float32)
    geometric_shapes: list of dicts. Each dict has 'color' (RGB) and
        'opacity' (0..1), plus EITHER:
            - 'rect': (x1, y1, x2, y2) for a filled rectangle, OR
            - 'mask': HxW uint8 mask (0..255) used as soft alpha for an
              arbitrary shape (stripes, dots, vignettes, etc.).
    """
    h, w, _ = original_rgb.shape
    
    # 1. Base color canvas: start with white, blend all user regions
    color_canvas = np.ones((h, w, 3), dtype=np.float32) * 255.0
    for layer in color_layers:
        # Wherever the layer is not plain white, blend it/apply it
        mask = np.any(layer < 254.0, axis=2)
        color_canvas[mask] = layer[mask]
        
    # Convert to 0-1 float range
    orig_float = original_rgb.astype(np.float32) / 255.0
    canvas_float = color_canvas.astype(np.float32) / 255.0
    
    # 2. Add geometric block overlays (Bauhaus/Mondrian/TikTok bypass style)
    if geometric_shapes:
        for shape in geometric_shapes:
            geom_color = np.array(shape['color'], dtype=np.float32) / 255.0
            opacity = float(shape.get('opacity', 0.85))

            mask = shape.get('mask')
            if mask is not None:
                # Soft-alpha composite using mask as 0..1 weight * opacity
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                alpha = (mask.astype(np.float32) / 255.0) * opacity
                alpha3 = alpha[..., None]
                canvas_float = (1.0 - alpha3) * canvas_float + alpha3 * geom_color
                continue

            # Rectangle fallback (existing behaviour)
            x1, y1, x2, y2 = shape['rect']
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            canvas_float[y1:y2, x1:x2] = (
                (1.0 - opacity) * canvas_float[y1:y2, x1:x2]
                + opacity * geom_color
            )
            
    # 3. Blend original lines with the colour canvas
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Adaptive ink/paper thresholds from the page's own histogram so the
    # blender works correctly across light, dark, and toned pages.
    gray_u8 = (gray * 255).astype(np.uint8)
    ink_thresh   = max(0.10, min(0.35, float(np.percentile(gray_u8, 12)) / 255.0 + 0.12))
    paper_thresh = max(0.72, min(0.96, float(np.percentile(gray_u8, 88)) / 255.0 - 0.03))

    ink_mask   = gray < ink_thresh
    paper_mask = gray > paper_thresh
    mid_mask   = ~(paper_mask | ink_mask)

    blended = np.zeros_like(canvas_float)

    if blend_mode == "Replace":
        # Use colour canvas directly — ideal for Neon Glow (dark-background layers).
        blended = canvas_float.copy()
        blended[ink_mask] = [0.03, 0.03, 0.04]
    else:
        blended[paper_mask] = canvas_float[paper_mask]
        blended[ink_mask]   = [0.03, 0.03, 0.04]

        if blend_mode == "Overlay":
            result = np.where(
                canvas_float <= 0.5,
                2.0 * orig_float * canvas_float,
                1.0 - 2.0 * (1.0 - orig_float) * (1.0 - canvas_float),
            )
            blended[mid_mask] = np.clip(result, 0, 1)[mid_mask]

        elif blend_mode == "Screen":
            result = 1.0 - (1.0 - orig_float) * (1.0 - canvas_float)
            blended[mid_mask] = np.clip(result, 0, 1)[mid_mask]

        elif blend_mode == "Hard Light":
            result = np.where(
                orig_float <= 0.5,
                2.0 * orig_float * canvas_float,
                1.0 - 2.0 * (1.0 - orig_float) * (1.0 - canvas_float),
            )
            blended[mid_mask] = np.clip(result, 0, 1)[mid_mask]

        else:  # Multiply (default)
            blended[mid_mask] = np.clip(orig_float * canvas_float * 1.25, 0, 1)[mid_mask]

    # Clamp and convert to uint8
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    # 4. Apply brightness and contrast adjustments
    if brightness != 1.0 or contrast != 1.0:
        blended = cv2.convertScaleAbs(blended, alpha=contrast, beta=int((brightness - 1.0) * 100))

    # 5. Subtle sharpening to restore line crispness
    kernel = np.array([[0, -0.3, 0],
                       [-0.3, 2.2, -0.3],
                       [0, -0.3, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(blended, -1, kernel)

    return np.clip(sharpened, 0, 255).astype(np.uint8)