import cv2
import numpy as np

def blend_layers(original_rgb, color_layers, geometric_shapes=None, brightness=1.0, contrast=1.0):
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
            
    # 3. Blend original lines + color canvas
    # Formula:
    # If the original pixel is very dark (ink line), keep it near black.
    # Otherwise, multiply the colors to preserve screentone/hatching textures.
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Ink lines threshold
    ink_mask = gray < 0.28
    paper_mask = gray > 0.90
    
    blended = np.zeros_like(canvas_float)
    
    # Paper zones: show clean colors
    blended[paper_mask] = canvas_float[paper_mask]
    
    # Ink zones: keep near-black ink
    blended[ink_mask] = [0.03, 0.03, 0.04]
    
    # Midtones/hatching/screentones: multiply blend to show original texture + colors
    mid_mask = ~(paper_mask | ink_mask)
    blended[mid_mask] = (orig_float * canvas_float * 1.25)[mid_mask]
    
    # Clamp and convert back to 0-255
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    
    # 4. Apply brightness and contrast adjustments
    # New value = old value * contrast + brightness offset
    if brightness != 1.0 or contrast != 1.0:
        # Center contrast around 128
        blended = cv2.convertScaleAbs(blended, alpha=contrast, beta=int((brightness - 1.0) * 100))
        
    # 5. Apply subtle sharpening to restore line crispness
    kernel = np.array([[0, -0.3, 0], 
                       [-0.3, 2.2, -0.3], 
                       [0, -0.3, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(blended, -1, kernel)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)
            
    # 3. Blend original lines + color canvas
    # Formula:
    # If the original pixel is very dark (ink line), keep it near black.
    # Otherwise, multiply the colors to preserve screentone/hatching textures.
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Ink lines threshold
    ink_mask = gray < 0.28
    paper_mask = gray > 0.90
    
    blended = np.zeros_like(canvas_float)
    
    # Paper zones: show clean colors
    blended[paper_mask] = canvas_float[paper_mask]
    
    # Ink zones: keep near-black ink
    blended[ink_mask] = [0.03, 0.03, 0.04]
    
    # Midtones/hatching/screentones: multiply blend to show original texture + colors
    mid_mask = ~(paper_mask | ink_mask)
    blended[mid_mask] = (orig_float * canvas_float * 1.25)[mid_mask]
    
    # Clamp and convert back to 0-255
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    
    # 4. Apply brightness and contrast adjustments
    # New value = old value * contrast + brightness offset
    if brightness != 1.0 or contrast != 1.0:
        # Center contrast around 128
        blended = cv2.convertScaleAbs(blended, alpha=contrast, beta=int((brightness - 1.0) * 100))
        
    # 5. Apply subtle sharpening to restore line crispness
    kernel = np.array([[0, -0.3, 0], 
                       [-0.3, 2.2, -0.3], 
                       [0, -0.3, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(blended, -1, kernel)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)