import numpy as np

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
