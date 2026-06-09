import cv2
import numpy as np

def detect_speech_bubbles(img_gray, min_area=1500, max_area=150000, solidity_thresh=0.75, white_thresh=240):
    """
    Detects possible speech bubbles in a grayscale manga image.
    Returns a list of dictionaries containing bounds:
    {'x': x, 'y': y, 'w': w, 'h': h, 'contour': contour}
    """
    # Threshold to find bright white regions (most bubbles are white)
    _, thresh = cv2.threshold(img_gray, white_thresh, 255, cv2.THRESH_BINARY)
    
    # Clean up small noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            # Check solidity (how compact/convex it is)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if solidity >= solidity_thresh:
                x, y, w, h = cv2.boundingRect(c)
                # Keep coordinates and the contour itself
                detected.append({
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h),
                    'area': float(area),
                    'solidity': float(solidity)
                })
                
    return detected

def erase_regions(img_rgb, regions, fill_color=(255, 255, 255)):
    """
    Fills specific rect/polygon regions on the image with a solid color (usually white).
    regions: list of dicts with {'x', 'y', 'w', 'h'} or custom drawings
    """
    out = img_rgb.copy()
    for r in regions:
        x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
        cv2.rectangle(out, (x, y), (x + w, y + h), fill_color, -1)
    return out
