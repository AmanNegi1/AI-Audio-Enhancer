import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_bytes):
    """
    Load image from bytes and return it in three formats:
    - numpy array (RGB)
    - numpy array (Grayscale)
    - PIL Image (RGB)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image bytes.")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(img_rgb)
    
    return img_rgb, img_gray, pil_img

def pil_to_bytes(pil_img, format="PNG"):
    """Convert a PIL image to bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()