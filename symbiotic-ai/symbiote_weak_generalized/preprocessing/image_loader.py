"""Image loading utilities with multi-format support."""

import os
from typing import Optional
import cv2
import numpy as np
from PIL import Image

# Optional: HEIC support (Apple image format). Install with: pip install pillow-heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIC = True
except ImportError:
    HAS_HEIC = False


def load_image_as_rgb(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image as RGB numpy array.
    
    Supports: JPG, JPEG, PNG (via OpenCV).
    Supports: HEIC (via Pillow + pillow-heif if installed).
    
    Returns:
        RGB image as numpy array (H, W, 3), or None if load failed
    """
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.heic':
        if not HAS_HEIC:
            return None
        try:
            pil_img = Image.open(image_path)
            pil_img = pil_img.convert('RGB')
            return np.array(pil_img)
        except Exception:
            return None
    
    # JPG, PNG, etc. via OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


__all__ = ['load_image_as_rgb', 'HAS_HEIC']
