"""Blur detection utilities using Laplacian variance."""

import cv2
import numpy as np


def is_blurry(image: np.ndarray, threshold: float) -> bool:
    """
    Check if an image is blurry using Laplacian variance.
    
    Uses the blur detection method from blurry.py:
    https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    
    Args:
        image: RGB or BGR image as numpy array
        threshold: Laplacian variance threshold (lower = more blurry)
    
    Returns:
        True if image is blurry (variance < threshold), False otherwise
    """
    # Convert to grayscale for Laplacian computation
    # Handle both RGB and BGR inputs
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:,:,0]
    else:
        gray = image
    
    # Compute Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


__all__ = ['is_blurry']
