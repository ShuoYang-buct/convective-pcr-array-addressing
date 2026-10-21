import numpy as np
import cv2

def bg_subtract_no_norm(gray, sigma):
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    sub = cv2.subtract(gray, bg)
    return np.clip(sub, 0, 255).astype(np.uint8)

def percentile_stretch(gray, low_p=1.0, high_p=99.5):
    low = np.percentile(gray, low_p)
    high = np.percentile(gray, high_p)
    if high <= low:
        high = low + 1
    stretched = np.clip((gray.astype(np.float32) - low) / (high - low) * 255.0, 0, 255)
    return stretched.astype(np.uint8)

def enhance_pipeline1_basic(gray):
    gray = cv2.medianBlur(gray, 3)
    gray = bg_subtract_no_norm(gray, 60)
    gray = percentile_stretch(gray, 1.0, 99.5)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64)).apply(gray)
    return gray
