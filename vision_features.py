"""Shared crop preprocessing and HOG+HSV extraction for training and inference."""
import cv2
import numpy as np
from skimage.feature import hog

TARGET_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
COLOR_BINS = (8, 8, 8)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _letterbox_square(image_bgr):
    """Pad shorter side with replicated edge pixels → square, no stretch."""
    h, w = image_bgr.shape[:2]
    size = max(h, w)
    pad_h, pad_w = size - h, size - w
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        image_bgr, top, pad_h - top, left, pad_w - left,
        borderType=cv2.BORDER_REPLICATE
    )


def preprocess_crop(image_bgr):
    """Letterbox to square, then resize uniformly. Returns (color, gray_clahe)."""
    squared = _letterbox_square(image_bgr)
    color = cv2.resize(squared, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, _clahe.apply(gray)


def extract_hog_features(gray_clahe):
    """8100-dim HOG vector from a 128×128 grayscale+CLAHE image."""
    return hog(gray_clahe, **HOG_PARAMS, feature_vector=True)


def extract_color_features(color_bgr):
    """512-dim HSV histogram from a 128×128 BGR image."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(COLOR_BINS),
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def extract_combined_features(color_bgr, gray_clahe):
    """Legacy: concatenated 8612-dim vector (kept for old scripts/backend)."""
    return np.concatenate([
        extract_hog_features(gray_clahe),
        extract_color_features(color_bgr)
    ])