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


def preprocess_crop(image_bgr):
    """Return (color resized crop, grayscale crop with CLAHE applied)."""
    color_resized = cv2.resize(image_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    return color_resized, gray_clahe


def extract_hog_features(gray_image):
    return hog(gray_image, **HOG_PARAMS)


def extract_color_histogram(color_image_bgr):
    hsv = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, COLOR_BINS,
        [0, 180, 0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()


def extract_combined_features(color_resized, gray_clahe):
    """Return one vector ordered as HOG values followed by HSV histogram."""
    return np.concatenate((extract_hog_features(gray_clahe), extract_color_histogram(color_resized)))
