"""Inference wrapper around the same feature functions used by training."""
import sys
from pathlib import Path
import cv2
import numpy as np
root = Path(__file__).resolve().parents[1]
if not (root / "vision_features.py").exists():  # Docker image layout
    root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from vision_features import preprocess_crop, extract_combined_features


def image_features(image_bytes):
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    color, gray = preprocess_crop(image)
    return extract_combined_features(color, gray).reshape(1, -1)
