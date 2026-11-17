"""V2 inference preprocessing: aspect-preserving letterbox and separate streams."""
import sys
from pathlib import Path
import cv2
import numpy as np
root = Path(__file__).resolve().parents[1]
if not (root / "vision_features.py").exists():  # Docker image layout
    root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from vision_features import preprocess_crop, extract_hog_features, extract_color_features


def image_stream_features(image_bytes):
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    color, gray = preprocess_crop(image)
    hog_features = extract_hog_features(gray).reshape(1, -1)
    color_features = extract_color_features(color).reshape(1, -1)
    return hog_features, color_features
