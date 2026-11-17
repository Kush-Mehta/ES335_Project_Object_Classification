"""Two-stream feature extraction from Preprocessed_v2. Saves features_hog.npy and features_color.npy."""
import os, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from config import PREPROCESSED_V2_DIR, MANIFEST_PATH, LABELS_PATH, FEATURES_HOG_PATH, FEATURES_COLOR_PATH
from vision_features import extract_hog_features, extract_color_features

GRAY_DIR = os.path.join(PREPROCESSED_V2_DIR, "gray_clahe")
COLOR_DIR = os.path.join(PREPROCESSED_V2_DIR, "color")

def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    hog_feats, color_feats, labels = [], [], []
    missing = 0
    for i, row in enumerate(rows):
        # image_path is like "Crops\stapler\00022_0.jpg"
        parts = Path(row["image_path"]).parts
        cls = parts[-2]       # e.g. "stapler"
        fname = parts[-1]     # e.g. "00022_0.jpg"

        gray = cv2.imread(os.path.join(GRAY_DIR, cls, fname), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(os.path.join(COLOR_DIR, cls, fname))

        if gray is None or color is None:
            print(f"Missing: {cls}/{fname}")
            missing += 1
            continue

        hog_feats.append(extract_hog_features(gray))
        color_feats.append(extract_color_features(color))
        labels.append(cls)

        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(rows)}")

    H = np.array(hog_feats)
    C = np.array(color_feats)
    y = np.array(labels)

    np.save(FEATURES_HOG_PATH, H)
    np.save(FEATURES_COLOR_PATH, C)
    np.save(LABELS_PATH, y)

    print(f"\nHOG:   {H.shape}  → {FEATURES_HOG_PATH}")
    print(f"Color: {C.shape}  → {FEATURES_COLOR_PATH}")
    print(f"Labels: {y.shape}")
    if missing:
        print(f"WARNING: {missing} files missing — did 04b finish correctly?")

if __name__ == "__main__":
    main()