import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import numpy as np
import cv2
from vision_features import extract_combined_features

from config import FILTERED_DIR, PREPROCESSED_DIR, MANIFEST_PATH, FEATURES_PATH, LABELS_PATH
GRAY_DIR = os.path.join(PREPROCESSED_DIR, "gray_clahe")
COLOR_DIR = os.path.join(PREPROCESSED_DIR, "color")
FEATURES_OUT = FEATURES_PATH
LABELS_OUT = LABELS_PATH

def main():
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    features_list = []
    labels_list = []

    for i, row in enumerate(rows):
        rel_path = row["image_path"]        # e.g. Crops/apple/00077_0.jpg
        label = row["label"]
        fname = os.path.basename(rel_path)

        gray_path = os.path.join(GRAY_DIR, label, fname)
        color_path = os.path.join(COLOR_DIR, label, fname)

        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)

        if gray_img is None or color_img is None:
            print(f"Warning: could not load {fname} for class {label}")
            continue

        combined = extract_combined_features(color_img, gray_img)
        features_list.append(combined)
        labels_list.append(label)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(rows)}")

    if not features_list:
        raise RuntimeError("No feature rows were created; check the preprocessed image files.")
    X = np.stack(features_list)
    y = np.array(labels_list)

    np.save(FEATURES_OUT, X)
    np.save(LABELS_OUT, y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"HOG feature length: {X.shape[1] - 512}, Color feature length: 512")
    print(f"Saved to {FEATURES_OUT} and {LABELS_OUT}")


if __name__ == "__main__":
    main()
