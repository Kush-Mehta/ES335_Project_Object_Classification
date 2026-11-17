"""Letterbox-preserving preprocessing (v2). Writes to Preprocessed_v2/."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from config import CROPS_DIR, PREPROCESSED_V2_DIR
from vision_features import preprocess_crop

def main():
    classes = [d for d in os.listdir(CROPS_DIR) if os.path.isdir(os.path.join(CROPS_DIR, d))]
    for cls in classes:
        os.makedirs(os.path.join(PREPROCESSED_V2_DIR, "color", cls), exist_ok=True)
        os.makedirs(os.path.join(PREPROCESSED_V2_DIR, "gray_clahe", cls), exist_ok=True)

    total = 0
    for cls in classes:
        for fname in os.listdir(os.path.join(CROPS_DIR, cls)):
            img = cv2.imread(os.path.join(CROPS_DIR, cls, fname))
            if img is None:
                continue
            color, gray = preprocess_crop(img)
            cv2.imwrite(str(os.path.join(PREPROCESSED_V2_DIR, "color", cls, fname)), color)
            cv2.imwrite(str(os.path.join(PREPROCESSED_V2_DIR, "gray_clahe", cls, fname)), gray)
            total += 1
        print(f"  {cls}: done")
    print(f"Total: {total}")

if __name__ == "__main__":
    main()