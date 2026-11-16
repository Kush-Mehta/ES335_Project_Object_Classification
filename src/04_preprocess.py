import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2

from config import FILTERED_DIR, CROPS_DIR, PREPROCESSED_DIR
from vision_features import preprocess_crop

def main():
    classes = [d for d in os.listdir(CROPS_DIR) if os.path.isdir(os.path.join(CROPS_DIR, d))]

    for cls in classes:
        os.makedirs(os.path.join(PREPROCESSED_DIR, "color", cls), exist_ok=True)
        os.makedirs(os.path.join(PREPROCESSED_DIR, "gray_clahe", cls), exist_ok=True)

    total = 0
    for cls in classes:
        crop_dir = os.path.join(CROPS_DIR, cls)
        fnames = os.listdir(crop_dir)
        processed = 0

        for fname in fnames:
            img_path = os.path.join(crop_dir, fname)
            image_bgr = cv2.imread(img_path)

            if image_bgr is None:
                print(f"Warning: could not read {img_path}")
                continue

            color_resized, gray_clahe = preprocess_crop(image_bgr)

            cv2.imwrite(os.path.join(PREPROCESSED_DIR, "color", cls, fname), color_resized)
            cv2.imwrite(os.path.join(PREPROCESSED_DIR, "gray_clahe", cls, fname), gray_clahe)

            total += 1
            processed += 1

        print(f"{cls}: {processed}/{len(fnames)} crops preprocessed")

    print(f"Total preprocessed: {total}")


if __name__ == "__main__":
    main()
