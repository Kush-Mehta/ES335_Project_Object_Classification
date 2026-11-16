import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
from PIL import Image
import xml.etree.ElementTree as ET

TARGET_CLASSES = {"cup", "toothpaste", "cans", "apple", "stapler"}

from config import FILTERED_DIR, ANNOTATIONS_DIR, IMAGES_DIR, CROPS_DIR, MANIFEST_PATH


def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        if label not in TARGET_CLASSES:
            continue

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        results.append((label, xmin, ymin, xmax, ymax))

    return results


def main():
    for cls in TARGET_CLASSES:
        os.makedirs(os.path.join(CROPS_DIR, cls), exist_ok=True)

    manifest_rows = []
    class_counts = {cls: 0 for cls in TARGET_CLASSES}

    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]

    for fname in xml_files:
        xml_path = os.path.join(ANNOTATIONS_DIR, fname)
        objects = parse_annotation(xml_path)

        img_name = fname.replace(".xml", ".jpg")
        img_path = os.path.join(IMAGES_DIR, img_name)

        if not objects or not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

        for idx, (label, xmin, ymin, xmax, ymax) in enumerate(objects):
            crop = image.crop((xmin, ymin, xmax, ymax))

            crop_fname = f"{os.path.splitext(img_name)[0]}_{idx}.jpg"
            crop_path = os.path.join(CROPS_DIR, label, crop_fname)
            crop.save(crop_path)

            manifest_rows.append((os.path.join("Crops", label, crop_fname), label))
            class_counts[label] += 1

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(manifest_rows)

    print(f"Total crops: {len(manifest_rows)}")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
