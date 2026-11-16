import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shutil
import xml.etree.ElementTree as ET

TARGET_CLASSES = {"cup", "toothpaste", "cans", "apple", "stapler"}

from config import RAW_ANNOTATIONS_DIR as ANNOTATIONS_DIR, RAW_IMAGES_DIR as IMAGES_DIR, FILTERED_DIR as OUT_DIR


def parse_annotation(xml_path):
    """
    Parses one VMRD annotation XML.
    Returns a list of (label, xmin, ymin, xmax, ymax) tuples
    for objects whose label is in TARGET_CLASSES.
    """
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
    out_ann_dir = os.path.join(OUT_DIR, "Annotations")
    out_img_dir = os.path.join(OUT_DIR, "JPEGImages")
    os.makedirs(out_ann_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    all_xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]

    kept = 0
    for fname in all_xml_files:
        xml_path = os.path.join(ANNOTATIONS_DIR, fname)
        objects = parse_annotation(xml_path)

        if not objects:
            continue  # no target-class objects in this image, skip it

        img_name = fname.replace(".xml", ".jpg")
        img_path = os.path.join(IMAGES_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: missing image for {fname}")
            continue

        shutil.copy2(xml_path, os.path.join(out_ann_dir, fname))
        shutil.copy2(img_path, os.path.join(out_img_dir, img_name))
        kept += 1

    print(f"Kept {kept} image/annotation pairs out of {len(all_xml_files)}")


if __name__ == "__main__":
    main()
