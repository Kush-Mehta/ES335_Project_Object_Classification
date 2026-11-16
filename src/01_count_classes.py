import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xml.etree.ElementTree as ET
from collections import Counter

from config import RAW_ANNOTATIONS_DIR as ANNOT_DIR

counts = Counter()
for fname in os.listdir(ANNOT_DIR):
    if not fname.endswith(".xml"):
        continue
    tree = ET.parse(os.path.join(ANNOT_DIR, fname))
    for obj in tree.getroot().findall("object"):
        counts[obj.find("name").text] += 1

for name, c in counts.most_common():
    print(f"{name}: {c}")
