import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FEATURES_PATH, LABELS_PATH

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("First feature vector (first 10 values):", X[0][:10])
print("First label:", y[0])
print("Unique labels:", np.unique(y))
