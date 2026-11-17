"""Shared paths and class settings for the VMRD classification project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "VMRD V2 fixed"
FILTERED_DIR = DATA_DIR / "Filtered"
RAW_ANNOTATIONS_DIR = DATA_DIR / "Annotations"
RAW_IMAGES_DIR = DATA_DIR / "JPEGImages"
ANNOTATIONS_DIR = FILTERED_DIR / "Annotations"
IMAGES_DIR = FILTERED_DIR / "JPEGImages"
CROPS_DIR = FILTERED_DIR / "Crops"
PREPROCESSED_DIR = FILTERED_DIR / "Preprocessed"
MANIFEST_PATH = FILTERED_DIR / "manifest.csv"
FEATURES_PATH = FILTERED_DIR / "features.npy"
LABELS_PATH = FILTERED_DIR / "labels.npy"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
WEB_DATA_DIR = ROOT / "docs" / "data"
PARAMS_PATH = RESULTS_DIR / "best_params.json"
TARGET_CLASSES = ("cup", "toothpaste", "cans", "apple", "stapler")
# --- NEW: letterbox preprocessing + two-stream features ---
PREPROCESSED_V2_DIR = FILTERED_DIR / "Preprocessed_v2"
FEATURES_HOG_PATH   = FILTERED_DIR / "features_hog.npy"
FEATURES_COLOR_PATH = FILTERED_DIR / "features_color.npy"
