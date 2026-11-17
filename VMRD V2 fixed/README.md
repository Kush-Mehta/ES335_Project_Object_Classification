# VMRD project data

This folder contains the downloaded VMRD dataset and the project-specific files generated from it. Large JPEG images, NumPy arrays, and model files are tracked with Git LFS; install and initialize Git LFS before staging the repository.

## Folder contents

- `Annotations/` — original XML object annotations for the source photographs.
- `JPEGImages/` — original VMRD photographs.
- `Grasps/` — grasp annotations supplied with VMRD; not used by this object-classification project.
- `Filtered/Annotations/` and `Filtered/JPEGImages/` — the 2,674 selected scenes and their annotations that contain at least one of the five target classes.
- `Filtered/Crops/<class>/` — 4,188 object crops extracted from annotated bounding boxes.
- `Filtered/Preprocessed/` — pipeline v1 square-resized color and grayscale-plus-CLAHE crops.
- `Filtered/Preprocessed_v2/` — pipeline v2 crops with aspect-preserving letterbox resizing, stored as color and grayscale-plus-CLAHE views.
- `Filtered/manifest.csv` — crop paths, labels, and source annotation details.
- `Filtered/features.npy`, `features_hog.npy`, `features_color.npy`, and `labels.npy` — feature arrays used by the training and evaluation scripts.

The classes are `apple`, `cans`, `cup`, `stapler`, and `toothpaste`. The project uses object names and bounding boxes; it does not use the grasp annotations.

The VMRD dataset was introduced in [Visual Manipulation Relationship Network for Autonomous Robotics](https://arxiv.org/abs/1802.08857). The dataset authors request citation of that paper. The dataset page describes downloads as free and open, but does not state a standard reuse license; check the source terms before redistributing the data publicly.
