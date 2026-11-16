# Classical ML for Object Classification

This project compares Logistic Regression, an RBF Support Vector Machine, and Random Forest using shape and color features from cropped VMRD objects.

## Problem and scope

Given one image crop that contains one object, predict whether it is an apple, can, cup, stapler, or toothpaste. A crop is one labelled object instance, not a full scene. VMRD bounding boxes stand in for an upstream detector. This project does not detect objects in a scene or predict whether a physical grasp will succeed.

## Data

VMRD (Visual Manipulation Relationship Dataset) supplies cluttered robotic-manipulation photographs and XML annotations. The original project used 4,683 annotated photographs; filtering kept 2,674 that contained at least one selected class. Cropping produced 4,188 object instances: stapler 636, toothpaste 892, cup 916, cans 877, apple 867. The XML `object/name` is the label and `bndbox` contains the crop coordinates. The `index`, `father`, `children`, and `Grasps/` data were not used because this project classifies object identity.

The repository keeps the code, manifest, labels, result figures, and text metrics. It excludes the original data, derived images, and the approximately 275 MB feature matrix. Download the [VMRD V2 archive](https://www.dropbox.com/s/ff0f4bqw4s1pxa2/VMRD%20V2%20fixed.tar.gz?dl=0), then place its `Annotations/` and `JPEGImages/` directories under `VMRD V2 fixed/`. The [VMRN paper](https://arxiv.org/abs/1802.08857) is the dataset citation. Check the source for current redistribution terms before sharing the images.

## Setup

The exact Python version used for the original run was not recorded in the supplied project files. Install a supported Python 3 release and the unpinned packages in `requirements.txt`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild and run order

Run from the repository root after the dataset is in place:

1. `python src/01_count_classes.py` - count object instances in the raw annotations.
2. `python src/02_filter_data.py` - copy scenes and XML files that contain a target class.
3. `python src/03_crop_and_build_manifest.py` - crop target boxes and write the manifest.
4. `python src/04_preprocess.py` - resize crops to 128 by 128 pixels and save color and grayscale-plus-CLAHE copies.
5. `python src/05_extract_features.py` - join 8,100 HOG values and a 512-value HSV histogram; save the feature matrix and labels.
6. `python src/06_train_baseline_models.py` - compare the initial model settings using a stratified 80/20 split and five-fold cross-validation on the training pool.
7. `python src/07_tune_hyperparameters.py` - search parameter grids using five-fold cross-validation on the training pool; save `results/best_params.json`.
8. `python src/08_final_evaluation.py` - fit tuned models on the training pool and evaluate the held-out test crops; save confusion matrices, a rejection-curve figure, `results/rejection_data.json`, and a copy for the website in `docs/data/`.
9. `python src/09_error_analysis.py` - save per-class precision, recall, F1, false-positive rate, and false-negative rate.

The 128 by 128 size gives every crop the same image dimensions, so HOG yields equal-length vectors. The grayscale copy receives CLAHE before HOG measures brightness-edge directions; the separate color copy is converted to HSV for a color histogram. The provided scripts fit `StandardScaler` on the training pool and apply it to the held-out test set. They fit the scaler before the training-pool cross-validation folds, so fold validation examples contribute to scaling statistics; this is a limitation of the supplied CV implementation. The held-out test set does not contribute to fitting the scaler.

## Results

Held-out accuracy is 0.9177 for Logistic Regression, 0.9177 for the RBF SVM, and 0.8831 for Random Forest. The [interactive project website](docs/index.html) explains the features, evaluation, class-level errors, limitations, and prediction API.

## Probability scores and thresholding

Logistic Regression returns softmax probabilities. Random Forest averages class fractions at the reached leaves, which is vote-like and is not guaranteed to be calibrated. An SVM's raw output is a distance from a separating boundary, not a probability; `CalibratedClassifierCV(method="sigmoid", cv=5)` applies Platt scaling using training-pool folds. The rejection rule compares the selected model's highest class score with a threshold. Since score calibration differs, the same threshold does not have identical meaning across models.

The demo expects one cropped object like the training samples. Its browser crop tool sends only the selected crop. Its "GRASP" badge is an identity-confidence gate, not a grasp-pose or physical-feasibility prediction.

## Demo API

1. With the features available, run `python src/10_export_model.py` to train demo artifacts on all labelled examples. These artifacts are for inference only and do not replace the reported held-out test results.
2. Install `backend/requirements.txt`, change to `backend/`, and run `uvicorn app:app --reload`.
3. A localhost website preview uses `http://127.0.0.1:8000` automatically. For a hosted backend, define `window.VMRD_API_URL` before `app.js` loads in `docs/index.html`.

Example request:

```bash
curl -X POST -F "image=@crop.jpg" -F "model=svm" -F "threshold=0.80" http://localhost:8000/predict
```

The endpoint accepts `svm`, `lr`, or `rf` and returns the selected prediction, confidence, all five class scores, and threshold decision. Its `/docs` page provides the generated API schema. The exported model files are generated locally and ignored by Git; no hosted endpoint is configured yet.

## Repository map

- `config.py` - shared local data and result paths.
- `src/` - numbered data, feature, training, evaluation, and model-export scripts.
- `results/` - manifest, labels, parameters, result text, and figures.
- `legacy/` - superseded exploratory analysis script.
- `docs/` - responsive interactive project website.
- `backend/` - FastAPI prediction endpoint and Docker setup.

Large data files, generated feature arrays, and model artifacts are excluded by `.gitignore`.

## AI-use disclosure

Codex assisted with repository organization, portable paths, and drafting the README, website, and API implementation. The dataset, pipeline facts, existing metrics, and model-selection results came from the supplied project brief and project artifacts. The existing held-out scores were transcribed from the saved result report; the website does not estimate new test scores from the plots. The student should review the code and be prepared to explain its choices and limitations.

## References and licensing

- Wang et al., "Visual Manipulation Relationship Network for Autonomous Robotics," ICRA 2018, [paper](https://arxiv.org/abs/1802.08857).
- [scikit-learn](https://scikit-learn.org/), [scikit-image](https://scikit-image.org/), and [OpenCV](https://opencv.org/).
- No code license has been selected. The VMRD images have separate terms and are not included here; consult the dataset source before redistributing them.


