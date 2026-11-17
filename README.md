# Classical ML for Object Classification

This project studies two versions of a five-class VMRD object classifier. Pipeline v1 concatenated shape and color features; pipeline v2 preserves crop proportions and combines separate HOG and color classifiers. The interactive website shows both versions, with v2 as the current result and prediction pipeline.

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

The original numbered scripts reproduce pipeline v1, the first working baseline:

1. `python src/01_count_classes.py` - count object instances in the raw annotations.
2. `python src/02_filter_data.py` - copy scenes and XML files that contain a target class.
3. `python src/03_crop_and_build_manifest.py` - crop target boxes and write the manifest.
4. `python src/04_preprocess.py` - resize crops to 128 by 128 pixels and save color and grayscale-plus-CLAHE copies.
5. `python src/05_extract_features.py` - join 8,100 HOG values and a 512-value HSV histogram; save the feature matrix and labels.
6. `python src/06_train_baseline_models.py` - compare the initial model settings using a stratified 80/20 split and five-fold cross-validation on the training pool.
7. `python src/07_tune_hyperparameters.py` - search parameter grids using five-fold cross-validation on the training pool; save `results/best_params.json`.
8. `python src/08_final_evaluation.py` - fit tuned models on the training pool and evaluate the held-out test crops; save confusion matrices, a rejection-curve figure, `results/rejection_data.json`, and a copy for the website in `docs/data/`.
9. `python src/09_error_analysis.py` - save per-class precision, recall, F1, false-positive rate, and false-negative rate.

The 128 by 128 size gives every crop the same image dimensions, so HOG yields equal-length vectors. The grayscale copy receives CLAHE before HOG measures brightness-edge directions; the separate color copy is converted to HSV for a color histogram. In v1, the script fit `StandardScaler` on the training pool and applied it to the held-out test set, but it fit that scaler before the CV folds, so fold validation examples contributed to scaling statistics. V2 fixes this during hyperparameter tuning by fitting separate scalers within each fold. In both versions, held-out test images do not contribute to scaler fitting.

## Pipeline v2: revised preprocessing and two-stream fusion

Run these after steps 1–3 above have created the target object crops and manifest:

1. `python src/04b_preprocess_v2.py` - pad the shorter crop dimension with replicated edge pixels, preserving shape, then resize uniformly to 128 × 128 and create the color and grayscale-plus-CLAHE views.
2. `python src/05b_extract_features_twostream.py` - save HOG (8,100 values) and HSV color (512 values) separately.
3. `python src/06b_train_twostream.py` - compare the separate streams with the baseline settings on the training pool.
4. `python src/07b_tune_twostream.py` - tune model settings and HOG fusion weight alpha using five-fold stratified cross-validation. Each fold fits its own stream scalers on that fold's training portion.
5. `python src/08b_final_evaluation_twostream.py` - fit the selected streams on the training pool and evaluate the fixed held-out test set; save v2 confusion matrices and the rejection curve.
6. `python src/09b_error_analysis_twostream.py` - write per-class v2 precision, recall, F1, FPR, and FNR.
7. `python src/10b_export_model_twostream.py` - export the two stream models, their scalers, and label encoder for local prediction. These inference artifacts use the training pool and do not replace the held-out evaluation.
8. `python src/11b_build_web_results_twostream.py` - compute exact v2 heatmap and rejection-curve data from the exported models on the same fixed 838-crop test split; write JSON and browser-ready data files.

V2 has separate scaler/model pairs for HOG and color. At prediction time it combines their class probabilities as `alpha × HOG + (1 − alpha) × color`; all three selected alpha values are 0.5.

## Results

On the held-out 838 crops, v2 accuracy is 0.9153 for Logistic Regression, 0.9200 for the RBF SVM, and 0.9200 for Random Forest. The original v1 results were 0.9177, 0.9177, and 0.8831 respectively. Random Forest's stapler false-negative rate fell from 23.62% in v1 to 10.24% in v2; SVM has the best v2 stapler recall (92.91%). These changes show a useful result on this fixed split, but because preprocessing, stream architecture, and tuning changed together, the comparison cannot assign the gains to only one change. The [interactive project website](docs/index.html) explains the versions, data, evaluation, class-level errors, limitations, and v2 prediction API.

## Probability scores and thresholding

In v2, Logistic Regression and SVM outputs are sigmoid-calibrated inside each stream using three-fold calibration; Random Forest supplies tree-vote probabilities. The HOG and color scores are then blended. Calibration makes the scores usable for fusion, but it does not guarantee perfect probability calibration. The rejection rule compares the highest fused score with a threshold. Since score calibration differs, the same threshold does not have identical meaning across models.

The demo uses the v2 stream models and letterbox preprocessing. Its browser crop tool sends only the selected crop. Its "GRASP" badge is an identity-confidence gate, not a grasp-pose or physical-feasibility prediction.

## Demo API

1. Build the v2 features and parameters, then run `python src/10b_export_model_twostream.py` to train/export the HOG and color demo models on the training pool. These artifacts are for inference only and do not replace the reported held-out test results.
2. Install `backend/requirements.txt`, change to `backend/`, and run `uvicorn app:app --reload`.
3. A localhost website preview uses `http://127.0.0.1:8000` automatically. For a hosted backend, define `window.VMRD_API_URL` before `app.js` loads in `docs/index.html`.

Example request:

```bash
curl -X POST -F "image=@crop.jpg" -F "model=svm" -F "threshold=0.80" http://localhost:8000/predict
```

The endpoint accepts `svm`, `lr`, or `rf` and returns the v2 fused prediction, confidence, all five class scores, fusion weight, and threshold decision. Random Forest is the default in both the website and API; callers can choose another model explicitly. Its `/docs` page provides the generated API schema. The exported model files are generated locally and ignored by Git; no hosted endpoint is configured yet.

The complete website snapshot from before the Random Forest default change is preserved in `docs/archive/v2-before-rf-default/`.

## Repository map

- `config.py` - shared local data and result paths.
- `src/` - numbered data, feature, training, evaluation, and model-export scripts.
- `results/` - manifest, labels, parameters, result text, and figures.
- `legacy/` - superseded exploratory analysis script.
- `docs/` - responsive interactive project website.
- `backend/` - FastAPI prediction endpoint and Docker setup.

Large data files, generated feature arrays, and model artifacts are excluded by `.gitignore`.

## AI-use disclosure

Claude (Anthropic) generated the Python code. Codex (OpenAI) assisted with building the website. Experimental choices, pipeline runs, and result verification were completed independently. The dataset and pipeline facts came from the supplied project artifacts. V1 held-out metrics came from the saved report; v2 metrics and interactive chart data were computed from the exported v2 models on the fixed held-out split, not estimated from plots. The project author should review the code and be prepared to explain its choices and limitations.

## Publish the website with GitHub Pages

The complete static website is in `docs/`. The workflow `.github/workflows/deploy-pages.yml` publishes that folder whenever code is pushed to `main`. After the repository is created, set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**. The site URL will be `https://<owner>.github.io/<repository>/`.

The charts, version selector, result tables, and other browser interactions run on the published site. GitHub Pages does not run the Python prediction API. The upload-and-predict demo connects to `http://127.0.0.1:8000` when the site is opened locally; a hosted prediction service would need its own deployment and a `window.VMRD_API_URL` setting.

## References and licensing

- Wang et al., "Visual Manipulation Relationship Network for Autonomous Robotics," ICRA 2018, [paper](https://arxiv.org/abs/1802.08857).
- [scikit-learn](https://scikit-learn.org/), [scikit-image](https://scikit-image.org/), and [OpenCV](https://opencv.org/).
- No code license has been selected. The VMRD images have separate terms and are not included here; consult the dataset source before redistributing them.


