"""Create interactive-site metrics from the exported v2 models and fixed test split."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import FEATURES_HOG_PATH, FEATURES_COLOR_PATH, LABELS_PATH, RESULTS_DIR

H = np.load(FEATURES_HOG_PATH)
C = np.load(FEATURES_COLOR_PATH)
y_raw = np.load(LABELS_PATH)
encoder = joblib.load(RESULTS_DIR / "label_encoder.joblib")
y = encoder.transform(y_raw)
classes = list(encoder.classes_)
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

scaler_h = joblib.load(RESULTS_DIR / "scaler_hog.joblib")
scaler_c = joblib.load(RESULTS_DIR / "scaler_color.joblib")
H_test = scaler_h.transform(H[test_idx])
C_test = scaler_c.transform(C[test_idx])
thresholds = np.linspace(0.0, 1.0, 101)
model_files = {
    "Logistic_Regression": ("logisticregression", "Logistic Regression"),
    "SVM_RBF": ("svm_rbf", "SVM (RBF)"),
    "Random_Forest": ("randomforest", "Random Forest"),
}
params = json.loads((RESULTS_DIR / "best_params_twostream.json").read_text(encoding="utf-8"))
results = {
    "classes": classes,
    "split": {"train_pool": int(len(train_idx)), "test": int(len(test_idx)), "test_size": 0.2, "random_state": 42},
    "models": {},
}

for key, (slug, display_name) in model_files.items():
    hog_model = joblib.load(RESULTS_DIR / f"model_{slug}_hog.joblib")
    color_model = joblib.load(RESULTS_DIR / f"model_{slug}_color.joblib")
    model_params = params[{"Logistic_Regression":"LogisticRegression", "SVM_RBF":"SVM_RBF", "Random_Forest":"RandomForest"}[key]]
    alpha = float(model_params["alpha"])
    hog_prob = hog_model.predict_proba(H_test)
    color_prob = color_model.predict_proba(C_test)
    probabilities = alpha * hog_prob + (1 - alpha) * color_prob
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    truth = y[test_idx]
    matrix = confusion_matrix(truth, prediction, labels=np.arange(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(len(classes)), zero_division=0
    )
    rows = []
    for i, cls in enumerate(classes):
        tp = int(matrix[i, i]); fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp); tn = int(matrix.sum() - tp - fp - fn)
        rows.append({"name": cls, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "p": float(precision[i]), "r": float(recall[i]), "f1": float(f1[i]),
                     "support": int(support[i]), "fpr": fp / (fp + tn) if fp + tn else 0.0,
                     "fnr": fn / (fn + tp) if fn + tp else 0.0})
    coverage, accepted_accuracy = [], []
    for threshold in thresholds:
        accepted = confidence >= threshold
        coverage.append(float(accepted.mean()))
        accepted_accuracy.append(float(accuracy_score(truth[accepted], prediction[accepted])) if accepted.any() else None)
    results["models"][key] = {
        "name": display_name,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "matrix": matrix.astype(int).tolist(), "rows": rows,
        "thresholds": thresholds.tolist(), "coverage": coverage,
        "accepted_accuracy": accepted_accuracy, "alpha": alpha,
        "params": model_params["params"],
    }

for path in (RESULTS_DIR / "results_v2.json", ROOT / "docs" / "data" / "v2_results.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
(ROOT / "docs" / "data" / "v2_results.js").write_text(
    "window.VMRD_V2_RESULTS = " + json.dumps(results, separators=(",", ":")) + ";\n",
    encoding="utf-8",
)
for key, model in results["models"].items():
    print(f"{model['name']}: accuracy={model['accuracy']:.4f}, macro-F1={model['macro_f1']:.4f}")
    stapler = next(row for row in model["rows"] if row["name"] == "stapler")
    print(f"  stapler recall={stapler['r']:.4f}, FNR={stapler['fnr']:.4f}")
print(f"Saved exact website results to docs/data/v2_results.json and v2_results.js")
