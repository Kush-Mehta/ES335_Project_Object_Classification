import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from config import FEATURES_PATH, LABELS_PATH, PARAMS_PATH, FIGURES_DIR, RESULTS_DIR, WEB_DATA_DIR
OUT_DIR = FIGURES_DIR

import os
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEB_DATA_DIR, exist_ok=True)

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

X_trainpool, X_test, y_trainpool, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_trainpool_scaled = scaler.fit_transform(X_trainpool)
X_test_scaled      = scaler.transform(X_test)

with open(PARAMS_PATH) as f:
    params = json.load(f)

classes = sorted(np.unique(y))

# Define the 3 tuned models
models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=3000,
        **params["LogisticRegression"]["best_params"]
    ),
    "SVM_RBF": CalibratedClassifierCV(
        SVC(kernel="rbf", **params["SVM_RBF"]["best_params"]),
        method="sigmoid", cv=5
    ),
    "Random_Forest": RandomForestClassifier(
        random_state=42,
        **params["RandomForest"]["best_params"]
    ),
}

# LR and RF already output probabilities natively;
# SVM needs CalibratedClassifierCV (same as before) for rejection curve

def get_probs(model, X):
    """Works for all 3 model types."""
    return model.predict_proba(X)

thresholds = np.arange(0.0, 1.01, 0.05)

plt.figure(figsize=(8, 5))
rejection_data = {}

for model_name, model in models.items():
    print(f"Fitting {model_name}...")
    model.fit(X_trainpool_scaled, y_trainpool)

    # --- Held-out test accuracy ---
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {acc:.4f}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name.replace('_',' ')}\nTest acc: {acc:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"cm_{model_name}.png"), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved.")

    # --- Rejection curve ---
    probs       = get_probs(model, X_test_scaled)
    confidences = np.max(probs, axis=1)
    pred_classes = model.classes_[np.argmax(probs, axis=1)]

    coverages  = []
    accuracies = []
    for t in thresholds:
        mask = confidences >= t
        coverages.append(mask.mean())
        accuracies.append(
            accuracy_score(y_test[mask], pred_classes[mask]) if mask.sum() > 0 else np.nan
        )

    rejection_data[model_name] = {"thresholds": thresholds.tolist(), "coverage": coverages,
                                  "accuracy": [None if np.isnan(v) else float(v) for v in accuracies]}

    plt.plot(coverages, accuracies, marker="o", label=model_name.replace("_", " "))
    print(f"  Rejection curve computed.")

    # Print key operating points
    print(f"  {'Threshold':>10}  {'Coverage':>8}  {'Accuracy':>8}")
    for t, c, a in zip(thresholds, coverages, accuracies):
        if not np.isnan(a):
            print(f"  {t:>10.2f}  {c:>8.3f}  {a:>8.3f}")

# --- Combined rejection curve plot ---
plt.xlabel("Coverage (fraction of items accepted)")
plt.ylabel("Accuracy on accepted items")
plt.title("Accuracy vs. Coverage — All Models (Tuned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rejection_curve_all_models.png"), dpi=150)
with open(RESULTS_DIR / "rejection_data.json", "w", encoding="utf-8") as f:
    json.dump(rejection_data, f, indent=2)
with open(WEB_DATA_DIR / "rejection_data.json", "w", encoding="utf-8") as f:
    json.dump(rejection_data, f, indent=2)
plt.show()
print(f"\nAll figures saved to {OUT_DIR}")
