"""Final held-out evaluation of two-stream late-fusion models."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, f1_score
from config import FEATURES_HOG_PATH, FEATURES_COLOR_PATH, LABELS_PATH, RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures_v2"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

H = np.load(FEATURES_HOG_PATH)
C = np.load(FEATURES_COLOR_PATH)
y_raw = np.load(LABELS_PATH)
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

H_train, H_test, C_train, C_test, y_train, y_test = train_test_split(
    H, C, y, test_size=0.2, stratify=y, random_state=42
)

scaler_h = StandardScaler().fit(H_train)
scaler_c = StandardScaler().fit(C_train)
H_tr = scaler_h.transform(H_train);  H_te = scaler_h.transform(H_test)
C_tr = scaler_c.transform(C_train);  C_te = scaler_c.transform(C_test)

with open(RESULTS_DIR / "best_params_twostream.json") as f:
    best = json.load(f)

def make_clf(name, params):
    if name == "LogisticRegression":
        return CalibratedClassifierCV(LogisticRegression(max_iter=1000, random_state=42, **params), cv=3)
    elif name == "SVM_RBF":
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=42, **params), cv=3)
    else:
        return RandomForestClassifier(random_state=42, **params)

MODEL_DISPLAY = {"LogisticRegression": "Logistic Regression", "SVM_RBF": "SVM RBF", "RandomForest": "Random Forest"}

fig_rej, ax_rej = plt.subplots(figsize=(8, 5))
thresholds = np.linspace(0.0, 1.0, 101)
web_results = {
    "classes": class_names,
    "split": {"train_pool": int(len(y_train)), "test": int(len(y_test)), "test_size": 0.2, "random_state": 42},
    "models": {},
}

for name, info in best.items():
    params = info["params"]
    alpha  = info["alpha"]
    label  = MODEL_DISPLAY[name]

    clf_h = make_clf(name, params)
    clf_c = make_clf(name, params)
    clf_h.fit(H_tr, y_train)
    clf_c.fit(C_tr, y_train)

    ph = clf_h.predict_proba(H_te)
    pc = clf_c.predict_proba(C_te)
    proba = alpha * ph + (1 - alpha) * pc          # fused probabilities
    conf  = proba.max(axis=1)
    preds = proba.argmax(axis=1)

    test_acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"{label:20s}  test acc = {test_acc:.4f}  (alpha={alpha})")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, preds, labels=np.arange(len(class_names)), zero_division=0
    )
    class_rows = []
    for i, cls in enumerate(class_names):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp); tn = int(cm.sum() - tp - fp - fn)
        class_rows.append({
            "name": cls, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "p": float(precision[i]), "r": float(recall[i]), "f1": float(f1[i]),
            "support": int(support[i]),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        })
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
    ax.set_title(f"{label} — test acc {test_acc:.4f}")
    fig.tight_layout()
    fname = f"cm_{name}_v2.png"
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

    # Rejection curve data
    coverages, accs = [], []
    for t in thresholds:
        mask = conf >= t
        coverages.append(float(mask.mean()))
        accs.append(float(accuracy_score(y_test[mask], preds[mask])) if mask.sum() else None)
    ax_rej.plot(coverages, accs, label=label)
    web_key = {"LogisticRegression": "Logistic_Regression", "SVM_RBF": "SVM_RBF", "RandomForest": "Random_Forest"}[name]
    web_results["models"][web_key] = {
        "name": label, "accuracy": float(test_acc), "macro_f1": float(macro_f1),
        "matrix": cm.astype(int).tolist(), "rows": class_rows,
        "thresholds": thresholds.tolist(), "coverage": coverages, "accepted_accuracy": accs,
        "alpha": float(alpha), "params": params,
    }

ax_rej.set_xlabel("Coverage (fraction of samples not rejected)")
ax_rej.set_ylabel("Accuracy on accepted samples")
ax_rej.set_title("Confidence-Threshold Rejection Curve (Two-Stream)")
ax_rej.legend()
ax_rej.grid(True)
fig_rej.tight_layout()
fig_rej.savefig(FIGURES_DIR / "rejection_curve_v2.png", dpi=150)
plt.close(fig_rej)
for path in (RESULTS_DIR / "results_v2.json", Path(__file__).resolve().parents[1] / "docs" / "data" / "v2_results.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(web_results, indent=2), encoding="utf-8")
web_js = "window.VMRD_V2_RESULTS = " + json.dumps(web_results, separators=(",", ":")) + ";\n"
(Path(__file__).resolve().parents[1] / "docs" / "data" / "v2_results.js").write_text(web_js, encoding="utf-8")
print(f"\nRejection curve and interactive website data saved.")
print(f"All figures → {FIGURES_DIR}")
