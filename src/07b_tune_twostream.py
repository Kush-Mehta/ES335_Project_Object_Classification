"""Tune alpha (fusion weight) + hyperparams for two-stream late-fusion models."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from config import FEATURES_HOG_PATH, FEATURES_COLOR_PATH, LABELS_PATH, RESULTS_DIR

H = np.load(FEATURES_HOG_PATH)
C = np.load(FEATURES_COLOR_PATH)
y_raw = np.load(LABELS_PATH)
le = LabelEncoder()
y = le.fit_transform(y_raw)

H_train, H_test, C_train, C_test, y_train, y_test = train_test_split(
    H, C, y, test_size=0.2, stratify=y, random_state=42
)

ALPHAS = [0.3, 0.5, 0.7]   # weight on HOG; 1-alpha goes to color

GRIDS = {
    "LogisticRegression": [{"C": c} for c in [0.01, 0.1, 1.0]],
    "SVM_RBF":            [{"C": c, "gamma": g} for c in [0.1, 1, 10] for g in ["scale", "auto"]],
    "RandomForest":       [{"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
                           for n in [100, 300] for d in [10, 20, None] for m in [1, 5]],
}

def make_clf(name, params):
    if name == "LogisticRegression":
        base = LogisticRegression(max_iter=1000, random_state=42, **params)
        return CalibratedClassifierCV(base, cv=3)
    elif name == "SVM_RBF":
        base = SVC(kernel="rbf", random_state=42, **params)
        return CalibratedClassifierCV(base, cv=3)
    else:
        return RandomForestClassifier(random_state=42, **params)

def fusion_cv(clf_h, clf_c, alpha, Xh, Xc, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accs = []
    for tr, va in skf.split(Xh, y):
        sh = StandardScaler().fit(Xh[tr])
        sc = StandardScaler().fit(Xc[tr])
        clf_h.fit(sh.transform(Xh[tr]), y[tr])
        clf_c.fit(sc.transform(Xc[tr]), y[tr])
        ph = clf_h.predict_proba(sh.transform(Xh[va]))
        pc = clf_c.predict_proba(sc.transform(Xc[va]))
        preds = np.argmax(alpha * ph + (1 - alpha) * pc, axis=1)
        accs.append(accuracy_score(y[va], preds))
    return np.mean(accs)

best_overall = {}

for name, grid in GRIDS.items():
    print(f"\nTuning {name} ...")
    best_acc, best_params, best_alpha = 0, None, None
    for params in grid:
        for alpha in ALPHAS:
            clf_h = make_clf(name, params)
            clf_c = make_clf(name, params)
            acc = fusion_cv(clf_h, clf_c, alpha, H_train, C_train, y_train)
            if acc > best_acc:
                best_acc, best_params, best_alpha = acc, params, alpha
    print(f"  Best CV acc = {best_acc:.4f}  alpha={best_alpha}  params={best_params}")
    best_overall[name] = {"params": best_params, "alpha": best_alpha, "cv_acc": best_acc}

out = RESULTS_DIR / "best_params_twostream.json"
with open(out, "w") as f:
    json.dump(best_overall, f, indent=2)
print(f"\nSaved → {out}")