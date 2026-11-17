"""Export tuned two-stream models + scalers to results/ for backend use."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import FEATURES_HOG_PATH, FEATURES_COLOR_PATH, LABELS_PATH, RESULTS_DIR

H = np.load(FEATURES_HOG_PATH)
C = np.load(FEATURES_COLOR_PATH)
y_raw = np.load(LABELS_PATH)
le = LabelEncoder()
y = le.fit_transform(y_raw)

H_train, _, C_train, _, y_train, _ = train_test_split(
    H, C, y, test_size=0.2, stratify=y, random_state=42
)
scaler_h = StandardScaler().fit(H_train)
scaler_c = StandardScaler().fit(C_train)

with open(RESULTS_DIR / "best_params_twostream.json") as f:
    best = json.load(f)

def make_clf(name, params):
    if name == "LogisticRegression":
        return CalibratedClassifierCV(LogisticRegression(max_iter=1000, random_state=42, **params), cv=3)
    elif name == "SVM_RBF":
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=42, **params), cv=3)
    else:
        return RandomForestClassifier(random_state=42, **params)

H_tr = scaler_h.transform(H_train)
C_tr = scaler_c.transform(C_train)

for name, info in best.items():
    clf_h = make_clf(name, info["params"])
    clf_c = make_clf(name, info["params"])
    clf_h.fit(H_tr, y_train)
    clf_c.fit(C_tr, y_train)

    slug = name.lower().replace(" ", "_")
    joblib.dump(clf_h, RESULTS_DIR / f"model_{slug}_hog.joblib")
    joblib.dump(clf_c, RESULTS_DIR / f"model_{slug}_color.joblib")
    print(f"Saved: model_{slug}_hog.joblib + model_{slug}_color.joblib")

joblib.dump(scaler_h, RESULTS_DIR / "scaler_hog.joblib")
joblib.dump(scaler_c, RESULTS_DIR / "scaler_color.joblib")
joblib.dump(le,       RESULTS_DIR / "label_encoder.joblib")
print("Saved: scaler_hog, scaler_color, label_encoder")
print(f"\nAll artifacts → {RESULTS_DIR}")