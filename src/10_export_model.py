"""Train full-data demo models; these artifacts are not evaluation results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from config import FEATURES_PATH, LABELS_PATH, PARAMS_PATH, RESULTS_DIR


def main():
    X, y = np.load(FEATURES_PATH), np.load(LABELS_PATH)
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = {
        "lr": LogisticRegression(max_iter=3000, **params["LogisticRegression"]["best_params"]),
        "svm": CalibratedClassifierCV(SVC(kernel="rbf", **params["SVM_RBF"]["best_params"]), method="sigmoid", cv=5),
        "rf": RandomForestClassifier(random_state=42, **params["RandomForest"]["best_params"]),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(Xs, y)
        joblib.dump({"scaler": scaler, "model": model, "classes": model.classes_}, RESULTS_DIR / f"model_{name}.joblib")
        print(f"Saved demo artifact: {RESULTS_DIR / f'model_{name}.joblib'}")


if __name__ == "__main__":
    main()
