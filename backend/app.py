"""Prediction API for single-object crops; it does not detect or assess grasps."""
import os
import json
from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocess import image_stream_features

app = FastAPI(title="VMRD object classification demo")
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "results"))
MODEL_SLUGS = {"svm": "svm_rbf", "lr": "logisticregression", "rf": "randomforest"}
PARAMETER_KEYS = {"svm": "SVM_RBF", "lr": "LogisticRegression", "rf": "RandomForest"}


@app.get("/health")
def health():
    return {"status": "ok", "pipeline": "v2", "models": list(MODEL_SLUGS)}


@lru_cache(maxsize=3)
def load_model_pair(model):
    """Load a selected HOG/color pair once; avoid reloading large SVMs per request."""
    slug = MODEL_SLUGS[model]
    key = PARAMETER_KEYS[model]
    with open(MODEL_DIR / "best_params_twostream.json", encoding="utf-8") as f:
        alpha = float(json.load(f)[key]["alpha"])
    return {
        "hog": joblib.load(MODEL_DIR / f"model_{slug}_hog.joblib"),
        "color": joblib.load(MODEL_DIR / f"model_{slug}_color.joblib"),
        "scaler_hog": joblib.load(MODEL_DIR / "scaler_hog.joblib"),
        "scaler_color": joblib.load(MODEL_DIR / "scaler_color.joblib"),
        "encoder": joblib.load(MODEL_DIR / "label_encoder.joblib"),
        "alpha": alpha,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...), model: str = Form("svm"), threshold: float = Form(0.8)):
    """Classify one user-cropped object with the v2 two-stream late-fusion pipeline."""
    if model not in {"svm", "lr", "rf"}:
        raise HTTPException(400, "model must be svm, lr, or rf")
    if not 0 <= threshold <= 1:
        raise HTTPException(400, "threshold must be between 0 and 1")
    try:
        payload = await image.read()
        hog_features, color_features = image_stream_features(payload)
        artifact = load_model_pair(model)
        hog_scaled = artifact["scaler_hog"].transform(hog_features)
        color_scaled = artifact["scaler_color"].transform(color_features)
        hog_probs = artifact["hog"].predict_proba(hog_scaled)[0]
        color_probs = artifact["color"].predict_proba(color_scaled)[0]
        probs = artifact["alpha"] * hog_probs + (1 - artifact["alpha"]) * color_probs
        class_indices = artifact["hog"].classes_.astype(int)
        classes = artifact["encoder"].inverse_transform(class_indices)
        top = int(probs.argmax())
    except FileNotFoundError:
        raise HTTPException(503, "The v2 demo models are missing. Build them with src/10b_export_model_twostream.py.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    confidence = float(probs[top])
    return {"predicted_class": str(classes[top]), "confidence": confidence,
            "all_class_probabilities": {str(c): float(p) for c, p in zip(classes, probs)},
            "decision": "GRASP" if confidence >= threshold else "FLAG_FOR_REVIEW",
            "threshold_used": threshold, "model_used": model,
            "pipeline_version": "v2", "fusion_alpha": artifact["alpha"]}
