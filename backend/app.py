"""Prediction API for single-object crops; it does not detect or assess grasps."""
import os
from pathlib import Path
import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocess import image_features

app = FastAPI(title="VMRD object classification demo")
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "results"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...), model: str = Form("svm"), threshold: float = Form(0.8)):
    """Classify one user-cropped object. This is not a scene detector or grasp-pose predictor."""
    if model not in {"svm", "lr", "rf"}:
        raise HTTPException(400, "model must be svm, lr, or rf")
    if not 0 <= threshold <= 1:
        raise HTTPException(400, "threshold must be between 0 and 1")
    try:
        payload = await image.read()
        features = image_features(payload)
        artifact = joblib.load(MODEL_DIR / f"model_{model}.joblib")
        probs = artifact["model"].predict_proba(artifact["scaler"].transform(features))[0]
        classes = artifact["classes"]
        top = int(probs.argmax())
    except FileNotFoundError:
        raise HTTPException(503, "Demo models are not installed. Build them with src/10_export_model.py.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    confidence = float(probs[top])
    return {"predicted_class": str(classes[top]), "confidence": confidence,
            "all_class_probabilities": {str(c): float(p) for c, p in zip(classes, probs)},
            "decision": "GRASP" if confidence >= threshold else "FLAG_FOR_REVIEW",
            "threshold_used": threshold, "model_used": model}
