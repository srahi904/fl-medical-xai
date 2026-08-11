"""
backend/app.py
---------------
FastAPI server that loads the trained (federated) global model and exposes
a REST API for the React frontend:

    POST /predict   -> upload an image, get back prediction + Grad-CAM heatmap
    GET  /history    -> FL training history (accuracy per round) for the dashboard chart
    GET  /health      -> simple healthcheck

Run locally:
    cd backend
    uvicorn app:app --reload --port 8000

The frontend (frontend/src/App.jsx) expects this to run on http://localhost:8000
"""

import base64
import io
import json
import os
import sys

import cv2
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model  # noqa: E402
from gradcam import generate_gradcam  # noqa: E402
from dataset import get_transforms  # noqa: E402

# ---- Config ----
MODEL_PATH = os.environ.get("MODEL_PATH", "../saved_models/global_model_final.pth")
CLASS_NAMES = os.environ.get("CLASS_NAMES", "NORMAL,PNEUMONIA").split(",")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "../results/history.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Federated Medical Diagnosis API")

# Allow the React dev server (localhost:5173 / 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = build_model(num_classes=len(CLASS_NAMES), pretrained=False, device=DEVICE)
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Loaded model weights from {MODEL_PATH}")
        else:
            print(f"WARNING: no weights found at {MODEL_PATH}. "
                  f"Using randomly initialized model -- train first!")
        _model.eval()
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "classes": CLASS_NAMES}


@app.get("/history")
def history():
    """Returns FL training history (round-by-round accuracy) for the dashboard chart."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {"round": [], "accuracy": [], "loss": [], "f1": [], "auc": []}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    model = get_model()

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    transform = get_transforms(train=False)
    tensor = transform(img)

    overlay, pred_class, confidence = generate_gradcam(model, tensor, device=DEVICE)

    # encode heatmap overlay as base64 PNG so the React app can render it directly
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", overlay_bgr)
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": round(confidence * 100, 2),
        "heatmap_base64": f"data:image/png;base64,{heatmap_b64}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
