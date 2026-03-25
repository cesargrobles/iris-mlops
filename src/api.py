# src/api.py
"""FastAPI serving layer for Iris predictions."""

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import wandb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.logger import get_logger
from src.utils import load_model
from src.infer import run_inference

logger = get_logger(__name__)

app = FastAPI(title="Iris MLOps API", version="0.1.0")

model = None


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionRequest(BaseModel):
    instances: List[IrisFeatures]


class PredictionResponse(BaseModel):
    predictions: List[Any]
    proba: List[float]


def download_prod_model() -> Path:
    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]
    artifact_name = os.environ.get("WANDB_MODEL_ARTIFACT", "model")

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="inference",
    )

    artifact_ref = f"{entity}/{project}/{artifact_name}:prod"
    logger.info(f"Downloading W&B artifact: {artifact_ref}")

    artifact = run.use_artifact(artifact_ref, type="model")
    artifact_dir = Path(artifact.download())

    candidates = list(artifact_dir.rglob("*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No .joblib file found in downloaded artifact at {artifact_dir}")

    logger.info(f"Downloaded model artifact to {candidates[0]}")
    return candidates[0]


@app.on_event("startup")
def load_pipeline():
    global model
    try:
        model_path = download_prod_model()
        model = load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        model = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    df = pd.DataFrame([instance.model_dump() for instance in request.instances])
    if df.empty:
        raise HTTPException(status_code=400, detail="No instances provided")

    preds = run_inference(model, df, include_proba=True)
    return PredictionResponse(
        predictions=preds["prediction"].tolist(),
        proba=preds["proba"].tolist(),
    )