from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import engineer_features, get_question_frequency_features


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
FEATURE_COLUMNS = [
    "len_diff",
    "common_words",
    "jaccard_sim",
    "word_match_share",
    "max_q_freq",
    "min_q_freq",
]


@dataclass
class InferenceArtifacts:
    xgb_model: object
    xgb_scaler: object
    xgb_weight: float
    lgbm_model: object
    lgbm_scaler: object
    lgbm_weight: float


class DuplicateQuestionRequest(BaseModel):
    question1: str = Field(..., min_length=1, max_length=2000)
    question2: str = Field(..., min_length=1, max_length=2000)


class DuplicateQuestionResponse(BaseModel):
    is_duplicate: bool
    duplicate_probability: float
    threshold: float
    model: str


def _positive_class_probability(raw_proba: np.ndarray) -> float:
    if raw_proba.ndim == 2:
        return float(raw_proba[0, 1])
    return float(raw_proba[0])


def _load_artifacts() -> InferenceArtifacts:
    weights = joblib.load(MODELS_DIR / "ensemble_weights.joblib")
    xgb_weight = float(weights["weight_xgb"])
    lgbm_weight = float(weights["weight_lgbm"])
    total_weight = xgb_weight + lgbm_weight
    if total_weight <= 0:
        raise ValueError("Invalid ensemble weights: total weight must be greater than zero.")

    xgb_model = joblib.load(MODELS_DIR / "tuned_xgboost_model.joblib")
    xgb_scaler = joblib.load(MODELS_DIR / "tuned_xgboost_scaler.joblib")
    lgbm_model = joblib.load(MODELS_DIR / "tuned_lightgbm_model.joblib")
    lgbm_scaler = joblib.load(MODELS_DIR / "tuned_lightgbm_scaler.joblib")

    return InferenceArtifacts(
        xgb_model=xgb_model,
        xgb_scaler=xgb_scaler,
        xgb_weight=xgb_weight,
        lgbm_model=lgbm_model,
        lgbm_scaler=lgbm_scaler,
        lgbm_weight=lgbm_weight,
    )


def _build_feature_frame(question1: str, question2: str) -> pd.DataFrame:
    frame = pd.DataFrame({"question1": [question1], "question2": [question2]})
    engineered = engineer_features(frame)
    engineered = get_question_frequency_features(engineered)
    return engineered[FEATURE_COLUMNS]


app = FastAPI(
    title="Quora Duplicate Question Detection API",
    version="1.0.0",
)

artifacts: InferenceArtifacts | None = None


@app.on_event("startup")
def load_model_artifacts() -> None:
    global artifacts
    artifacts = _load_artifacts()


@app.get("/health")
def health() -> dict[str, str]:
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
    return {"status": "ok"}


@app.post("/predict", response_model=DuplicateQuestionResponse)
def predict(payload: DuplicateQuestionRequest) -> DuplicateQuestionResponse:
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    features = _build_feature_frame(payload.question1, payload.question2)

    xgb_scaled = artifacts.xgb_scaler.transform(features)
    lgbm_scaled = artifacts.lgbm_scaler.transform(features)

    xgb_proba = _positive_class_probability(np.asarray(artifacts.xgb_model.predict_proba(xgb_scaled)))
    lgbm_proba = _positive_class_probability(np.asarray(artifacts.lgbm_model.predict_proba(lgbm_scaled)))

    total_weight = artifacts.xgb_weight + artifacts.lgbm_weight
    duplicate_probability = (
        artifacts.xgb_weight * xgb_proba + artifacts.lgbm_weight * lgbm_proba
    ) / total_weight
    model_name = "tuned-xgboost-lightgbm-weighted-ensemble"

    threshold = 0.5
    return DuplicateQuestionResponse(
        is_duplicate=duplicate_probability >= threshold,
        duplicate_probability=round(float(duplicate_probability), 6),
        threshold=threshold,
        model=model_name,
    )
