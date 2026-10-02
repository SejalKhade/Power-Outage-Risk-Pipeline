"""
api/main.py
-----------
FastAPI REST endpoint for Power Outage Risk predictions.
Loads best_model.pkl and returns risk classification for any utility.

Run locally:
    pip install fastapi uvicorn
    uvicorn api.main:app --reload
    Open: http://localhost:8000/docs

Resume claim this enables:
    "served predictions via FastAPI REST endpoint"
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Power Outage Risk API",
    description="Predicts high-risk electric utilities using EIA-861 + NOAA storm data.",
    version="1.0.0",
)

# Load model once at startup
MODEL_PATH = "outputs/models/best_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded: {type(model.named_steps['classifier']).__name__}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")


# ── Request schema ─────────────────────────────────────────────────
class UtilityFeatures(BaseModel):
    """Input features for a single utility."""
    weather_event_count:        float = Field(..., description="Total storm events linked to this utility")
    total_damage_usd:           float = Field(..., description="Total property + crop damage in USD")
    INJURIES_DIRECT:            float = Field(0.0, description="Direct injuries from storm events")
    DEATHS_DIRECT:              float = Field(0.0, description="Direct deaths from storm events")
    MAGNITUDE:                  float = Field(0.0, description="Max storm magnitude recorded")
    months_with_events:         float = Field(0.0, description="Number of months with storm events")
    County_Count:               float = Field(1.0, description="Number of counties served")
    TRE:                        int   = Field(0, description="1 if utility is in TRE NERC region")
    FRCC:                       int   = Field(0)
    MRO:                        int   = Field(0)
    NPCC:                       int   = Field(0)
    RFC:                        int   = Field(0)
    SERC:                       int   = Field(0)
    SPP:                        int   = Field(0)
    WECC:                       int   = Field(0)
    ERCOT:                      int   = Field(0)
    PJM:                        int   = Field(0)
    NYISO:                      int   = Field(0)
    MISO:                       int   = Field(0)
    ISONE:                      int   = Field(0)
    log_total_damage:           float = Field(0.0, description="log1p of total damage")
    human_impact_score:         float = Field(0.0, description="Injuries + Deaths×10")
    Ownership:                  str   = Field("Cooperative", description="Utility ownership type")
    NERC_Region:                str   = Field("SERC", description="Primary NERC region")

    class Config:
        json_schema_extra = {
            "example": {
                "weather_event_count": 82,
                "total_damage_usd": 1_500_000,
                "INJURIES_DIRECT": 2,
                "DEATHS_DIRECT": 0,
                "MAGNITUDE": 45.0,
                "months_with_events": 7,
                "County_Count": 5,
                "TRE": 0, "FRCC": 0, "MRO": 0, "NPCC": 0,
                "RFC": 0, "SERC": 1, "SPP": 0, "WECC": 0,
                "ERCOT": 0, "PJM": 0, "NYISO": 0, "MISO": 0, "ISONE": 0,
                "log_total_damage": 14.22,
                "human_impact_score": 2.0,
                "Ownership": "Cooperative",
                "NERC_Region": "SERC",
            }
        }


# ── Response schema ────────────────────────────────────────────────
class RiskPrediction(BaseModel):
    risk_label:       str
    risk_probability: float
    confidence:       str
    message:          str


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "project": "Power Outage Risk Analysis",
        "author":  "Sejal Khade — MS Data Science, UT Arlington",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "model_type":   type(model.named_steps["classifier"]).__name__ if model else None,
    }


@app.post("/predict", response_model=RiskPrediction)
def predict(features: UtilityFeatures):
    """
    Predict whether a utility is High Risk based on
    storm exposure, grid structure, and regional features.
    Returns risk label, probability, and confidence level.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py first to generate best_model.pkl"
        )

    # Build feature dataframe matching training columns
    row = {
    "weather_event_count":       features.weather_event_count,
    "total_damage_usd":          features.total_damage_usd,
    "total_property_damage_usd": features.total_damage_usd,
    "total_crops_damage_usd":    features.total_damage_usd * 0.1,
    "log_total_damage":          float(np.log1p(features.total_damage_usd)),
    "INJURIES_DIRECT":           features.INJURIES_DIRECT,
    "INJURIES_INDIRECT":         features.INJURIES_DIRECT * 0.5,
    "DEATHS_DIRECT":             features.DEATHS_DIRECT,
    "DEATHS_INDIRECT":           features.DEATHS_DIRECT * 0.5,
    "human_impact_score":        features.INJURIES_DIRECT + features.DEATHS_DIRECT * 10,
    "MAGNITUDE":                 features.MAGNITUDE,
    "months_with_events":        features.months_with_events,
    "County_Count":              features.County_Count,
    "TRE":   features.TRE,
    "FRCC":  features.FRCC,
    "MRO":   features.MRO,
    "NPCC":  features.NPCC,
    "RFC":   features.RFC,
    "SERC":  features.SERC,
    "SPP":   features.SPP,
    "WECC":  features.WECC,
    "CAISO": 0,
    "ERCOT": features.ERCOT,
    "PJM":   features.PJM,
    "NYISO": features.NYISO,
    "MISO":  features.MISO,
    "ISONE": features.ISONE,
    "Ownership":   features.Ownership,
    "NERC Region": features.NERC_Region,
}
    X = pd.DataFrame([row])

    # Predict
    proba      = float(model.predict_proba(X)[0][1])
    label      = "High Risk" if proba >= 0.5 else "Medium/Low Risk"
    confidence = (
        "High"   if proba >= 0.75 or proba <= 0.25 else
        "Medium" if proba >= 0.60 or proba <= 0.40 else
        "Low"
    )

    messages = {
        "High Risk":        f"Utility flagged as HIGH RISK (probability: {proba:.2%}). Recommend infrastructure review.",
        "Medium/Low Risk":  f"Utility classified as lower risk (probability: {proba:.2%}). Monitor storm exposure.",
    }

    return RiskPrediction(
        risk_label       = label,
        risk_probability = round(proba, 4),
        confidence       = confidence,
        message          = messages[label],
    )


@app.get("/model-info")
def model_info():
    """Returns information about the deployed model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type":    type(model.named_steps["classifier"]).__name__,
        "pipeline_steps": list(model.named_steps.keys()),
        "description":   (
            "Logistic Regression trained on Utility + Weather features. "
            "Leakage-corrected ROC-AUC: 0.667. PR-AUC: 0.322. "
            "Identifies top 20% highest-risk utilities across 50 US states."
        ),
    }