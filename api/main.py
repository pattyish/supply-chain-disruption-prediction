from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Supply Chain Disruption Predictor")


class PredictRequest(BaseModel):
    shipping_pressure: float
    port_wait_time: float
    weather_risk: float
    distance: float


MODEL_PATH = Path("models/delay_model.pkl")
_model_bundle = None


def load_model():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model artifact not found; run training first.")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


@app.post("/predict")
def predict_delay(req: PredictRequest):
    try:
        bundle = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    model = bundle["model"]
    features = np.array([req.shipping_pressure, req.port_wait_time, req.weather_risk, req.distance/1000.0]).reshape(1, -1)
    try:
        prob = float(model.predict_proba(features)[0][1])
    except Exception:
        prob = float(model.predict(features)[0])

    # Risk bands
    if prob < 0.3:
        risk = "LOW"
    elif prob < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {"delay_probability": prob, "risk_level": risk}
