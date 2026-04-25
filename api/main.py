from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Supply Chain Disruption Predictor")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "delay_model.pkl"
ROUTES_PATH = ROOT / "data" / "processed" / "synthetic_routes.csv"
SHIPMENTS_PATH = ROOT / "data" / "processed" / "synthetic_shipments.csv"
PREDICTION_LOG_PATH = ROOT / "data" / "processed" / "prediction_log.jsonl"
FEEDBACK_PATH = ROOT / "data" / "processed" / "feedback_labels.csv"

_model_bundle: dict[str, Any] | None = None
_routes_df: pd.DataFrame | None = None
_stats: dict[str, dict[str, float]] | None = None


class PredictRequest(BaseModel):
    shipping_pressure: float = Field(..., ge=0.0)
    port_wait_time: float = Field(..., ge=0.0)
    weather_risk: float = Field(..., ge=0.0, le=2.0)
    distance: float = Field(..., gt=0.0)


class ImpactRequest(PredictRequest):
    cost_per_delay_hour_usd: float = Field(50.0, ge=0.0)
    sla_penalty_usd: float = Field(500.0, ge=0.0)


class OptimizeRouteRequest(ImpactRequest):
    origin: str
    destination: str
    budget_usd: float | None = Field(default=None, ge=0.0)


class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_delayed: int = Field(..., ge=0, le=1)
    actual_delay_hours: float = Field(0.0, ge=0.0)
    notes: str | None = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_model() -> dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. Run `python src/models/train_minimal.py` first."
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def load_routes() -> pd.DataFrame:
    global _routes_df
    if _routes_df is None:
        if not ROUTES_PATH.exists():
            raise FileNotFoundError(
                f"Route file not found at {ROUTES_PATH}. Run `python data/generate_synthetic_data.py` first."
            )
        _routes_df = pd.read_csv(ROUTES_PATH)
    return _routes_df


def load_stats() -> dict[str, dict[str, float]]:
    global _stats
    if _stats is None:
        if not SHIPMENTS_PATH.exists():
            raise FileNotFoundError(
                f"Synthetic shipment stats missing at {SHIPMENTS_PATH}. Run `python data/generate_synthetic_data.py` first."
            )
        df = pd.read_csv(SHIPMENTS_PATH)
        stats: dict[str, dict[str, float]] = {}
        for col in ["shipping_pressure", "port_wait_time", "weather_risk", "distance"]:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(max(df[col].std(ddof=0), 1e-6)),
            }
        _stats = stats
    return _stats


def _risk_band(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    if prob < 0.6:
        return "MEDIUM"
    return "HIGH"


def _predict_probability(req: PredictRequest) -> float:
    bundle = load_model()
    model = bundle["model"]
    features = pd.DataFrame(
        [
            {
                "shipping_pressure": req.shipping_pressure,
                "port_wait_time": req.port_wait_time,
                "weather_risk": req.weather_risk,
                "distance_km": req.distance / 1000.0,
            }
        ]
    )
    try:
        prob = float(model.predict_proba(features)[0][1])
    except Exception:
        prob = float(model.predict(features)[0])
    return float(np.clip(prob, 0.0, 1.0))


def _expected_delay_hours(req: PredictRequest, prob: float) -> float:
    base_delay = max(req.port_wait_time * 0.16 + req.weather_risk * 2.5 + req.distance / 1500.0, 0.0)
    return float(base_delay * prob)


def _economic_impact(req: ImpactRequest, prob: float) -> dict[str, float]:
    expected_delay_hours = _expected_delay_hours(req, prob)
    delay_cost = expected_delay_hours * req.cost_per_delay_hour_usd
    expected_penalty = req.sla_penalty_usd * prob
    return {
        "expected_delay_hours": round(expected_delay_hours, 3),
        "expected_delay_cost_usd": round(delay_cost, 2),
        "expected_sla_penalty_usd": round(expected_penalty, 2),
        "expected_total_impact_usd": round(delay_cost + expected_penalty, 2),
    }


def _log_prediction(payload: dict[str, Any]) -> None:
    _ensure_parent(PREDICTION_LOG_PATH)
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _playbook_actions(root_causes: list[str], risk_level: str) -> list[str]:
    actions: list[str] = []
    if any("port_wait_time" in c for c in root_causes):
        actions.append("Request slot reassignment and expedite dock coordination.")
    if any("shipping_pressure" in c for c in root_causes):
        actions.append("Pre-book backup carriers and rebalance lane allocation.")
    if any("weather_risk" in c for c in root_causes):
        actions.append("Switch vulnerable legs to safer hubs and increase buffer time.")
    if any("distance" in c for c in root_causes):
        actions.append("Evaluate shorter alternate lane and split-load strategy.")
    if risk_level == "HIGH":
        actions.append("Escalate to operations incident queue with hourly monitoring.")
    return actions[:4]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Supply Chain Disruption Predictor",
        "capabilities": [
            "delay_probability",
            "economic_impact_scoring",
            "prescriptive_reroute_optimization",
            "anomaly_root_cause_and_playbooks",
            "provenance_and_active_learning",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model_ok = MODEL_PATH.exists()
    routes_ok = ROUTES_PATH.exists()
    shipments_ok = SHIPMENTS_PATH.exists()
    return {
        "status": "ok" if model_ok else "degraded",
        "model_ready": model_ok,
        "synthetic_routes_ready": routes_ok,
        "synthetic_shipments_ready": shipments_ok,
    }


@app.post("/predict")
def predict_delay(req: PredictRequest) -> dict[str, Any]:
    try:
        prob = _predict_probability(req)
        bundle = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    risk = _risk_band(prob)
    prediction_id = str(uuid4())
    model_meta = bundle.get("meta", {})
    result = {
        "prediction_id": prediction_id,
        "delay_probability": round(prob, 6),
        "risk_level": risk,
        "model_name": model_meta.get("name", "unknown"),
        "model_version": model_meta.get("version", "unknown"),
    }
    _log_prediction(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "type": "predict",
            "request": req.model_dump(),
            "response": result,
        }
    )
    return result


@app.post("/impact")
def impact(req: ImpactRequest) -> dict[str, Any]:
    try:
        prob = _predict_probability(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    impact_data = _economic_impact(req, prob)
    return {
        "delay_probability": round(prob, 6),
        "risk_level": _risk_band(prob),
        **impact_data,
    }


@app.post("/optimize/reroute")
def optimize_reroute(req: OptimizeRouteRequest) -> dict[str, Any]:
    try:
        routes = load_routes()
        base_prob = _predict_probability(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    lane = routes[
        (routes["origin"].str.lower() == req.origin.lower())
        & (routes["destination"].str.lower() == req.destination.lower())
    ].copy()

    if lane.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No synthetic routes found for lane {req.origin} -> {req.destination}.",
        )

    candidates = []
    for _, row in lane.iterrows():
        route_distance = float(row["distance_km"])
        route_reliability = float(row["reliability"])
        route_wait = max(req.port_wait_time * (1.06 - route_reliability), 0.1)
        route_request = PredictRequest(
            shipping_pressure=req.shipping_pressure * (1.02 - route_reliability),
            port_wait_time=route_wait,
            weather_risk=req.weather_risk,
            distance=route_distance,
        )
        route_prob = _predict_probability(route_request)
        route_impact_req = ImpactRequest(
            **route_request.model_dump(),
            cost_per_delay_hour_usd=req.cost_per_delay_hour_usd,
            sla_penalty_usd=req.sla_penalty_usd,
        )
        impact_data = _economic_impact(route_impact_req, route_prob)
        total_cost = float(row["base_cost_usd"]) + impact_data["expected_total_impact_usd"]

        candidates.append(
            {
                "route_id": row["route_id"],
                "origin": row["origin"],
                "destination": row["destination"],
                "distance_km": route_distance,
                "transit_hours": float(row["transit_hours"]),
                "reliability": route_reliability,
                "base_cost_usd": float(row["base_cost_usd"]),
                "delay_probability": round(route_prob, 6),
                "risk_level": _risk_band(route_prob),
                "expected_total_impact_usd": impact_data["expected_total_impact_usd"],
                "optimized_total_cost_usd": round(total_cost, 2),
            }
        )

    if req.budget_usd is not None:
        candidates = [c for c in candidates if c["optimized_total_cost_usd"] <= req.budget_usd]

    if not candidates:
        raise HTTPException(status_code=422, detail="No route candidates satisfy the provided budget.")

    ranked = sorted(candidates, key=lambda x: (x["optimized_total_cost_usd"], x["delay_probability"]))
    return {
        "baseline_delay_probability": round(base_prob, 6),
        "baseline_risk_level": _risk_band(base_prob),
        "recommended_route": ranked[0],
        "alternatives": ranked,
    }


@app.post("/analyze/anomaly")
def analyze_anomaly(req: PredictRequest) -> dict[str, Any]:
    try:
        stats = load_stats()
        prob = _predict_probability(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    values = req.model_dump()
    zscores: dict[str, float] = {}
    for k, val in values.items():
        mean = stats[k]["mean"]
        std = stats[k]["std"]
        zscores[k] = round((float(val) - mean) / std, 3)

    root_causes = [
        f"{k} out-of-pattern (z={v})"
        for k, v in sorted(zscores.items(), key=lambda i: abs(i[1]), reverse=True)
        if abs(v) >= 1.0
    ][:3]
    anomaly_score = float(np.mean([abs(v) for v in zscores.values()]))
    is_anomaly = anomaly_score >= 1.1 or prob >= 0.65
    risk = _risk_band(prob)

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 4),
        "delay_probability": round(prob, 6),
        "risk_level": risk,
        "root_causes": root_causes,
        "playbook_actions": _playbook_actions(root_causes, risk),
        "feature_zscores": zscores,
    }


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict[str, Any]:
    _ensure_parent(FEEDBACK_PATH)
    file_exists = FEEDBACK_PATH.exists()

    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "prediction_id",
                "actual_delayed",
                "actual_delay_hours",
                "notes",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_id": req.prediction_id,
                "actual_delayed": req.actual_delayed,
                "actual_delay_hours": req.actual_delay_hours,
                "notes": req.notes or "",
            }
        )

    return {"status": "saved", "message": "Feedback stored for active-learning loop."}


@app.get("/provenance")
def provenance() -> dict[str, Any]:
    bundle = load_model()
    model_meta = bundle.get("meta", {})

    prediction_count = 0
    if PREDICTION_LOG_PATH.exists():
        with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as f:
            prediction_count = sum(1 for _ in f)

    feedback_count = 0
    positive_labels = 0
    if FEEDBACK_PATH.exists():
        df = pd.read_csv(FEEDBACK_PATH)
        feedback_count = int(len(df))
        if "actual_delayed" in df:
            positive_labels = int(df["actual_delayed"].sum())

    return {
        "model": {
            "name": model_meta.get("name", "unknown"),
            "version": model_meta.get("version", "unknown"),
            "features": model_meta.get("features", []),
            "target": model_meta.get("target", "unknown"),
            "source": model_meta.get("source", "unknown"),
        },
        "active_learning": {
            "prediction_log_count": prediction_count,
            "feedback_count": feedback_count,
            "positive_feedback_labels": positive_labels,
            "feedback_rate": round(feedback_count / max(prediction_count, 1), 4),
        },
    }
