from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["shipping_pressure", "port_wait_time", "weather_risk", "distance_km"]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "processed" / "synthetic_shipments.csv"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run `python data/generate_synthetic_data.py` first."
        )

    df = pd.read_csv(data_path)
    df = df.copy()
    df["distance_km"] = df["distance"] / 1000.0

    X = df[FEATURES]
    y = df["delayed"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1200, random_state=42)),
        ]
    )
    model.fit(X, y)

    bundle = {
        "model": model,
        "meta": {
            "name": "delay_logistic_baseline",
            "version": "v1.0.0",
            "features": FEATURES,
            "target": "delayed",
            "source": str(data_path),
        },
    }

    output_path = model_dir / "delay_model.pkl"
    joblib.dump(bundle, output_path)
    print(f"Wrote model artifact: {output_path}")


if __name__ == "__main__":
    main()
