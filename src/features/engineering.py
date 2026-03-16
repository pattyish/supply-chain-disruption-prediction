"""Feature engineering helpers.

Provide functions that accept the merged CSV and produce features suitable for ML.
"""
import pandas as pd
from pathlib import Path


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example feature: normalize distance
    if "distance_km" in df.columns:
        df["distance_km_norm"] = df["distance_km"] / 1000.0
    else:
        df["distance_km_norm"] = 0.0

    # ensure numeric columns exist
    for col in ["shipping_pressure_index", "port_wait_time", "weather_risk"]:
        if col not in df.columns:
            df[col] = 0.0

    # Create a binary target if not present
    if "delay_risk" not in df.columns:
        df["delay_risk"] = (df.get("delay_minutes", 0) > 30).astype(int)

    # Select common ML columns
    features = ["shipping_pressure_index", "port_wait_time", "weather_risk", "distance_km_norm"]
    return df[features + ["delay_risk"]]


def save_features(input_csv: Path, output_csv: Path):
    df = pd.read_csv(input_csv)
    feat = basic_features(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(output_csv, index=False)
    return output_csv
