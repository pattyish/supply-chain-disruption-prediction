"""Train baseline models for delay risk.

Usage:
    python src/models/train.py --input data/processed/merged_supply_chain_data.csv --output models/delay_model.pkl
"""
import argparse
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def train_model(input_csv: Path, output_path: Path):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "delay_risk" not in df.columns:
        raise ValueError("Input CSV must contain 'delay_risk' target column")

    X = df.drop(columns=["delay_risk"]).select_dtypes(include=["number"]).fillna(0)
    y = df["delay_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Try XGBoost if available
    if XGBClassifier is not None:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, output_path)
    print("Model saved to", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_model(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
