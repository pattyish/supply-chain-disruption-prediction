"""
Simple ingestion helpers for local CSV/XLSX raw files.
These are lightweight utilities that read the raw dataset files from `data/raw/`
and write a simple merged CSV to `data/processed/merged_supply_chain_data.csv`.

Note: This module does not download remote data. Place raw files in `data/raw/`.
"""
from pathlib import Path
import pandas as pd


RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_gscpi(path: Path = RAW / "gscpi_data.xlsx") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GSCPI file not found at {path}")
    return pd.read_excel(path)


def load_ais(path: Path = RAW / "ais_shipping.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AIS file not found at {path}")
    return pd.read_csv(path)


def load_port_stats(path: Path = RAW / "port_container_stats.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Port stats file not found at {path}")
    return pd.read_csv(path)


def simple_merge(output: Path = PROCESSED / "merged_supply_chain_data.csv") -> Path:
    """Perform a basic merge of available datasets into a single CSV.

    This is a placeholder merge: users should replace with domain-specific joins.
    """
    parts = {}
    try:
        parts["gscpi"] = load_gscpi()
    except FileNotFoundError:
        parts["gscpi"] = pd.DataFrame()
    try:
        parts["ais"] = load_ais()
    except FileNotFoundError:
        parts["ais"] = pd.DataFrame()
    try:
        parts["port"] = load_port_stats()
    except FileNotFoundError:
        parts["port"] = pd.DataFrame()

    # Simple strategy: join on closest date/month where possible; otherwise create placeholder rows
    if not parts["ais"].empty:
        df = parts["ais"].copy()
        df["shipping_pressure_index"] = parts.get("gscpi", pd.DataFrame()).get("gscpi", pd.Series([None]*len(df))).values
        df["port_wait_time"] = parts.get("port", pd.DataFrame()).get("vessel_wait_time", pd.Series([None]*len(df))).values
        df["delay_minutes"] = 0
    else:
        # fallback: create empty dataframe with recommended columns
        df = pd.DataFrame(columns=["shipment_id","origin_port","destination","distance_km","shipping_pressure_index","port_wait_time","weather_risk","traffic_level","delay_minutes","delay_risk"]) 

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    out = simple_merge()
    print("Wrote merged file to", out)
