**Architecture**

This document describes the main components of the AI Supply Chain Disruption Prediction System.

```mermaid
flowchart TD
    A[Supply Chain Data Sources] -->|raw files| B[Data Pipeline]
    B --> C[Feature Engineering]
    C --> D[ML Models \n(Logistic / RandomForest / XGBoost)]
    D --> E[Risk Scoring Engine]
    E --> F[FastAPI Prediction Service]
    F --> G[Streamlit Dashboard]
    style A fill:#f9f,stroke:#333,stroke-width:1px
    style G fill:#bbf,stroke:#333,stroke-width:1px
```

Components
- Supply Chain Data Sources: GSCPI, AIS, Port stats, NOAA weather.
- Data Pipeline: local ingestion, cleaning, and merge (see `src/data_pipeline/ingest.py`).
- Feature Engineering: transforms merged events into ML-ready features (`src/features/engineering.py`).
- ML Models: baseline classifiers trained in `src/models/train.py`.
- Risk Scoring Engine: converts model outputs to risk bands.
- FastAPI service: exposes `/predict` for dashboard and other clients (`api/main.py`).
