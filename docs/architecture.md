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

Overview
- Supply chain sources: GSCPI, AIS, port container stats, weather and other operational indicators.
- Pipeline: ingestion, cleaning, simple merge, feature engineering, model training, serving, and dashboarding.

Folder structure & important files
- [README.md](README.md): project overview and quick start.
- [requirements.txt](requirements.txt): Python dependencies.
- [api/](api/) — prediction API (FastAPI): [api/main.py](api/main.py).
- [dashboard/](dashboard/) — Streamlit dashboard: [dashboard/app.py](dashboard/app.py).
- [notebooks/](notebooks/) — analysis and experiments: [notebooks/exploratory_analysis.ipynb](notebooks/exploratory_analysis.ipynb).
- [src/data_pipeline/](src/data_pipeline/) — ingestion and merging helpers: [src/data_pipeline/ingest.py](src/data_pipeline/ingest.py).
- [src/features/](src/features/) — feature engineering utilities: [src/features/engineering.py](src/features/engineering.py).
- [src/models/](src/models/) — model training and evaluation: [src/models/train.py](src/models/train.py).
- [data/](data/) — dataset folders (git-kept): [data/raw](data/raw) and [data/processed](data/processed).
- [docs/](docs/) — documentation (this file).

How to use / notes
- Environment: create a virtual environment and install `requirements.txt` (see project root for commands).
- Data: place raw files in `data/raw/` (the ingestion utilities read from there). Processed outputs are written to `data/processed/` by `src/data_pipeline/ingest.py`.
- Quick ingestion: run `python -m src.data_pipeline.ingest` or run the module directly to produce `data/processed/merged_supply_chain_data.csv`.
- Notebook: open [notebooks/exploratory_analysis.ipynb](notebooks/exploratory_analysis.ipynb) for a seeded EDA workflow that loads the first CSV it finds under the repo.

Recommendations
- Keep sample or production datasets outside the repo; store pointers or small samples under `data/raw/` and document them in `README.md`.
- Add a `CONTRIBUTING.md` or small developer README with how to run the API and dashboard locally (venv, uvicorn, streamlit commands).

If you want, I can also add a short `docs/dev-setup.md` with exact commands for Windows/macOS/Linux.
