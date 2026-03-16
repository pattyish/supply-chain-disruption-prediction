AI System for Predictive Monitoring of Supply Chain Disruptions
===============================================================

Why this project
-----------------
Supply chains are vital to the United States economy. Congestion at ports, slowdowns in vessel movement, shortages of equipment, and extreme weather produce cascading delays that impact manufacturing, retail, and consumers nationwide. Predicting disruptions ahead of time helps logistics teams reroute shipments, allocate resources, and reduce costly delays.

Project objective
-----------------
Develop an AI system that predicts supply chain disruptions before they occur by analyzing shipping activity (AIS), port congestion data, the Global Supply Chain Pressure Index (GSCPI), and weather indicators. The system outputs a delay probability, a risk band, and suggested operational actions.

Quick example output
--------------------
Shipment ID: SHP2031
Route: Los Angeles → Chicago

Delay Probability: 71%
Risk Level: HIGH
Suggested Action: Reroute through Dallas hub

Repository layout
-----------------
- data/
  - raw/                # place downloaded raw datasets here
  - processed/          # processed datasets created by pipeline
- notebooks/
  - exploratory_analysis.ipynb
- src/
  - data_pipeline/      # ingestion scripts
  - features/           # feature engineering
  - models/             # training scripts
- api/
  - main.py             # FastAPI prediction service
- dashboard/
  - app.py              # Streamlit dashboard
- models/
  - delay_model.pkl     # trained model artifact (output)
- docs/
  - architecture.md     # architecture diagram and notes

Datasets to download and place in `data/raw/`
---------------------------------------------
1. Global Supply Chain Pressure Index (GSCPI) — New York Fed
   https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx
   Save as: `data/raw/gscpi_data.xlsx`

2. AIS Maritime Shipping Data
   https://marinecadastre.gov/ais/
   Save as: `data/raw/ais_shipping.csv`

3. Port Congestion Data (e.g., Port of Los Angeles statistics)
   https://www.portoflosangeles.org/business/statistics/container-statistics
   Save as: `data/raw/port_container_stats.csv`

4. NOAA Weather Data (optional)
   https://www.ncdc.noaa.gov/cdo-web/datasets
   Save as: `data/raw/weather_data.csv`

Getting started
---------------
1. Create and activate a virtual environment:

   python -m venv .venv
   .venv\Scripts\activate

2. Install dependencies:

   pip install -r requirements.txt

3. Place raw datasets into `data/raw/`.

4. Run the exploratory notebook:

   jupyter notebook notebooks/exploratory_analysis.ipynb

5. Create processed features and train a model (example):

   python src/data_pipeline/ingest.py
   python -m src.features.engineering
   python src/models/train.py --input data/processed/merged_supply_chain_data.csv --output models/delay_model.pkl

6. Start the API and dashboard:

   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   streamlit run dashboard/app.py

Architecture
------------
See [docs/architecture.md](docs/architecture.md) for the system diagram and component responsibilities.

Next steps and opportunities
----------------------------
- Add automated ingestion (streaming AIS), scheduling, and MLOps (CI/CD for models).
- Improve feature engineering with spatial joins, ETA predictions, and weather-model coupling.
- Add monitoring and alerts for model drift and data quality.
# logistics-network-optimization
