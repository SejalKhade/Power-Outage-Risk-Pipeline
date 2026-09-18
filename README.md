# ⚡ Power Outage Risk Analysis Pipeline

End-to-end ML pipeline identifying high-risk electric utilities across the United States using EIA-861 reliability data and NOAA Storm Events data.

[![CI](https://github.com/SejalKhade/Power-Outage-Risk-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/SejalKhade/Power-Outage-Risk-Dashboard/actions/workflows/ci.yml)
[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-HuggingFace-orange)](https://sejjjallll-power-outage-risk-dashboard.hf.space)

**🔗 Live Dashboard →** [sejjjallll-power-outage-risk-dashboard.hf.space](https://huggingface.co/spaces/Sejjjallll/power-outage-risk-dashboard)

---

## Project Outcomes

| Metric | Value |
|---|---|
| Records processed | **3,441,222** raw → 1,677 utility-level |
| Features engineered | **46** |
| Utilities classified High Risk | **336 of 1,677 (20%)** across 50 states |
| MLflow experiments | **28** (7 classifiers × 2 feature sets × 2 thresholds) |
| Best model | Logistic Regression + Weather features |
| ROC-AUC | **0.668** (leakage-corrected from 1.0) |
| PR-AUC | **0.325** |
| Weather feature uplift | **+54% ROC-AUC** (0.50 → 0.77) |
| Estimated economic loss | **$74.5B** annually |

---

## Screenshots

**Live Dashboard (Gradio + Folium on Hugging Face Spaces)**
![Dashboard](docs/images/dashboard.png)

**FastAPI Endpoint (Swagger UI)**
![FastAPI Swagger](docs/images/api-swagger.png)

**FastAPI Prediction Response**
![FastAPI Predict](docs/images/api-predict.png)

**MLflow — 28 Tracked Experiments**
![MLflow](docs/images/mlflow.png)

---

## Architecture

```
Stage 1 — src/preprocess.py    Raw CSV (3.44M rows) -> Clean Parquet
Stage 2 — src/features.py      Clean Parquet -> 1,677 utility-level features
Stage 3 — src/train.py         28 experiments -> best_model.pkl + MLflow logs
Stage 4 — api/main.py          FastAPI REST endpoint with /predict /health /docs
Dashboard — dashboard/app_gradio.py    Gradio + Folium maps deployed to HF Spaces
```

---

## Key Technical Contribution — Data Leakage Detection

Initial models returned ROC-AUC = 1.0 — a clear signal of leakage.

**Root cause:** Three engineered features (`estimated_annual_loss_usd`, `nerc_sla_breach_risk`, `sla_breach_margin_min`) were derived directly from SAIDI, which defines the target variable. The model was predicting risk from risk.

**Fix:** Removed all SAIDI-derived features from the feature set, retrained across all 28 experiments. ROC-AUC corrected from 1.0 → 0.668.

This is the difference between a model that looks good in development and one that would actually work in production.

---

## Tech Stack

**ML/Data:** Python · Scikit-learn · XGBoost · LightGBM · MLflow · Pandas · NumPy
**API:** FastAPI · Pydantic · Uvicorn
**Dashboard:** Gradio · Folium · Plotly
**DevOps:** Docker · GitHub Actions (CI/CD) · Hugging Face Spaces

---

## Run Locally

```bash
git clone https://github.com/SejalKhade/Power-Outage-Risk-Dashboard.git
cd Power-Outage-Risk-Dashboard
pip install -r requirements.txt

# Run pipeline stages
python -m src.preprocess     # 3.44M rows -> clean parquet
python -m src.features       # -> 1,677 utility features
python -m src.train          # 28 MLflow experiments

# Run API
uvicorn api.main:app --reload                 # http://localhost:8000/docs

# Run dashboard locally
python dashboard/app_gradio.py                # http://localhost:7860

# Or run containerized API
docker build -t power-outage-risk .
docker run -p 8000:8000 power-outage-risk
```

---

## Repository Structure

```
power-outage-risk-dashboard/
├── src/
│   ├── preprocess.py        Stage 1 — data pipeline
│   ├── features.py          Stage 2 — feature engineering
│   ├── train.py             Stage 3 — ML training + MLflow
│   └── utils.py             Shared helpers
├── api/
│   └── main.py              FastAPI REST endpoint
├── dashboard/
│   └── app_gradio.py        Gradio dashboard (HF Spaces)
├── outputs/models/          best_model.pkl, sensitivity_results.csv, metrics.json
├── docs/images/             Screenshots
├── .github/workflows/ci.yml GitHub Actions CI pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Author

**Sejal Khade**
MS Data Science · University of Texas at Arlington · May 2026

[GitHub](https://github.com/SejalKhade) · [LinkedIn](https://linkedin.com/in/sejallk) · [Live Dashboard](https://sejjjallll-power-outage-risk-dashboard.hf.space)
