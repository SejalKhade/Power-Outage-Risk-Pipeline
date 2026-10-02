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

## Medallion Pipeline (Bronze -> Silver -> Gold)

`python -m src.medallion` lands the raw CSV and builds the analysis tables in three layers.
Engine: **DuckDB SQL** for Bronze/Silver and the Gold summary; Gold utility features reuse the
existing `src/features.py`. (Not Spark. 3.4M rows / 1.5 GB runs on one machine in about 90 seconds.)

| Layer | What it holds | Rules |
|---|---|---|
| **Bronze** | Raw CSV as-is, every column kept as text, plus `_batch_id`, `_ingested_at`, `_source_file`, `_row_id` | Append-only: a re-run writes a new batch, never overwrites one |
| **Silver** | Typed, validated, de-duplicated events (same cleaning rules as `src/preprocess.py`, in SQL) | Required columns must exist or the run fails; invalid rows go to a **quarantine** table with a reason code; coercions are counted in the manifest |
| **Gold** | `utility_features` (1 row per utility, from `src/features.py`) and `state_risk_summary` | Built only from Silver |

Every run writes `data/manifest/run_<batch>.json` (source file hash, counts per layer, rejects by
reason, coercion counts) and enforces `bronze_rows == quarantined + duplicates_removed + silver_rows`.

**Measured on the real 3.44M-row file**

| Check | Result |
|---|---|
| Bronze rows | 3,441,325 |
| Quarantined (invalid US state) | 103 |
| Exact duplicates removed | 937,077 |
| Silver rows | **2,504,145**, identical row count to the original `preprocess.py` output |
| Gold utility features | 1,677 rows x 46 columns; **no feature column differs** from the original output (tolerance 1e-9) |
| High-risk utilities | 336 of 1,677 (20%), unchanged |
| Silver rows differing from the original output | 207 (0.008%), in the 16th significant digit of one damage value, because pandas' default CSV parser rounds `4179999.9999999995` to `4180000.0` and Silver keeps the source value |

Tests (`tests/test_medallion.py`, 13): bronze lineage and append-only behaviour, row-count identity,
quarantine reasons, type and cleaning rules, fail-fast on a missing column, and parity with the
original pandas cleaner on a synthetic file.

```bash
python -m src.medallion --raw data/raw/merged_utility_storm_2024.csv
python -m src.medallion --raw ... --sync-legacy   # also copies gold features to data/processed/ for train.py and the dashboard
```

Known limits: runs single-machine; Silver is rebuilt in full from one Bronze batch (no incremental
merge across batches); the `negative_value` quarantine rule matched 0 rows on this dataset.

---

## Key Technical Contribution — Data Leakage Detection

Initial models returned ROC-AUC = 1.0 — a clear signal of leakage.

**Root cause:** Three engineered features (`estimated_annual_loss_usd`, `nerc_sla_breach_risk`, `sla_breach_margin_min`) were derived directly from SAIDI, which defines the target variable. The model was predicting risk from risk.

**Fix:** Removed all SAIDI-derived features from the feature set, retrained across all 28 experiments. ROC-AUC corrected from 1.0 → 0.668.

This is the difference between a model that looks good in development and one that would actually work in production.

---

## Tech Stack

**ML/Data:** Python · Scikit-learn · XGBoost · LightGBM · MLflow · Pandas · NumPy · DuckDB (SQL)
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
python -m src.medallion --sync-legacy   # Bronze -> Silver -> Gold (see above)
# ...or the original step-by-step stages:
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
│   ├── medallion.py         Bronze -> Silver -> Gold (DuckDB SQL)
│   ├── preprocess.py        Stage 1 — data pipeline (original pandas version)
│   ├── features.py          Stage 2 — feature engineering
│   ├── train.py             Stage 3 — ML training + MLflow
│   └── utils.py             Shared helpers
├── tests/                   test_utils.py, test_medallion.py
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
