# ⚡ Power Outage Risk Analysis Pipeline

Which US electric utilities are most at risk of poor reliability, and does storm exposure help predict it?
This project joins EIA-861 reliability data with NOAA Storm Events (2024), builds a 1,677-utility risk dataset,
trains and compares 28 models, and serves the result through a REST API and a live dashboard.

[![CI](https://github.com/SejalKhade/Power-Outage-Risk-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/SejalKhade/Power-Outage-Risk-Pipeline/actions/workflows/ci.yml)
[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-HuggingFace-orange)](https://sejjjallll-power-outage-risk-dashboard.hf.space)

**🔗 Live Dashboard →** [sejjjallll-power-outage-risk-dashboard.hf.space](https://huggingface.co/spaces/Sejjjallll/power-outage-risk-dashboard)

---

## What this project does

Utilities report how reliable their service is: **SAIDI** (minutes a customer is without power per year) and
**SAIFI** (how often outages happen). NOAA separately logs severe-weather events. Step by step:

1. **Data.** EIA-861 reliability and service-territory data merged with the 2024 NOAA Storm Events, giving
   3.44M utility x storm-event rows (`merged_utility_storm_2024.csv`, 1.5 GB, not stored in the repo).
2. **Clean.** Rows outside the 50 US states are dropped, numeric fields are coerced, exact duplicates removed.
   This can be run as the original pandas stage (`src/preprocess.py`) or as the Bronze/Silver/Gold pipeline
   below (`src/medallion.py`), which produces the same result and adds quarantine, lineage and a run manifest.
3. **Features.** Everything is aggregated to **one row per utility** (1,677 utilities, 46 columns): storm counts,
   damage and magnitude by event type, grid structure (NERC region, ownership, counties served) and
   SAIDI/SAIFI percentile ranks.
4. **Label.** A utility is **High Risk** if it is in the top 20% by the average of its SAIDI and SAIFI percentile
   ranks (336 of 1,677).
5. **Model.** 28 tracked experiments (7 classifiers x 2 feature sets x 2 thresholds), selecting the best by PR-AUC.
   Using only utility attributes, no model beats chance (ROC-AUC 0.50); adding storm features lifts the best
   model to 0.77. The selected model (Logistic Regression + weather features) scores ROC-AUC 0.667, PR-AUC 0.322.
6. **Serve.** A FastAPI service (`/predict`, `/health`, `/model-info`) returns a risk label and probability for a
   utility, and a Gradio + Folium dashboard on Hugging Face Spaces lets you filter by state, ownership and NERC
   region, view risk on maps, and compare utilities.

Scope: one year of data (2024), a single train/test evaluation per experiment, and a risk label derived from
SAIDI/SAIFI, so this ranks relative reliability risk; it does not forecast individual outages.

---

## Project Outcomes

| Metric | Value |
|---|---|
| Records processed | **3,441,325** raw rows → 3,441,222 in the 50 states → 2,504,145 after de-duplication → 1,677 utilities |
| Features engineered | **46** |
| Utilities classified High Risk | **336 of 1,677 (20%)**; utilities span all 50 states |
| MLflow experiments | **28** (7 classifiers × 2 feature sets × 2 thresholds) |
| Best model | Logistic Regression + Weather features (selected by PR-AUC) |
| ROC-AUC | **0.667** (leakage-corrected from 1.0) |
| PR-AUC | **0.322** |
| Weather feature uplift | **+54% ROC-AUC** (0.50 utility-only → 0.77 best weather model) |
| Estimated economic loss | **$74.5B** annually (a rough model: SAIDI hours x a customer proxy of 50,000 per county x a DOE cost per customer-hour; not a measured loss) |

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
| Quarantined (not one of the 50 states) | 103 (84 `DC`, 18 `CN`, 1 blank) |
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

Known limits: `DC` is quarantined because the original filter lists only the 50 states; runs single-machine; Silver is rebuilt in full from one Bronze batch (no incremental
merge across batches); the `negative_value` quarantine rule matched 0 rows on this dataset.

---

## Key Technical Contribution — Data Leakage Detection

Initial models returned ROC-AUC = 1.0 — a clear signal of leakage.

**Root cause:** Three engineered features (`estimated_annual_loss_usd`, `nerc_sla_breach_risk`, `sla_breach_margin_min`) were derived directly from SAIDI, which defines the target variable. The model was predicting risk from risk.

**Fix:** Removed all SAIDI-derived features from the feature set, retrained across all 28 experiments. ROC-AUC corrected from 1.0 → 0.667.

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
git clone https://github.com/SejalKhade/Power-Outage-Risk-Pipeline.git
cd Power-Outage-Risk-Pipeline
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
Power-Outage-Risk-Pipeline/
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
