"""
medallion.py
────────────
Bronze -> Silver -> Gold for the EIA-861 x NOAA Storm Events dataset.

ENGINE   : DuckDB (SQL) for Bronze and Silver and the Gold aggregates. Gold utility
           features reuse the existing pandas logic in src/features.py.
           (Not Spark: the data is 3.4M rows / 1.5 GB, which one machine handles.)

BRONZE   : the raw CSV landed as-is. Every source column kept as text, plus lineage
           (_batch_id, _ingested_at, _source_file, _row_id). Append-only: each run
           writes a new batch folder and never overwrites an old one.
SILVER   : typed, validated, de-duplicated. Same cleaning rules as src/preprocess.py,
           expressed in SQL, plus an explicit contract:
             - required columns must exist (run fails fast otherwise)
             - hard rules send a row to quarantine with a reason code
             - soft expectations are counted in the manifest, rows untouched
GOLD     : analysis-ready tables: utility_features (1 row per utility, from
           src/features.py) and state_risk_summary.

Every run writes data/manifest/run_<batch>.json and checks the row-count identity
    bronze_rows == quarantined + duplicates_removed + silver_rows
and aborts if it does not hold.

Usage:
    python -m src.medallion --raw data/raw/merged_utility_storm_2024.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.preprocess import KEEP_COLUMNS, NUMERIC_COLS, TEXT_COLS, VALID_US_STATES

logger = logging.getLogger(__name__)

DAMAGE_COLS = ["DAMAGE_PROPERTY_USD", "DAMAGE_CROPS_USD"]
FLOAT_COLS = list(NUMERIC_COLS) + DAMAGE_COLS          # coerced to DOUBLE, null/NaN -> 0.0
INT_COLS = ["Utility Number"]
NONNEGATIVE_COLS = [c for c in FLOAT_COLS
                    if c.startswith(("IEEE_", "INJURIES_", "DEATHS_", "DAMAGE_"))]
REJECT_REASONS = ("invalid_state", "negative_value")
DATE_FORMAT = "%d-%b-%y %H:%M:%S"                      # e.g. 01-APR-24 10:40:00


class ContractError(Exception):
    """Raised when the data violates the Bronze -> Silver contract."""


def _q(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def _lit(path: Path | str) -> str:
    return str(Path(path).as_posix()).replace("'", "''")


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════
# BRONZE
# ══════════════════════════════════════════════════════════════════

def build_bronze(con: duckdb.DuckDBPyConnection, raw_csv: Path, bronze_dir: Path,
                 batch_id: str, ingested_at: str, limit: int | None = None) -> dict:
    """Land the raw CSV as text Parquet with lineage columns. Never overwrites a batch."""
    batch_dir = bronze_dir / f"batch_{batch_id}"
    if batch_dir.exists():
        raise FileExistsError(f"Bronze batch already exists (append-only): {batch_dir}")
    batch_dir.mkdir(parents=True)
    target = batch_dir / "part-0.parquet"
    limit_sql = f" LIMIT {int(limit)}" if limit else ""

    con.execute(f"""
        COPY (
            SELECT *,
                   '{batch_id}'                    AS _batch_id,
                   TIMESTAMP '{ingested_at}'       AS _ingested_at,
                   '{raw_csv.name.replace("'", "''")}' AS _source_file,
                   row_number() OVER ()            AS _row_id
            FROM read_csv('{_lit(raw_csv)}', header = true, all_varchar = true,
                          ignore_errors = true, store_rejects = true){limit_sql}
        ) TO '{_lit(target)}' (FORMAT parquet, COMPRESSION zstd)
    """)
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{_lit(target)}')").fetchone()[0]
    try:
        rejected_lines = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
    except duckdb.Error:
        rejected_lines = 0
    return {"path": str(target), "rows": rows, "malformed_lines_skipped": rejected_lines}


# ══════════════════════════════════════════════════════════════════
# SILVER
# ══════════════════════════════════════════════════════════════════

def _typed_select(bronze_glob: str) -> str:
    """SQL that types and cleans every kept column (same rules as src/preprocess.py)."""
    cols = []
    for c in KEEP_COLUMNS:
        n = _q(c)
        if c in FLOAT_COLS:
            cast = f"TRY_CAST(trim({n}) AS DOUBLE)"
            cols.append(f"CASE WHEN {cast} IS NULL OR isnan({cast}) THEN 0.0 ELSE {cast} END AS {n}")
        elif c in INT_COLS:
            cols.append(f"TRY_CAST(TRY_CAST(trim({n}) AS DOUBLE) AS BIGINT) AS {n}")
        elif c in TEXT_COLS:
            cols.append(f"COALESCE(NULLIF(trim({n}), ''), 'Unknown') AS {n}")
        else:
            cols.append(f"{n}")
    cols.append(
        f"TRY_STRPTIME({_q('BEGIN_DATE_TIME')}, '{DATE_FORMAT}') AS event_ts")
    states = ", ".join(f"'{s}'" for s in sorted(VALID_US_STATES))
    negative = " OR ".join(
        f"TRY_CAST(trim({_q(c)}) AS DOUBLE) < 0" for c in NONNEGATIVE_COLS)
    reason = (f"CASE WHEN {_q('State')} IS NULL OR {_q('State')} NOT IN ({states}) THEN 'invalid_state' "
              f"WHEN {negative} THEN 'negative_value' END AS _reject_reason")
    return (f"SELECT _row_id, {', '.join(cols)}, {reason} "
            f"FROM read_parquet('{bronze_glob}')")


def _soft_expectations(con: duckdb.DuckDBPyConnection, bronze_glob: str) -> dict:
    """Counts that never change the data: values coerced to 0 and missing keys."""
    parts = []
    for c in FLOAT_COLS:
        n = _q(c)
        cast = f"TRY_CAST(trim({n}) AS DOUBLE)"
        parts.append(f"count(*) FILTER (WHERE {n} IS NULL OR trim({n}) = '' OR isnan({cast})) "
                     f"AS \"{c}__null_or_nan\"")
        parts.append(f"count(*) FILTER (WHERE trim({n}) <> '' AND {cast} IS NULL) "
                     f"AS \"{c}__unparseable\"")
    parts.append(f"count(*) FILTER (WHERE TRY_CAST(trim({_q('Utility Number')}) AS DOUBLE) IS NULL) "
                 f"AS utility_number_missing")
    row = con.execute(f"SELECT {', '.join(parts)} FROM read_parquet('{bronze_glob}')").df().iloc[0]
    out = {"utility_number_missing": int(row["utility_number_missing"]), "coerced_to_zero": {}}
    for c in FLOAT_COLS:
        out["coerced_to_zero"][c] = {"null_or_nan": int(row[f"{c}__null_or_nan"]),
                                     "unparseable": int(row[f"{c}__unparseable"])}
    return out


def build_silver(con: duckdb.DuckDBPyConnection, bronze_target: str, silver_dir: Path) -> dict:
    bronze_glob = _lit(bronze_target)

    have = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{bronze_glob}')").fetchall()}
    missing = [c for c in KEEP_COLUMNS if c not in have]
    if missing:
        raise ContractError(f"Bronze batch is missing required columns: {missing}")

    silver_dir.mkdir(parents=True, exist_ok=True)
    silver_path = silver_dir / "outage_events.parquet"
    quarantine_path = silver_dir / "quarantine.parquet"

    con.execute(f"CREATE OR REPLACE TEMP TABLE typed AS {_typed_select(bronze_glob)}")
    bronze_rows = con.execute("SELECT count(*) FROM typed").fetchone()[0]

    reasons = dict(con.execute(
        "SELECT _reject_reason, count(*) FROM typed WHERE _reject_reason IS NOT NULL GROUP BY 1").fetchall())
    quarantined = sum(reasons.values())

    silver_cols = ", ".join([_q(c) for c in KEEP_COLUMNS] + ["event_ts"])
    con.execute(f"""COPY (SELECT _row_id, _reject_reason, {silver_cols} FROM typed
                         WHERE _reject_reason IS NOT NULL)
                    TO '{_lit(quarantine_path)}' (FORMAT parquet, COMPRESSION zstd)""")
    after_filter = bronze_rows - quarantined
    con.execute(f"""COPY (SELECT DISTINCT {silver_cols} FROM typed WHERE _reject_reason IS NULL)
                    TO '{_lit(silver_path)}' (FORMAT parquet, COMPRESSION zstd)""")
    silver_rows = con.execute(f"SELECT count(*) FROM read_parquet('{_lit(silver_path)}')").fetchone()[0]
    duplicates_removed = after_filter - silver_rows
    con.execute("DROP TABLE typed")

    if bronze_rows != quarantined + duplicates_removed + silver_rows:
        raise ContractError(
            f"Row-count identity broken: bronze={bronze_rows} quarantined={quarantined} "
            f"duplicates={duplicates_removed} silver={silver_rows}")

    return {
        "path": str(silver_path), "quarantine_path": str(quarantine_path),
        "bronze_rows_in": bronze_rows, "quarantined": quarantined,
        "quarantine_by_reason": {r: int(reasons.get(r, 0)) for r in REJECT_REASONS},
        "duplicates_removed": duplicates_removed, "rows": silver_rows,
        "soft_expectations": _soft_expectations(con, bronze_glob),
    }


# ══════════════════════════════════════════════════════════════════
# GOLD
# ══════════════════════════════════════════════════════════════════

def build_gold(con: duckdb.DuckDBPyConnection, silver_path: str, gold_dir: Path) -> dict:
    from src.features import build_utility_features

    gold_dir.mkdir(parents=True, exist_ok=True)
    features_path = gold_dir / "utility_features.parquet"
    features = build_utility_features(input_path=silver_path, output_path=str(features_path))

    state_path = gold_dir / "state_risk_summary.parquet"
    risk_col = next((c for c in features.columns if c.lower() in ("high_risk", "risk_label", "is_high_risk")), None)
    state_col = "State" if "State" in features.columns else None
    if state_col and risk_col:
        con.register("gold_features", features)
        con.execute(f"""
            COPY (
                SELECT {_q(state_col)} AS state,
                       count(*)                                  AS utilities,
                       sum(CAST({_q(risk_col)} AS INTEGER))      AS high_risk_utilities,
                       round(100.0 * sum(CAST({_q(risk_col)} AS INTEGER)) / count(*), 1) AS high_risk_pct
                FROM gold_features GROUP BY 1 ORDER BY high_risk_utilities DESC, utilities DESC
            ) TO '{_lit(state_path)}' (FORMAT parquet, COMPRESSION zstd)""")
        con.unregister("gold_features")
        states = int(con.execute(f"SELECT count(*) FROM read_parquet('{_lit(state_path)}')").fetchone()[0])
    else:
        states = 0
    return {"utility_features_path": str(features_path), "utilities": int(len(features)),
            "feature_columns": int(features.shape[1]), "state_summary_path": str(state_path) if states else None,
            "states": states, "risk_column": risk_col}


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════

def new_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]


def write_manifest(manifest: dict, data_dir: str | Path) -> Path:
    """Stamp completion, record the identity check, and write data/manifest/run_<batch>.json."""
    manifest["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    manifest["identity_check"] = "bronze_rows == quarantined + duplicates_removed + silver_rows : PASS"
    mdir = Path(data_dir) / "manifest"
    mdir.mkdir(parents=True, exist_ok=True)
    path = mdir / f"run_{manifest['batch_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def run(raw_csv: str | Path, data_dir: str | Path = "data", batch_id: str | None = None,
        build_gold_layer: bool = True, sync_legacy: bool = False,
        limit: int | None = None) -> dict:
    raw_csv, data_dir = Path(raw_csv), Path(data_dir)
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_csv}")
    batch_id = batch_id or new_batch_id()
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"batch_id": batch_id, "started_at": ingested_at, "engine": f"duckdb {duckdb.__version__}",
                      "source": {"file": raw_csv.name, "bytes": raw_csv.stat().st_size}}
    con = duckdb.connect(str(data_dir / ".work.duckdb"))
    try:
        t = time.time()
        manifest["source"]["sha256"] = _sha256(raw_csv)
        logger.info("BRONZE  landing %s", raw_csv.name)
        manifest["bronze"] = build_bronze(con, raw_csv, data_dir / "bronze" / "outage_events",
                                          batch_id, ingested_at, limit)
        manifest["bronze"]["seconds"] = round(time.time() - t, 1)
        logger.info("BRONZE  %s rows (%s malformed lines skipped)",
                    f"{manifest['bronze']['rows']:,}", manifest["bronze"]["malformed_lines_skipped"])

        t = time.time()
        logger.info("SILVER  typing, validating, de-duplicating")
        manifest["silver"] = build_silver(con, manifest["bronze"]["path"], data_dir / "silver")
        manifest["silver"]["seconds"] = round(time.time() - t, 1)
        s = manifest["silver"]
        logger.info("SILVER  %s rows | quarantined %s %s | duplicates removed %s",
                    f"{s['rows']:,}", f"{s['quarantined']:,}", s["quarantine_by_reason"],
                    f"{s['duplicates_removed']:,}")

        if build_gold_layer:
            t = time.time()
            logger.info("GOLD    building utility features and state summary")
            manifest["gold"] = build_gold(con, s["path"], data_dir / "gold")
            manifest["gold"]["seconds"] = round(time.time() - t, 1)
            logger.info("GOLD    %s utilities x %s columns",
                        f"{manifest['gold']['utilities']:,}", manifest["gold"]["feature_columns"])
            if sync_legacy:
                legacy = data_dir / "processed"
                legacy.mkdir(parents=True, exist_ok=True)
                shutil.copy2(manifest["gold"]["utility_features_path"], legacy / "utility_features.parquet")
                manifest["gold"]["synced_to"] = str(legacy / "utility_features.parquet")
    finally:
        con.close()

    write_manifest(manifest, data_dir)
    return manifest


def _cli() -> None:
    # features.py / utils.py print box-drawing characters; make that safe on Windows consoles
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Bronze -> Silver -> Gold pipeline")
    ap.add_argument("--raw", default="data/raw/merged_utility_storm_2024.csv")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--skip-gold", action="store_true")
    ap.add_argument("--sync-legacy", action="store_true",
                    help="also copy gold utility_features to data/processed/ for train.py and the dashboard")
    ap.add_argument("--limit", type=int, default=None, help="dev only: read the first N rows")
    a = ap.parse_args()
    m = run(a.raw, a.data_dir, build_gold_layer=not a.skip_gold, sync_legacy=a.sync_legacy, limit=a.limit)
    print(json.dumps({k: m[k] for k in ("batch_id", "identity_check")}, indent=2))


if __name__ == "__main__":
    _cli()
