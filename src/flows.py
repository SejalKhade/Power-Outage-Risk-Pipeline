"""
flows.py
────────
Prefect orchestration of the Bronze -> Silver -> Gold pipeline (src/medallion.py),
with a data-quality gate and optional publishing to Amazon S3.

    land bronze -> build silver -> QUALITY GATE -> build gold -> write manifest -> publish to S3

What the orchestrator adds over `python -m src.medallion`:
  * each layer is its own task: per-step state, duration and logs in the Prefect UI
  * retries for transient failures (file reads, S3 uploads with back-off)
  * a quality gate that FAILS the run when too many rows are quarantined, before Gold is built
  * a schedule:  python -m src.flows --raw <csv> --serve "0 6 * * *"

Usage:
    python -m src.flows --raw data/raw/merged_utility_storm_2024.csv
    python -m src.flows --raw ... --s3-bucket my-bucket          # also publish to S3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from prefect import flow, get_run_logger, task

from src import medallion
from src.medallion import ContractError


class DataQualityError(RuntimeError):
    """Raised by the quality gate when a batch is too dirty to promote."""


def _connect(data_dir: str):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(Path(data_dir) / ".work.duckdb"))


@task(name="land-bronze", retries=2, retry_delay_seconds=10)
def land_bronze(raw_csv: str, data_dir: str, batch_id: str, ingested_at: str, limit: int | None) -> dict:
    con = _connect(data_dir)
    try:
        out = medallion.build_bronze(con, Path(raw_csv), Path(data_dir) / "bronze" / "outage_events",
                                     batch_id, ingested_at, limit)
    finally:
        con.close()
    get_run_logger().info("Bronze landed: %s rows", f"{out['rows']:,}")
    return out


@task(name="build-silver")   # contract violations are real findings: no retries
def build_silver_layer(bronze_path: str, data_dir: str) -> dict:
    con = _connect(data_dir)
    try:
        out = medallion.build_silver(con, bronze_path, Path(data_dir) / "silver")
    finally:
        con.close()
    get_run_logger().info("Silver: %s rows, %s quarantined, %s duplicates removed",
                          f"{out['rows']:,}", f"{out['quarantined']:,}", f"{out['duplicates_removed']:,}")
    return out


@task(name="quality-gate")
def quality_gate(silver: dict, max_quarantine_pct: float) -> float:
    pct = 100.0 * silver["quarantined"] / max(silver["bronze_rows_in"], 1)
    get_run_logger().info("Quarantine rate %.4f%% (limit %.2f%%)", pct, max_quarantine_pct)
    if pct > max_quarantine_pct:
        raise DataQualityError(
            f"{pct:.2f}% of rows quarantined ({silver['quarantine_by_reason']}); "
            f"limit is {max_quarantine_pct}%. Gold was not built.")
    return pct


@task(name="build-gold")
def build_gold_layer(silver_path: str, data_dir: str) -> dict:
    con = _connect(data_dir)
    try:
        return medallion.build_gold(con, silver_path, Path(data_dir) / "gold")
    finally:
        con.close()


@task(name="publish-s3", retries=3, retry_delay_seconds=[10, 30, 60])
def publish_s3(data_dir: str, bucket: str, batch_id: str, prefix: str, region: str | None) -> dict:
    from src import s3_publish
    out = s3_publish.publish(data_dir, bucket, batch_id, prefix, region)
    get_run_logger().info("Published %d objects to s3://%s/%s", len(out["objects"]), bucket, prefix)
    return {k: out[k] for k in ("bucket", "prefix", "batch_id", "objects")}


@flow(name="power-outage-medallion", log_prints=True)
def medallion_flow(raw_csv: str, data_dir: str = "data", max_quarantine_pct: float = 1.0,
                   build_gold: bool = True, s3_bucket: str | None = None, s3_prefix: str = "power-outage",
                   s3_region: str | None = None, limit: int | None = None) -> dict:
    """Run the full pipeline. Returns the run manifest (also written to data/manifest/)."""
    raw = Path(raw_csv)
    if not raw.exists():
        raise FileNotFoundError(f"Raw file not found: {raw}")
    batch_id = medallion.new_batch_id()
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    manifest = {"batch_id": batch_id, "started_at": ingested_at, "orchestrator": "prefect",
                "source": {"file": raw.name, "bytes": raw.stat().st_size, "sha256": medallion._sha256(raw)}}

    manifest["bronze"] = land_bronze(str(raw), data_dir, batch_id, ingested_at, limit)
    manifest["silver"] = build_silver_layer(manifest["bronze"]["path"], data_dir)
    manifest["quality_gate"] = {"quarantine_pct": quality_gate(manifest["silver"], max_quarantine_pct),
                                "limit_pct": max_quarantine_pct}
    if build_gold:
        manifest["gold"] = build_gold_layer(manifest["silver"]["path"], data_dir)
    medallion.write_manifest(manifest, data_dir)
    if s3_bucket:
        manifest["s3"] = publish_s3(data_dir, s3_bucket, batch_id, s3_prefix, s3_region)
        medallion.write_manifest(manifest, data_dir)     # record the S3 result in the manifest too
    return manifest


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Orchestrated Bronze -> Silver -> Gold pipeline")
    ap.add_argument("--raw", default="data/raw/merged_utility_storm_2024.csv")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-quarantine-pct", type=float, default=1.0)
    ap.add_argument("--skip-gold", action="store_true")
    ap.add_argument("--s3-bucket"); ap.add_argument("--s3-prefix", default="power-outage"); ap.add_argument("--s3-region")
    ap.add_argument("--limit", type=int, default=None, help="dev only: first N rows")
    ap.add_argument("--serve", metavar="CRON", help='serve on a schedule, e.g. "0 6 * * *"')
    a = ap.parse_args()
    params = dict(raw_csv=a.raw, data_dir=a.data_dir, max_quarantine_pct=a.max_quarantine_pct,
                  build_gold=not a.skip_gold, s3_bucket=a.s3_bucket, s3_prefix=a.s3_prefix,
                  s3_region=a.s3_region, limit=a.limit)
    if a.serve:
        medallion_flow.serve(name="power-outage-scheduled", cron=a.serve, parameters=params)
    else:
        m = medallion_flow(**params)
        print(json.dumps({k: m[k] for k in ("batch_id", "quality_gate", "identity_check") if k in m}, indent=2))


if __name__ == "__main__":
    main()
