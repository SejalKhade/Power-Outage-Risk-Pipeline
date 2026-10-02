"""
s3_publish.py
─────────────
Publish the Bronze / Silver / Gold layers to Amazon S3 and describe them to Athena.

LAYOUT (one folder per table, one partition per run, Athena/Glue-friendly):

    s3://<bucket>/<prefix>/bronze/outage_events/batch_id=<id>/part-0.parquet      (raw text, as landed)
    s3://<bucket>/<prefix>/silver/outage_events/batch_id=<id>/outage_events.parquet
    s3://<bucket>/<prefix>/silver/quarantine/batch_id=<id>/quarantine.parquet
    s3://<bucket>/<prefix>/gold/utility_features/batch_id=<id>/utility_features.parquet
    s3://<bucket>/<prefix>/gold/state_risk_summary/batch_id=<id>/state_risk_summary.parquet
    s3://<bucket>/<prefix>/manifest/run_<id>.json

Silver and Gold files are re-written with snake_case column names before upload
("Utility Number" -> utility_number, "SPP.1" -> spp_1) because Athena columns cannot
contain spaces or dots unquoted. Bronze is uploaded untouched (it is the raw record).

Credentials come from the normal AWS chain (environment variables, ~/.aws, SSO, IAM role).
Nothing is read from or written to the repo.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import duckdb

ATHENA_TYPES = {
    "VARCHAR": "string", "DOUBLE": "double", "BIGINT": "bigint", "INTEGER": "int",
    "BOOLEAN": "boolean", "DATE": "date", "FLOAT": "float", "SMALLINT": "smallint",
}


def snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return s or "col"


def sanitized_copy(src: Path, dst: Path) -> list[tuple[str, str]]:
    """Write src Parquet to dst with snake_case, de-duplicated column names. Returns [(name, athena_type)]."""
    con = duckdb.connect()
    try:
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src.as_posix()}')").fetchall()
        seen: dict[str, int] = {}
        select, schema = [], []
        for name, dtype, *_ in cols:
            new = snake(name)
            if new in seen:
                seen[new] += 1
                new = f"{new}_{seen[new]}"
            else:
                seen[new] = 0
            base = dtype.split("(")[0].upper()
            atype = "timestamp" if base.startswith("TIMESTAMP") else ATHENA_TYPES.get(base, "string")
            select.append(f'"{name}" AS {new}')
            schema.append((new, atype))
        con.execute(f"COPY (SELECT {', '.join(select)} FROM read_parquet('{src.as_posix()}')) "
                    f"TO '{dst.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        return schema
    finally:
        con.close()


def _plan(data_dir: Path, batch_id: str) -> list[dict]:
    """What to upload: (local path, table folder, file name, sanitize?)."""
    d = Path(data_dir)
    items = [
        dict(local=next((d / "bronze" / "outage_events").glob(f"batch_{batch_id}/part-0.parquet"), None),
             layer="bronze", table="outage_events", name="part-0.parquet", sanitize=False),
        dict(local=d / "silver" / "outage_events.parquet", layer="silver", table="outage_events",
             name="outage_events.parquet", sanitize=True),
        dict(local=d / "silver" / "quarantine.parquet", layer="silver", table="quarantine",
             name="quarantine.parquet", sanitize=True),
        dict(local=d / "gold" / "utility_features.parquet", layer="gold", table="utility_features",
             name="utility_features.parquet", sanitize=True),
        dict(local=d / "gold" / "state_risk_summary.parquet", layer="gold", table="state_risk_summary",
             name="state_risk_summary.parquet", sanitize=True),
    ]
    return [i for i in items if i["local"] is not None and Path(i["local"]).exists()]


def _client(region: str | None = None):
    import boto3
    return boto3.client("s3", region_name=region) if region else boto3.client("s3")


def publish(data_dir: str | Path, bucket: str, batch_id: str, prefix: str = "power-outage",
            region: str | None = None, client=None) -> dict:
    """Upload every layer that exists for this batch. Verifies each object's size after upload."""
    s3 = client or _client(region)
    prefix = prefix.strip("/")
    uploaded, schemas = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for item in _plan(Path(data_dir), batch_id):
            src = Path(item["local"])
            if item["sanitize"]:
                clean = Path(tmp) / f"{item['layer']}_{item['table']}.parquet"
                schemas[f"{item['layer']}_{item['table']}"] = sanitized_copy(src, clean)
                src = clean
            key = f"{prefix}/{item['layer']}/{item['table']}/batch_id={batch_id}/{item['name']}"
            s3.upload_file(str(src), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size != src.stat().st_size:
                raise RuntimeError(f"Upload verification failed for {key}: {size} != {src.stat().st_size}")
            uploaded.append({"key": key, "bytes": size, "layer": item["layer"], "table": item["table"]})
        manifest = Path(data_dir) / "manifest" / f"run_{batch_id}.json"
        if manifest.exists():
            key = f"{prefix}/manifest/run_{batch_id}.json"
            s3.upload_file(str(manifest), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
            uploaded.append({"key": key, "bytes": manifest.stat().st_size, "layer": "manifest", "table": "manifest"})
    return {"bucket": bucket, "prefix": prefix, "batch_id": batch_id, "objects": uploaded, "schemas": schemas}


# ── Athena ─────────────────────────────────────────────────────────────────

def athena_ddl(result: dict, database: str) -> list[str]:
    """CREATE DATABASE + one external table per Silver/Gold table, partitioned by batch_id."""
    stmts = [f"CREATE DATABASE IF NOT EXISTS {database}"]
    for name, cols in result["schemas"].items():
        layer, table = name.split("_", 1)
        loc = f"s3://{result['bucket']}/{result['prefix']}/{layer}/{table}/"
        col_sql = ",\n  ".join(f"`{c}` {t}" for c, t in cols)
        stmts.append(
            f"CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{layer}_{table} (\n  {col_sql}\n)\n"
            f"PARTITIONED BY (batch_id string)\nSTORED AS PARQUET\nLOCATION '{loc}'\n"
            f"TBLPROPERTIES ('parquet.compression'='ZSTD')")
        stmts.append(
            f"ALTER TABLE {database}.{layer}_{table} ADD IF NOT EXISTS "
            f"PARTITION (batch_id='{result['batch_id']}') LOCATION '{loc}batch_id={result['batch_id']}/'")
    return stmts


def athena_query(client, sql: str, output_location: str, database: str | None = None,
                 workgroup: str = "primary", timeout_s: int = 180):
    """Run one Athena statement and return the rows (header row first). Raises on failure/timeout."""
    import time
    kwargs = dict(QueryString=sql, ResultConfiguration={"OutputLocation": output_location}, WorkGroup=workgroup)
    if database:
        kwargs["QueryExecutionContext"] = {"Database": database}
    qid = client.start_query_execution(**kwargs)["QueryExecutionId"]
    deadline = time.time() + timeout_s
    while True:
        st = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if st["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if time.time() > deadline:
            raise TimeoutError(f"Athena query {qid} still running after {timeout_s}s")
        time.sleep(1.5)
    if st["State"] != "SUCCEEDED":
        raise RuntimeError(f"Athena {st['State']}: {st.get('StateChangeReason', '')}\n{sql[:200]}")
    rows = client.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    return [[c.get("VarCharValue") for c in r["Data"]] for r in rows]
