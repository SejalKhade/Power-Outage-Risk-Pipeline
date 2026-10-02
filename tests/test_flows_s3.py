"""Tests for the Prefect flow (src/flows.py) and the S3 / Athena publisher (src/s3_publish.py).
S3 is mocked with moto, so no AWS account or network is needed."""
import csv
import json
import os
import sys
from pathlib import Path

import boto3
import duckdb
import pytest
from moto import mock_aws
from prefect.testing.utilities import prefect_test_harness

sys.path.insert(0, os.path.abspath("."))

from src import flows, medallion, s3_publish  # noqa: E402
from src.preprocess import KEEP_COLUMNS  # noqa: E402

BUCKET = "test-bucket"


def _row(**over):
    r = {c: "" for c in KEEP_COLUMNS}
    r.update({"Utility Number": "100", "Utility Name": "Acme", "State": "TX", "Ownership": "Cooperative",
              "NERC Region": "SPP", "SPP.1": "Y", "County_Count": "2", "EVENT_TYPE": "Hail", "MONTH_NAME": "April",
              "MAGNITUDE": "1.5", "DAMAGE_PROPERTY_USD": "1000", "DAMAGE_CROPS_USD": "0",
              "INJURIES_DIRECT": "0", "INJURIES_INDIRECT": "0", "DEATHS_DIRECT": "0", "DEATHS_INDIRECT": "0",
              "IEEE_AllEvents_SAIDI_min_per_yr": "100", "BEGIN_DATE_TIME": "01-APR-24 10:40:00"})
    r.update(over)
    return r


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture(scope="module", autouse=True)
def prefect_env():
    with prefect_test_harness():
        yield


@pytest.fixture()
def aws(monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "x", "AWS_SECRET_ACCESS_KEY": "x", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=BUCKET)
        yield s3


@pytest.fixture()
def clean_csv(tmp_path):
    rows = [_row(**{"Utility Number": str(100 + i)}) for i in range(40)] + [_row()]   # last row = exact duplicate
    return _write(tmp_path / "clean.csv", rows)


# ── S3 publishing ──────────────────────────────────────────────────────────
class TestS3Publish:
    def test_uploads_layers_with_expected_keys_and_verified_sizes(self, clean_csv, tmp_path, aws):
        m = medallion.run(clean_csv, tmp_path / "d", batch_id="b1", build_gold_layer=False)
        out = s3_publish.publish(tmp_path / "d", BUCKET, "b1", prefix="po")
        keys = {o["key"] for o in out["objects"]}
        assert "po/bronze/outage_events/batch_id=b1/part-0.parquet" in keys
        assert "po/silver/outage_events/batch_id=b1/outage_events.parquet" in keys
        assert "po/silver/quarantine/batch_id=b1/quarantine.parquet" in keys
        assert "po/manifest/run_b1.json" in keys
        listed = {o["Key"] for o in aws.list_objects_v2(Bucket=BUCKET)["Contents"]}
        assert keys == listed
        assert m["silver"]["rows"] == 40

    def test_silver_is_published_with_snake_case_columns(self, clean_csv, tmp_path, aws):
        medallion.run(clean_csv, tmp_path / "d", batch_id="b2", build_gold_layer=False)
        s3_publish.publish(tmp_path / "d", BUCKET, "b2", prefix="po")
        local = tmp_path / "silver.parquet"
        aws.download_file(BUCKET, "po/silver/outage_events/batch_id=b2/outage_events.parquet", str(local))
        cols = [r[0] for r in duckdb.connect().execute(f"DESCRIBE SELECT * FROM read_parquet('{local.as_posix()}')").fetchall()]
        assert "utility_number" in cols and "spp_1" in cols
        assert not any(" " in c or "." in c for c in cols)

    def test_data_is_encrypted_and_republish_is_idempotent(self, clean_csv, tmp_path, aws):
        medallion.run(clean_csv, tmp_path / "d", batch_id="b3", build_gold_layer=False)
        first = s3_publish.publish(tmp_path / "d", BUCKET, "b3", prefix="po")
        second = s3_publish.publish(tmp_path / "d", BUCKET, "b3", prefix="po")
        assert len(first["objects"]) == len(second["objects"])
        assert len(aws.list_objects_v2(Bucket=BUCKET)["Contents"]) == len(first["objects"])
        head = aws.head_object(Bucket=BUCKET, Key="po/silver/outage_events/batch_id=b3/outage_events.parquet")
        assert head["ServerSideEncryption"] == "AES256"

    def test_athena_ddl_is_partitioned_and_points_at_the_table_folder(self, clean_csv, tmp_path, aws):
        medallion.run(clean_csv, tmp_path / "d", batch_id="b4", build_gold_layer=False)
        out = s3_publish.publish(tmp_path / "d", BUCKET, "b4", prefix="po")
        ddl = "\n".join(s3_publish.athena_ddl(out, "power_outage"))
        assert "CREATE DATABASE IF NOT EXISTS power_outage" in ddl
        assert "PARTITIONED BY (batch_id string)" in ddl
        assert f"LOCATION 's3://{BUCKET}/po/silver/outage_events/'" in ddl
        assert "`utility_number` bigint" in ddl
        assert "ADD IF NOT EXISTS PARTITION (batch_id='b4')" in ddl


# ── Flow ───────────────────────────────────────────────────────────────────
class TestFlow:
    def test_runs_all_layers_and_writes_manifest(self, clean_csv, tmp_path):
        m = flows.medallion_flow(str(clean_csv), str(tmp_path / "d"), build_gold=False)
        assert m["orchestrator"] == "prefect"
        assert m["silver"]["rows"] == 40 and m["silver"]["duplicates_removed"] == 1
        assert m["quality_gate"]["quarantine_pct"] == 0.0
        saved = json.loads((tmp_path / "d" / "manifest" / f"run_{m['batch_id']}.json").read_text())
        assert "PASS" in saved["identity_check"]

    def test_quality_gate_stops_a_dirty_batch_before_gold(self, tmp_path):
        rows = [_row(State="ZZ") for _ in range(30)] + [_row(**{"Utility Number": str(i)}) for i in range(70)]
        csv_path = _write(tmp_path / "dirty.csv", rows)
        with pytest.raises(flows.DataQualityError, match="quarantined"):
            flows.medallion_flow(str(csv_path), str(tmp_path / "d"), max_quarantine_pct=5.0)
        assert not (tmp_path / "d" / "gold").exists()

    def test_publishes_to_s3_when_a_bucket_is_given(self, clean_csv, tmp_path, aws):
        m = flows.medallion_flow(str(clean_csv), str(tmp_path / "d"), build_gold=False,
                                 s3_bucket=BUCKET, s3_prefix="po")
        assert len(m["s3"]["objects"]) >= 4
        assert aws.list_objects_v2(Bucket=BUCKET)["KeyCount"] == len(m["s3"]["objects"])

    def test_transient_upload_failure_is_retried(self, clean_csv, tmp_path, aws, monkeypatch):
        real, calls = s3_publish.publish, {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("network blip")
            return real(*a, **kw)

        monkeypatch.setattr(s3_publish, "publish", flaky)
        monkeypatch.setattr(flows, "publish_s3", flows.publish_s3.with_options(retry_delay_seconds=[0]))
        m = flows.medallion_flow(str(clean_csv), str(tmp_path / "d"), build_gold=False, s3_bucket=BUCKET)
        assert calls["n"] == 2 and m["s3"]["objects"]
