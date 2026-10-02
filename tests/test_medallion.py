"""Tests for the Bronze -> Silver -> Gold pipeline (src/medallion.py)."""
import csv
import json
import sys
import os

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath("."))

from src import medallion
from src.medallion import ContractError
from src.preprocess import KEEP_COLUMNS, NUMERIC_COLS, TEXT_COLS, _clean_single_chunk

DEFAULTS = {
    "Utility Number": "100", "Utility Name": "Acme Power", "State": "TX",
    "Ownership": "Cooperative", "NERC Region": "SPP",
    "County_Count": "2.0",
    "IEEE_AllEvents_SAIDI_min_per_yr": "100.5", "IEEE_AllEvents_SAIFI_times_per_yr": "1.2",
    "IEEE_AllEvents_CAIDI_min_per_interruption": "80.0",
    "IEEE_NoMED_SAIDI_min_per_yr": "90.0", "IEEE_NoMED_SAIFI_times_per_yr": "1.1",
    "IEEE_NoMED_CAIDI_min_per_interruption": "75.0",
    "EVENT_TYPE": "Hail", "MONTH_NAME": "April", "MAGNITUDE": "1.5",
    "DAMAGE_PROPERTY_USD": "1000.0", "DAMAGE_CROPS_USD": "0.0",
    "INJURIES_DIRECT": "0", "INJURIES_INDIRECT": "0", "DEATHS_DIRECT": "0", "DEATHS_INDIRECT": "0",
    "BEGIN_DATE_TIME": "01-APR-24 10:40:00",
}


def row(**over):
    r = {c: DEFAULTS.get(c, "") for c in KEEP_COLUMNS}
    r["EXTRA_COL"] = "ignored"
    r.update(over)
    return r


GOOD = [
    row(),
    row(**{"Utility Number": "200", "State": "CA", "EVENT_TYPE": "Wildfire", "TRE": "Y"}),
    row(**{"Utility Number": "300", "State": "NY", "MAGNITUDE": "NaN"}),              # NaN -> 0.0
    row(**{"Utility Number": "400", "Ownership": "", "NERC Region": " WECC "}),        # blank -> Unknown, trim
    row(**{"Utility Number": "500", "IEEE_AllEvents_SAIDI_min_per_yr": "abc"}),        # unparseable -> 0.0
]
BAD = [
    row(State="ZZ"),                                                                   # invalid state
    row(State=""),                                                                     # missing state
    row(**{"Utility Number": "600", "INJURIES_DIRECT": "-3"}),                         # negative value
]
DUPLICATES = [GOOD[0].copy(), GOOD[1].copy()]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture()
def raw(tmp_path):
    return write_csv(tmp_path / "raw.csv", GOOD + BAD + DUPLICATES)


@pytest.fixture()
def result(raw, tmp_path):
    return medallion.run(raw, tmp_path / "data", batch_id="t1", build_gold_layer=False)


def q(sql, *args):
    return duckdb.connect().execute(sql, list(args)).df()


class TestBronze:
    def test_lands_every_source_row(self, result):
        assert result["bronze"]["rows"] == len(GOOD) + len(BAD) + len(DUPLICATES)

    def test_all_source_columns_are_text_and_extra_columns_survive(self, result):
        d = q(f"DESCRIBE SELECT * FROM read_parquet('{result['bronze']['path']}')")
        types = dict(zip(d["column_name"], d["column_type"]))
        for c in KEEP_COLUMNS + ["EXTRA_COL"]:
            assert types[c] == "VARCHAR", c

    def test_lineage_columns(self, result):
        df = q(f"SELECT * FROM read_parquet('{result['bronze']['path']}')")
        assert (df["_batch_id"] == "t1").all()
        assert (df["_source_file"] == "raw.csv").all()
        assert df["_row_id"].is_unique and df["_ingested_at"].notna().all()

    def test_bronze_is_append_only(self, raw, tmp_path):
        medallion.run(raw, tmp_path / "data", batch_id="a", build_gold_layer=False)
        medallion.run(raw, tmp_path / "data", batch_id="b", build_gold_layer=False)
        assert {p.name for p in (tmp_path / "data/bronze/outage_events").iterdir()} == {"batch_a", "batch_b"}
        with pytest.raises(FileExistsError):
            medallion.run(raw, tmp_path / "data", batch_id="a", build_gold_layer=False)


class TestSilver:
    def test_row_count_identity(self, result):
        s = result["silver"]
        assert s["bronze_rows_in"] == s["quarantined"] + s["duplicates_removed"] + s["rows"]

    def test_counts(self, result):
        s = result["silver"]
        assert s["quarantine_by_reason"] == {"invalid_state": 2, "negative_value": 1}
        assert s["duplicates_removed"] == 2
        assert s["rows"] == len(GOOD)

    def test_quarantine_table_has_reason_and_row_id(self, result):
        qt = q(f"SELECT * FROM read_parquet('{result['silver']['quarantine_path']}')")
        assert sorted(qt["_reject_reason"]) == ["invalid_state", "invalid_state", "negative_value"]
        assert qt["_row_id"].notna().all()

    def test_types(self, result):
        d = q(f"DESCRIBE SELECT * FROM read_parquet('{result['silver']['path']}')")
        t = dict(zip(d["column_name"], d["column_type"]))
        assert t["Utility Number"] == "BIGINT"
        assert t["MAGNITUDE"] == "DOUBLE" and t["DAMAGE_PROPERTY_USD"] == "DOUBLE"
        assert t["event_ts"].startswith("TIMESTAMP")

    def test_cleaning_rules(self, result):
        df = q(f"SELECT * FROM read_parquet('{result['silver']['path']}')").set_index("Utility Number")
        assert df.loc[300, "MAGNITUDE"] == 0.0                               # NaN -> 0
        assert df.loc[500, "IEEE_AllEvents_SAIDI_min_per_yr"] == 0.0         # unparseable -> 0
        assert df.loc[400, "Ownership"] == "Unknown"                         # blank -> Unknown
        assert df.loc[400, "NERC Region"] == "WECC"                          # trimmed
        assert df.loc[100, "event_ts"] == pd.Timestamp("2024-04-01 10:40:00")

    def test_soft_expectations_are_counted_not_acted_on(self, result):
        z = result["silver"]["soft_expectations"]["coerced_to_zero"]
        assert z["IEEE_AllEvents_SAIDI_min_per_yr"]["unparseable"] == 1
        assert z["MAGNITUDE"]["null_or_nan"] == 1

    def test_missing_required_column_fails_fast(self, tmp_path):
        rows = [{k: v for k, v in r.items() if k != "EVENT_TYPE"} for r in GOOD]
        path = write_csv(tmp_path / "bad.csv", rows)
        with pytest.raises(ContractError, match="EVENT_TYPE"):
            medallion.run(path, tmp_path / "data", batch_id="x", build_gold_layer=False)

    def test_manifest_written(self, result, tmp_path):
        m = json.loads((tmp_path / "data/manifest/run_t1.json").read_text())
        assert m["batch_id"] == "t1" and len(m["source"]["sha256"]) == 64
        assert "PASS" in m["identity_check"]


class TestParityWithLegacyPreprocess:
    """Silver must equal the original pandas cleaning (src/preprocess.py) on the rows
    the legacy code keeps. The only intended differences are the new quarantine rule
    (negative values) and dropping the redundant *_USD_USD columns."""

    def test_same_rows_and_values(self, tmp_path):
        rows = [r for r in GOOD + DUPLICATES]              # exclude rows legacy would handle differently
        path = write_csv(tmp_path / "p.csv", rows + [row(State="ZZ")])
        res = medallion.run(path, tmp_path / "data", batch_id="p", build_gold_layer=False)

        raw = pd.read_csv(path, low_memory=False)
        legacy = _clean_single_chunk(raw[[c for c in KEEP_COLUMNS if c in raw.columns]].copy())
        legacy = legacy.drop_duplicates()
        silver = q(f"SELECT * FROM read_parquet('{res['silver']['path']}')")

        assert len(silver) == len(legacy)
        key = ["Utility Number", "EVENT_TYPE"]
        a = silver.sort_values(key).reset_index(drop=True)
        b = legacy.sort_values(key).reset_index(drop=True)
        for c in NUMERIC_COLS + ["DAMAGE_PROPERTY_USD", "DAMAGE_CROPS_USD"]:
            assert a[c].astype(float).tolist() == b[c].astype(float).tolist(), c
        for c in TEXT_COLS + ["State", "Utility Name", "TRE"]:
            assert a[c].fillna("").astype(str).tolist() == b[c].fillna("").astype(str).tolist(), c
        assert a["Utility Number"].astype("int64").tolist() == b["Utility Number"].astype("int64").tolist()
