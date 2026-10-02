"""
One-command proof that the pipeline publishes to REAL Amazon S3 and is queryable in Athena.

Prerequisites (AWS free tier is enough):
    1. AWS credentials available to boto3 (aws configure, SSO, or AWS_ACCESS_KEY_ID/SECRET env vars)
    2. An S3 bucket you own, e.g.  aws s3 mb s3://my-power-outage-bucket
    3. Athena available in that region (default workgroup "primary" works)

Run:
    export AWS_BUCKET=my-power-outage-bucket      # required
    export AWS_REGION=us-east-1                   # optional
    python scripts/aws_check.py --raw data/raw/merged_utility_storm_2024.csv --limit 200000

What it does:
    1. runs the Prefect flow (Bronze -> Silver -> quality gate -> Gold) and publishes every layer to S3
    2. creates an Athena database and one external table per Silver/Gold table (partitioned by batch_id)
    3. counts rows IN ATHENA and compares them with the counts in the local run manifest
    4. writes docs/aws_run.md (UTC time, region, object list, Athena counts, PASS/FAIL)
Exit code is non-zero on any mismatch or error. Costs: a few cents at most with --limit.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/merged_utility_storm_2024.csv")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--limit", type=int, default=None, help="read only the first N rows (cheaper test run)")
    ap.add_argument("--prefix", default="power-outage")
    ap.add_argument("--database", default=os.environ.get("ATHENA_DATABASE", "power_outage"))
    ap.add_argument("--workgroup", default=os.environ.get("ATHENA_WORKGROUP", "primary"))
    a = ap.parse_args()

    bucket = os.environ.get("AWS_BUCKET")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not bucket:
        print("Set AWS_BUCKET to the name of an S3 bucket you own. See the docstring at the top of this file.")
        return 2

    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    from src import flows, s3_publish

    try:
        boto3.client("sts", region_name=region).get_caller_identity()
    except (BotoCoreError, ClientError) as e:
        print(f"AWS credentials problem: {e}")
        return 1

    print("1/4 Running the pipeline and publishing to S3 ...", flush=True)
    m = flows.medallion_flow(a.raw, a.data_dir, s3_bucket=bucket, s3_prefix=a.prefix, s3_region=region,
                             limit=a.limit)
    batch = m["batch_id"]
    published = s3_publish.publish(a.data_dir, bucket, batch, a.prefix, region)   # idempotent; gives schemas

    print("2/4 Creating Athena database and tables ...", flush=True)
    athena = boto3.client("athena", region_name=region)
    out_loc = f"s3://{bucket}/athena-results/"
    for stmt in s3_publish.athena_ddl(published, a.database):
        s3_publish.athena_query(athena, stmt, out_loc, workgroup=a.workgroup)

    print("3/4 Counting rows in Athena and comparing with the manifest ...", flush=True)
    expected = {
        "silver_outage_events": m["silver"]["rows"],
        "silver_quarantine": m["silver"]["quarantined"],
    }
    if "gold" in m:
        expected["gold_utility_features"] = m["gold"]["utilities"]
    checks, ok = [], True
    for table, want in expected.items():
        rows = s3_publish.athena_query(
            athena, f"SELECT count(*) FROM {a.database}.{table} WHERE batch_id = '{batch}'",
            out_loc, a.database, a.workgroup)
        got = int(rows[1][0])
        good = got == want
        ok &= good
        checks.append((table, want, got, good))
        print(f"   {table:24s} manifest={want:>10,}  athena={got:>10,}  {'OK' if good else 'MISMATCH'}")

    print("4/4 Writing docs/aws_run.md ...", flush=True)
    masked = bucket[:3] + "***"
    lines = [
        "# AWS run evidence", "",
        f"- Run at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Result: **{'PASS' if ok else 'FAIL'}**",
        f"- Region: `{region or 'default'}`   Bucket: `{masked}`   Prefix: `{a.prefix}`   Batch: `{batch}`",
        f"- Rows published (limit={a.limit}): bronze {m['bronze']['rows']:,}, silver {m['silver']['rows']:,}",
        "", "## Row counts: run manifest vs Athena", "",
        "| Table | Manifest | Athena | Match |", "|---|---:|---:|:---:|",
        *[f"| {t} | {w:,} | {g:,} | {'yes' if good else 'NO'} |" for t, w, g, good in checks],
        "", "## Objects in S3", "",
        *[f"- `{o['key']}` ({o['bytes']:,} bytes)" for o in published["objects"]],
    ]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "aws_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{'PASS' if ok else 'FAIL'}. Evidence: docs/aws_run.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
