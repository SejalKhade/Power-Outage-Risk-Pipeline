"""
features.py
-----------
PURPOSE : Convert 3.4M row clean dataset → 1,677 utility-level ML dataset
INPUT   : data/processed/dashboard_clean_dataset.parquet
OUTPUT  : data/processed/utility_features.parquet

WHY THIS STAGE EXISTS:
    Your ML model trains on ONE ROW PER UTILITY (1,677 utilities).
    But your clean dataset has 3.4M rows — multiple storm events
    per utility per year.

    This file:
    1. Aggregates storm events → utility-level summaries
    2. Builds risk scores from SAIDI/SAIFI percentile ranks
    3. Adds economic impact estimates in dollars
    4. Creates binary ML labels (high_risk = 1 or 0)
    5. Saves a clean utility-level parquet ready for model training
"""

import os
import pandas as pd
import numpy as np
from src.utils import ensure_dir, log_step, log_section


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

INPUT_PATH  = "data/processed/dashboard_clean_dataset.parquet"
OUTPUT_DIR  = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "utility_features.parquet")

# These columns identify a utility — they stay as-is
IDENTITY_COLS = [
    "Utility Number",
    "Utility Name",
    "State",
    "Ownership",
    "NERC Region",
    "County_Count",
]

# These are the reliability metrics we aggregate with MAX
# WHY MAX not MEAN:
#   Each utility appears thousands of times in the dataset
#   (once per storm event per year). The reliability metrics
#   (SAIDI, SAIFI, CAIDI) are only filled on ONE row per utility
#   per year — the rest are 0.
#   MAX correctly picks up the real value. MEAN would divide
#   it by thousands of rows and give you near-zero for everyone.
RELIABILITY_COLS = [
    "IEEE_AllEvents_SAIDI_min_per_yr",
    "IEEE_AllEvents_SAIFI_times_per_yr",
    "IEEE_AllEvents_CAIDI_min_per_interruption",
    "IEEE_NoMED_SAIDI_min_per_yr",
    "IEEE_NoMED_SAIFI_times_per_yr",
    "IEEE_NoMED_CAIDI_min_per_interruption",
]

# Storm features we aggregate with SUM
# WHY SUM: Total damage and injuries accumulate across all storms
STORM_SUM_COLS = [
    "DAMAGE_PROPERTY_USD",
    "DAMAGE_CROPS_USD",
    "INJURIES_DIRECT",
    "INJURIES_INDIRECT",
    "DEATHS_DIRECT",
    "DEATHS_INDIRECT",
]

# NERC region binary flags — take MAX (1 if utility is in that region)
NERC_BINARY_COLS = [
    "TRE", "FRCC", "MRO", "NPCC", "RFC",
    "SERC", "SPP", "WECC", "CAISO", "ERCOT",
    "PJM", "NYISO", "MISO", "ISONE",
]

# Leakage columns — NEVER use these as ML features
# WHY: These are derived FROM the target (risk_score/high_risk).
# Using them would let the model cheat — predicting risk from risk.
# This is called data leakage and gives false accuracy scores.
LEAKAGE_COLS = [
    # Direct target and derived labels
    "high_risk",
    "risk_score",
    "risk_category",
    "saidi_rank_pct",
    "saifi_rank_pct",

    # ── THE CRITICAL ADDITIONS ──
    # These are calculated FROM SAIDI/SAIFI which defines the target.
    # Including them lets the model predict risk FROM risk — cheating.

    # estimated_annual_loss = SAIDI × customers × $27
    # The model just learns: high loss = high SAIDI = high risk
    "estimated_annual_loss_usd",

    # nerc_sla_breach_risk = (SAIDI > 150) — literally the target in binary form
    "nerc_sla_breach_risk",

    # sla_breach_margin_min = SAIDI - 150 — SAIDI shifted by a constant
    "sla_breach_margin_min",

    # The raw reliability metrics that DEFINE the target
    # WHY EXCLUDE THESE:
    #   high_risk = top 20% by SAIDI/SAIFI percentile rank
    #   If we train on SAIDI to predict high_risk, the model trivially
    #   learns "big SAIDI = high risk" — not a useful prediction.
    #   The interesting question is: can storm exposure, grid structure,
    #   and regional features predict reliability WITHOUT using SAIDI itself?
    "IEEE_AllEvents_SAIDI_min_per_yr",
    "IEEE_AllEvents_SAIFI_times_per_yr",
    "IEEE_AllEvents_CAIDI_min_per_interruption",
    "IEEE_NoMED_SAIDI_min_per_yr",
    "IEEE_NoMED_SAIFI_times_per_yr",
    "IEEE_NoMED_CAIDI_min_per_interruption",

    # total_damage_usd and log_total_damage stay in as WEATHER features
    # They measure storm impact, not reliability performance
]

# Weather feature names — used for Utility-Only vs Utility+Weather comparison
WEATHER_FEATURES = {
    # Storm damage features
    "total_damage_usd",
    "total_property_damage_usd",
    "total_crops_damage_usd",
    "log_total_damage",

    # Storm human impact
    "total_injuries",
    "human_impact_score",
    "INJURIES_DIRECT",
    "INJURIES_INDIRECT",
    "DEATHS_DIRECT",
    "DEATHS_INDIRECT",

    # Storm event characteristics
    "MAGNITUDE",
    "weather_event_count",
    "months_with_events",
}

# Economic cost per customer per hour of outage
# Source: US Department of Energy, 2022
DOE_COST_PER_CUSTOMER_HOUR_USD = 27.0

# NERC reliability standard threshold
# Utilities above this SAIDI are considered at breach risk
NERC_SAIDI_THRESHOLD = 150.0


# ══════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════

def build_utility_features(input_path: str = INPUT_PATH,
                           output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Runs all feature engineering steps in sequence.
    Returns a DataFrame with 1 row per utility, ready for ML training.

    input_path / output_path default to the original locations, so existing
    callers are unchanged. The medallion pipeline passes Silver -> Gold paths.
    """
    log_section("STAGE 2 — FEATURE ENGINEERING")

    # ── Load clean parquet ─────────────────────────────────────────
    print(f"  Loading: {input_path}")
    df = pd.read_parquet(input_path)
    log_step("LOADED", df)

    # ── Step A: Fix numeric types ──────────────────────────────────
    df = _fix_numeric_types(df)
    print("  ✓ Step A: Numeric types fixed")

    # ── Step B: Aggregate to utility level ─────────────────────────
    utility_df = _aggregate_to_utility(df)
    log_step("AGGREGATED", utility_df)
    print(f"  ✓ Step B: Aggregated to {len(utility_df):,} utilities")

    # ── Step C: Add risk scores ────────────────────────────────────
    utility_df = _add_risk_scores(utility_df)
    print("  ✓ Step C: Risk scores added")

    # ── Step D: Add economic impact ────────────────────────────────
    utility_df = _add_economic_impact(utility_df)
    print("  ✓ Step D: Economic impact added")

    # ── Step E: Add ML labels ──────────────────────────────────────
    utility_df = _add_ml_labels(utility_df)
    print("  ✓ Step E: ML labels added")

    # ── Step F: Save output ────────────────────────────────────────
    ensure_dir(os.path.dirname(output_path) or ".")
    utility_df.to_parquet(output_path, index=False)
    print(f"\n  ✓ Saved → {output_path}")
    log_step("FINAL", utility_df)

    return utility_df


# ══════════════════════════════════════════════════════════════════
# STEP A — Fix numeric types
# ══════════════════════════════════════════════════════════════════

def _fix_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all numeric columns to proper float types.

    WHY THIS STEP EXISTS:
        When pandas reads mixed-type columns (numbers + strings like
        'N/A' or '--'), it stores everything as object (string) type.
        Math operations on object columns fail or give wrong results.
        We force-convert to float here so all downstream code is safe.

    errors='coerce' means:
        If a value cannot be converted to float, replace with NaN.
        Then fillna(0.0) replaces NaN with 0.
        Result: every numeric column is guaranteed to be float with no NaN.
    """
    all_numeric = (
        RELIABILITY_COLS +
        STORM_SUM_COLS +
        NERC_BINARY_COLS +
        ["MAGNITUDE", "County_Count"]
    )

    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df


# ══════════════════════════════════════════════════════════════════
# STEP B — Aggregate to utility level
# ══════════════════════════════════════════════════════════════════

def _aggregate_to_utility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses 3.4M rows → 1 row per utility.

    AGGREGATION RULES:
        Identity cols  → first value (same for all rows of a utility)
        Reliability    → MAX (real value is on only one row, rest are 0)
        Storm totals   → SUM (accumulate across all storms)
        Storm counts   → COUNT / NUNIQUE
        NERC flags     → MAX (1 if utility is in that region)
        Magnitude      → MAX (worst storm intensity)

    WHY GROUP BY 'Utility Number' NOT 'Utility Name':
        Utility names can have typos or slight variations.
        'Utility Number' is a unique ID assigned by EIA — guaranteed unique.
    """
    print("  Aggregating 3.4M rows to utility level...")

    # Build aggregation dictionary dynamically
    # (only include columns that actually exist in the dataframe)
    agg_dict = {}

    # Identity columns — take first value
    for col in ["Utility Name", "State", "Ownership",
                "NERC Region", "County_Count"]:
        if col in df.columns:
            agg_dict[col] = "first"

    # Reliability metrics — take MAX
    for col in RELIABILITY_COLS:
        if col in df.columns:
            agg_dict[col] = "max"

    # Storm damage and human impact — SUM
    for col in STORM_SUM_COLS:
        if col in df.columns:
            agg_dict[col] = "sum"

    # Storm event counts
    if "EVENT_TYPE" in df.columns:
        agg_dict["EVENT_TYPE"] = "count"    # total storm events
    if "MONTH_NAME" in df.columns:
        agg_dict["MONTH_NAME"] = "nunique"  # how many different months had events

    # Storm intensity — MAX magnitude
    if "MAGNITUDE" in df.columns:
        agg_dict["MAGNITUDE"] = "max"

    # NERC binary flags — MAX (0 or 1)
    for col in NERC_BINARY_COLS:
        if col in df.columns:
            agg_dict[col] = "max"

    # Run aggregation
    utility_df = df.groupby("Utility Number", as_index=False).agg(agg_dict)

    # Rename aggregated storm columns to clearer names
    rename_map = {}
    if "EVENT_TYPE" in utility_df.columns:
        rename_map["EVENT_TYPE"] = "weather_event_count"
    if "MONTH_NAME" in utility_df.columns:
        rename_map["MONTH_NAME"] = "months_with_events"
    if "DAMAGE_PROPERTY_USD" in utility_df.columns:
        rename_map["DAMAGE_PROPERTY_USD"] = "total_property_damage_usd"
    if "DAMAGE_CROPS_USD" in utility_df.columns:
        rename_map["DAMAGE_CROPS_USD"] = "total_crops_damage_usd"

    utility_df = utility_df.rename(columns=rename_map)

    # Add combined damage column
    if "total_property_damage_usd" in utility_df.columns:
        utility_df["total_damage_usd"] = (
            utility_df["total_property_damage_usd"].fillna(0) +
            utility_df.get("total_crops_damage_usd",
                           pd.Series(0, index=utility_df.index)).fillna(0)
        )

    return utility_df


# ══════════════════════════════════════════════════════════════════
# STEP C — Add risk scores
# ══════════════════════════════════════════════════════════════════

def _add_risk_scores(utility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks every utility by SAIDI and SAIFI percentile.
    Creates a composite risk score and High/Medium/Low category.

    HOW PERCENTILE RANK WORKS:
        A utility with SAIDI = 500 min/yr gets rank 0.95 if 95% of
        utilities have lower SAIDI. This means it is in the top 5%
        worst performers — High Risk.

    WHY PERCENTILE NOT RAW VALUES:
        Raw SAIDI varies hugely by region and climate.
        A utility in Florida (hurricane country) will always have
        higher raw SAIDI than one in Nevada (desert, few storms).
        Percentile rank compares each utility against ALL others —
        a fair, standardised measure of relative performance.

    RISK TIERS:
        Top 20% (rank >= 0.80) → High Risk
        Middle 30% (0.50-0.80) → Medium Risk
        Bottom 50% (rank < 0.50) → Low Risk

    WHY TOP 20% FOR HIGH RISK:
        Your capstone notebook tested Top 10% vs Top 20%.
        Top 20% gave PR-AUC 0.6185 vs Top 10% gave only 0.4184.
        More balanced labels = better model performance.
    """
    saidi_col = "IEEE_AllEvents_SAIDI_min_per_yr"
    saifi_col = "IEEE_AllEvents_SAIFI_times_per_yr"

    if saidi_col in utility_df.columns:
        # pct=True gives rank as fraction between 0 and 1
        # na_option='bottom' puts missing values at the bottom
        utility_df["saidi_rank_pct"] = utility_df[saidi_col].rank(
            pct=True, na_option='bottom'
        )

    if saifi_col in utility_df.columns:
        utility_df["saifi_rank_pct"] = utility_df[saifi_col].rank(
            pct=True, na_option='bottom'
        )

    # Composite score = average of both percentile ranks
    if "saidi_rank_pct" in utility_df.columns and \
       "saifi_rank_pct" in utility_df.columns:
        utility_df["risk_score"] = (
            utility_df["saidi_rank_pct"] +
            utility_df["saifi_rank_pct"]
        ) / 2

        # Assign risk categories
        q80 = utility_df["risk_score"].quantile(0.80)
        q50 = utility_df["risk_score"].quantile(0.50)

        utility_df["risk_category"] = np.where(
            utility_df["risk_score"] >= q80, "High Risk",
            np.where(
                utility_df["risk_score"] >= q50, "Medium Risk",
                "Low Risk"
            )
        )

        # Print distribution
        dist = utility_df["risk_category"].value_counts()
        print(f"\n  Risk category distribution:")
        for cat, count in dist.items():
            pct = count / len(utility_df) * 100
            print(f"    {cat:<15}: {count:>5} utilities ({pct:.1f}%)")

    return utility_df


# ══════════════════════════════════════════════════════════════════
# STEP D — Add economic impact
# ══════════════════════════════════════════════════════════════════

def _add_economic_impact(utility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Translates technical SAIDI metric into dollar impact.

    FORMULA:
        Annual loss = (SAIDI minutes / 60) × customers × $27/hour

    WHY THIS MATTERS FOR YOUR RESUME:
        Technical metrics like SAIDI mean nothing to a business person.
        'This utility causes $47M annual economic loss to its customers'
        is a board-level insight that drives investment decisions.
        This is what Business Analyst and Operations Analyst roles do.

    CUSTOMER PROXY:
        EIA 861 does not always report customer counts in this dataset.
        We use County_Count × 50,000 as a rough proxy.
        (Average US county has ~50,000 electricity customers)
    """
    saidi_col = "IEEE_AllEvents_SAIDI_min_per_yr"

    if saidi_col in utility_df.columns:
        county_count = utility_df.get(
            "County_Count",
            pd.Series(1, index=utility_df.index)
        ).fillna(1).clip(lower=1)  # minimum 1 county

        # Annual economic loss estimate
        utility_df["estimated_annual_loss_usd"] = (
            (utility_df[saidi_col] / 60)   # minutes → hours
            * county_count
            * 50_000                        # customers per county proxy
            * DOE_COST_PER_CUSTOMER_HOUR_USD
        ).round(0)

        # NERC SLA breach risk flag
        # WHY: Utilities above 150 min/yr SAIDI are at risk of
        # regulatory action under NERC reliability standards
        utility_df["nerc_sla_breach_risk"] = (
            utility_df[saidi_col] > NERC_SAIDI_THRESHOLD
        ).astype(int)

        # How far above/below the threshold
        utility_df["sla_breach_margin_min"] = (
            utility_df[saidi_col] - NERC_SAIDI_THRESHOLD
        ).round(1)

        # Log transform of damage for ML
        # WHY: Damage ranges from $0 to $billions — extreme skew.
        # log1p(x) = log(x+1) compresses this range.
        # The +1 handles zeros (log(0) is undefined).
        if "total_damage_usd" in utility_df.columns:
            utility_df["log_total_damage"] = np.log1p(
                utility_df["total_damage_usd"].fillna(0)
            )

        # Combined human impact score
        if all(c in utility_df.columns for c in
               ["INJURIES_DIRECT", "DEATHS_DIRECT"]):
            utility_df["human_impact_score"] = (
                utility_df["INJURIES_DIRECT"].fillna(0) +
                (utility_df["DEATHS_DIRECT"].fillna(0) * 10)
                # deaths weighted 10x more than injuries
            )

    return utility_df


# ══════════════════════════════════════════════════════════════════
# STEP E — Add ML labels
# ══════════════════════════════════════════════════════════════════

def _add_ml_labels(utility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary classification label for ML training.

    high_risk = 1 → utility is in top 20% by composite risk score
    high_risk = 0 → utility is in bottom 80%

    WHY BINARY NOT MULTI-CLASS:
        We tested 3-class (High/Medium/Low) in early experiments.
        Binary classification gave better PR-AUC scores because
        the class boundary is cleaner with just two categories.
        Your capstone notebook confirmed this.
    """
    if "risk_score" in utility_df.columns:
        threshold = utility_df["risk_score"].quantile(0.80)
        utility_df["high_risk"] = (
            utility_df["risk_score"] >= threshold
        ).astype(int)

        n_high  = utility_df["high_risk"].sum()
        n_total = len(utility_df)
        print(f"\n  ML Labels:")
        print(f"    high_risk = 1 : {n_high:>5} utilities ({n_high/n_total*100:.1f}%)")
        print(f"    high_risk = 0 : {n_total-n_high:>5} utilities ({(n_total-n_high)/n_total*100:.1f}%)")

    return utility_df


# ══════════════════════════════════════════════════════════════════
# HELPER — get feature sets for ML training
# ══════════════════════════════════════════════════════════════════

def get_feature_sets(utility_df: pd.DataFrame) -> dict:
    """
    Returns two feature sets for the sensitivity analysis.

    WHY TWO SETS:
        Your capstone proved weather features add +0.065 PR-AUC.
        We preserve both sets so train.py can benchmark them.

    RETURNS:
        {
            'Utility Only':      [list of column names],
            'Utility + Weather': [list of column names]
        }
    """
    # All numeric columns except leakage and identity columns
    exclude = set(LEAKAGE_COLS + IDENTITY_COLS + ["Utility Number"])
    all_numeric = utility_df.select_dtypes(include=[np.number]).columns
    candidate_features = [c for c in all_numeric if c not in exclude]

    utility_only = [
        c for c in candidate_features
        if c not in WEATHER_FEATURES
    ]

    utility_weather = candidate_features.copy()

    print(f"\n  Feature sets:")
    print(f"    Utility Only    : {len(utility_only)} features")
    print(f"    Utility+Weather : {len(utility_weather)} features")

    return {
        "Utility Only":      utility_only,
        "Utility + Weather": utility_weather,
    }


# ══════════════════════════════════════════════════════════════════
# RUN AS SCRIPT
# Usage: python -m src.features
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    utility_df = build_utility_features()

    print(f"\n{'═'*50}")
    print("  FEATURE ENGINEERING COMPLETE")
    print(f"  Utilities: {len(utility_df):,}")
    print(f"  Features:  {len(utility_df.columns)} columns")
    print(f"  Output:    {OUTPUT_PATH}")
    print(f"{'═'*50}\n")

    # Show feature sets available for training
    feature_sets = get_feature_sets(utility_df)