#!/usr/bin/env python3
"""
Inject marked-null symbols into MNAR test data.

For each missing attribute, groups NULL rows into clusters that share a
marked-null symbol. Symbols are assigned across DIFFERENT separating-set
strata so the PoE combination is non-trivial.

Each output CSV gets an extra column per missing attr: `<attr>_nullsym`
containing the integer symbol ID (or NaN for observed rows).
Rows sharing the same symbol ID must receive the same imputed value.
"""

import os, json, argparse
import numpy as np
import pandas as pd

MCDB_DATA_DIR = "data/mcdb_test_data"
JSON_PATH = "configs/unsafe_mnar_set_queries.json"

DATASETS = {
    "bank": {
        "missing_attrs": ["day", "contact", "duration", "housing"],
        "ordering": {
            "contact": ["age", "balance", "campaign", "month"],
            "day": ["age", "balance", "campaign", "contact", "default", "education", "housing"],
            "housing": ["age", "balance", "campaign"],
            "duration": ["age", "balance", "campaign", "contact", "day", "default", "education"],
        },
    },
    "nyc": {
        "missing_attrs": ["trip_duration", "pickup_longitude", "dropoff_longitude"],
        "ordering": {
            "pickup_longitude": ["passenger_count", "pickup_latitude", "store_and_fwd_flag"],
            "trip_duration": ["passenger_count", "pickup_latitude", "pickup_longitude", "store_and_fwd_flag"],
            "vendor_id": ["passenger_count", "pickup_longitude", "store_and_fwd_flag", "trip_duration"],
        },
    },
    "bitcoin": {
        "missing_attrs": ["looped", "neighbors", "count"],
        "ordering": {
            "neighbors": ["income", "weight", "year"],
            "count": ["income", "neighbors", "weight", "year"],
            "looped": ["count", "income", "neighbors", "weight", "year"],
        },
    },
}


def _bin_for_strata(series, n_bins=10):
    """Bin a column to create discrete strata."""
    try:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().sum() == 0 or num.nunique() <= n_bins:
            return series.astype(str)
        return pd.qcut(num, q=n_bins, duplicates="drop").astype(str)
    except (ValueError, TypeError):
        return series.astype(str)


def assign_marked_nulls(df, attr, ordering, group_size=3, rng=None):
    """
    Assign marked-null symbols to NULL cells of `attr`.
    Shuffles NULL row indices and chunks into groups of `group_size`.
    Shuffling ensures symbols span different strata naturally.
    Returns a Series of symbol IDs (NaN for observed rows).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    miss_mask = df[attr].isna().values
    n_miss = int(miss_mask.sum())
    if n_miss == 0:
        return pd.Series(np.nan, index=df.index)

    miss_positions = np.where(miss_mask)[0]
    rng.shuffle(miss_positions)

    # Assign symbol IDs: consecutive chunks of group_size
    sym_ids = np.full(len(df), np.nan)
    n_symbols = (n_miss + group_size - 1) // group_size
    for sym_id in range(n_symbols):
        start = sym_id * group_size
        end = min(start + group_size, n_miss)
        sym_ids[miss_positions[start:end]] = sym_id

    return pd.Series(sym_ids, index=df.index)


def inject_dataset(name, rates, group_size, seed):
    info = DATASETS[name]
    rng = np.random.default_rng(seed)

    for rate in rates:
        csv_path = os.path.join(MCDB_DATA_DIR, "%s_mnar_mcdb_%d.csv" % (name, rate))
        if not os.path.isfile(csv_path):
            print("  Skipping %s (not found)" % csv_path)
            continue

        df = pd.read_csv(csv_path)
        print("  %s (%d rows)" % (os.path.basename(csv_path), len(df)))

        for attr in info["missing_attrs"]:
            if attr not in df.columns:
                continue
            sym_col = "%s_nullsym" % attr
            df[sym_col] = assign_marked_nulls(
                df, attr, info["ordering"], group_size, rng)
            n_syms = df[sym_col].dropna().nunique()
            n_miss = df[attr].isna().sum()
            avg_grp = n_miss / max(n_syms, 1)
            print("    %s: %d nulls -> %d symbols (avg %.1f per symbol)" % (
                attr, n_miss, n_syms, avg_grp))

        out_path = os.path.join(MCDB_DATA_DIR, "%s_mnar_mcdb_%d_marked.csv" % (name, rate))
        df.to_csv(out_path, index=False)
        print("    Saved: %s" % out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_size", type=int, default=3)
    parser.add_argument("--rates", type=str, default="5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rates = [int(r) for r in args.rates.split(",")]

    for name in ["bank", "nyc", "bitcoin"]:
        print("\n%s:" % name.upper())
        inject_dataset(name, rates, args.group_size, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
