"""
Section 5.3 — Marked-null direct estimation for NON-AGGREGATION (set) queries.

Two cases based on where marked nulls appear:

Case 1: Marked nulls in separating-set or non-selection attributes ONLY.
  → Tuple independence is preserved.
  → The log-transform identity holds.
  → Standard Section 5.2 estimator applies directly.
  → Per-symbol distributions resolve P_tilde(t) via composition.
  → No MC needed — runs the same SQL as unmarked Section 5.2.

Case 2: Marked nulls in SELECTION attributes.
  → Tuples sharing the same symbol have identical predicate outcomes.
  → The product formula ∏_t (1 - P_tilde(t)) assumes independence — invalid.
  → Fall back to MC: sample W worlds from per-symbol distributions,
    run the query on each, average the results.
"""

import os, re, json, time, math
import psycopg2
import pandas as pd
import numpy as np
from io import StringIO
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from direct_marked_agg import build_symbol_distributions, sample_world
from nonAgg_direct import (
    run_direct_per_tuple, _load_table,
    _parse_set_query, _parse_where_clauses,
    CONN_PARAMS, JSON_PATH, DISCRETIZED_JSON_PATH,
)

MC_WORLDS = 50


def _norm_val(v):
    s = str(v)
    if s in ("nan", "None"):
        return s
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return s
    except (ValueError, TypeError):
        return s


def _has_marked_nulls_in_selection(df, query, missing_attrs):
    """
    Check if any marked-null symbol appears in a selection attribute
    (an attribute referenced in the WHERE clause).
    """
    parsed = _parse_set_query(query)
    if not parsed:
        return False
    where_clauses = _parse_where_clauses(parsed["where_str"])
    where_attrs = set(col for col, _, _ in where_clauses)

    for attr in missing_attrs:
        if attr not in where_attrs:
            continue
        sym_col = "%s_nullsym" % attr
        if sym_col in df.columns and df[sym_col].notna().any():
            return True
    return False


def _load_table_from_df(conn, df, table_name):
    """Load a DataFrame into PostgreSQL."""
    cur = conn.cursor()
    df_l = df.copy()
    df_l.columns = df_l.columns.str.lower()
    cols = list(df_l.columns)
    type_map = {'int64': 'BIGINT', 'float64': 'NUMERIC'}
    ddl = ", ".join('"%s" %s' % (c, type_map.get(str(df_l[c].dtype), 'TEXT')) for c in cols)
    cur.execute('DROP TABLE IF EXISTS "%s"' % table_name)
    cur.execute('CREATE TABLE "%s" (%s)' % (table_name, ddl))
    buf = StringIO()
    df_l.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    col_list = ", ".join('"%s"' % c for c in cols)
    cur.copy_expert(
        'COPY "%s"(%s) FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')' % (table_name, col_list),
        buf)
    conn.commit()
    cur.execute('ANALYZE "%s"' % table_name)
    conn.commit()


def run_marked_nonAgg(conn, query, table_name, gt_table,
                       df_marked, missing_attrs, ordering,
                       method="mgraph", W=MC_WORLDS, seed=42):
    """
    Marked-null non-aggregation estimation.

    Case 1 (no marked nulls in selection attrs): run Section 5.2 directly.
    Case 2 (marked nulls in selection attrs): MC over symbol configurations.
    """
    t0 = time.time()

    selection_case = _has_marked_nulls_in_selection(df_marked, query, missing_attrs)

    if not selection_case:
        # Case 1: standard Section 5.2 — tuple independence preserved
        # Just run run_direct_per_tuple on the existing table.
        # The marked nulls in non-selection attrs don't affect the estimator.
        result = run_direct_per_tuple(
            conn, query, table_name, gt_table,
            missing_attrs, ordering,
        )
        result["case"] = "analytical"
        result["W"] = 0
        return result

    # Case 2: MC fallback — sample W worlds, run Section 5.2 on each
    rng = np.random.default_rng(seed)
    symbol_dists = build_symbol_distributions(
        df_marked, missing_attrs, ordering, method)

    tv_sum = 0.0
    dw_sum = 0.0
    n_ok = 0

    for w in range(W):
        df_w = sample_world(df_marked, missing_attrs, symbol_dists, rng)

        temp_table = "_mknull_w%d" % w
        try:
            _load_table_from_df(conn, df_w, temp_table)
        except Exception:
            continue

        r = run_direct_per_tuple(
            conn, query, temp_table, gt_table,
            missing_attrs, ordering,
        )

        try:
            cur = conn.cursor()
            cur.execute('DROP TABLE IF EXISTS "%s"' % temp_table)
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

        if r.get("error"):
            continue

        n_ok += 1
        tv_sum += r.get("tv_prob", 0.0)
        dw_sum += r.get("delta_w", 0.0)

    elapsed = time.time() - t0

    if n_ok == 0:
        return {
            "tv_prob": 1.0, "delta_w": 0.0,
            "time_s": elapsed, "W": 0,
            "case": "mc_fallback", "error": "no worlds completed",
        }

    return {
        "tv_prob": tv_sum / n_ok,
        "delta_w": dw_sum / n_ok,
        "time_s": elapsed,
        "W": n_ok,
        "case": "mc_fallback",
    }


def main():
    conn = psycopg2.connect(**CONN_PARAMS)
    conn.autocommit = False

    json_path = JSON_PATH
    if os.path.isfile(DISCRETIZED_JSON_PATH):
        json_path = DISCRETIZED_JSON_PATH
        print("Using discretized JSON: %s" % json_path, flush=True)
    else:
        print("Using original JSON: %s" % json_path, flush=True)

    with open(json_path) as f:
        cfg = json.load(f)

    results = []
    MCDB_DATA_DIR = "data/mcdb_test_data"

    for group_key in ["bank_manr1_set", "nyc_manr1_set", "bit_manr1_set"]:
        if group_key not in cfg:
            continue

        blocks = cfg[group_key]
        for block_key, meta in blocks.items():
            csvs = meta.get("csv", [])
            tables = meta.get("table", [])
            complete_csvs = meta.get("complete_csv", [])
            complete_tables = meta.get("complete_table", [])
            missing_attrs = [a.lower() for a in meta.get("missing_attrs_single", [])]
            ordering = {k.lower(): [c.lower() for c in v]
                        for k, v in meta.get("ordering_single", {}).items()}
            queries = meta.get("queries", [])

            if not csvs or not queries:
                continue

            ds_name = None
            for dn in ["bank", "nyc", "bitcoin"]:
                if dn in block_key.lower() or dn in group_key.lower():
                    ds_name = dn
                    break
            rate = None
            for r in [5, 10, 20]:
                if str(r) in block_key:
                    rate = r
                    break

            if not ds_name or not rate:
                continue

            marked_csv = os.path.join(MCDB_DATA_DIR,
                                       "%s_mnar_mcdb_%d_marked.csv" % (ds_name, rate))
            if not os.path.isfile(marked_csv):
                continue

            print("\n" + "=" * 60, flush=True)
            print("Block: %s / %s" % (group_key, block_key), flush=True)
            print("=" * 60, flush=True)

            csv_path = csvs[0]
            table_name = tables[0]
            gt_csv = complete_csvs[0] if complete_csvs else None
            gt_table = complete_tables[0] if complete_tables else None

            try:
                conn.rollback()
            except Exception:
                pass

            _load_table(conn, csv_path, table_name)
            if gt_csv and gt_table:
                _load_table(conn, gt_csv, gt_table)

            df_marked = pd.read_csv(marked_csv)
            df_marked.columns = df_marked.columns.str.lower()
            print("  Loaded marked data: %d rows" % len(df_marked), flush=True)

            nonjoin_qs = [q for q in queries if "JOIN" not in q.upper()]

            for qi, query in enumerate(nonjoin_qs[:4]):
                print("\n  Q%d: %s" % (qi + 1, query[:90]), flush=True)

                sel_case = _has_marked_nulls_in_selection(
                    df_marked, query, missing_attrs)
                print("    Marked nulls in selection: %s → %s" % (
                    sel_case, "MC fallback" if sel_case else "analytical"),
                    flush=True)

                for mc_method, mc_label in [("naive", "Naive-Marked-Direct"),
                                             ("mgraph", "mGraph-Marked-Direct")]:
                    print("    %s..." % mc_label, end="", flush=True)

                    result = run_marked_nonAgg(
                        conn, query, table_name, gt_table or table_name,
                        df_marked, missing_attrs, ordering,
                        method=mc_method, W=MC_WORLDS, seed=42,
                    )

                    print(" TV=%.4f  dw=%.4f  t=%.1fs  case=%s  W=%d" % (
                        result.get("tv_prob", 0),
                        result.get("delta_w", 0),
                        result.get("time_s", 0),
                        result.get("case", "?"),
                        result.get("W", 0)),
                        flush=True)

                    results.append({
                        "group": group_key, "block": block_key,
                        "query_idx": qi + 1, "method": mc_label,
                        "tv_prob": result.get("tv_prob", np.nan),
                        "delta_w": result.get("delta_w", np.nan),
                        "time_s": result.get("time_s", np.nan),
                        "case": result.get("case", ""),
                        "W": result.get("W", 0),
                        "error": result.get("error", ""),
                    })

    conn.close()

    df_out = pd.DataFrame(results)
    out_path = "direct_marked_nonAgg_results.csv"
    df_out.to_csv(out_path, index=False)

    print("\n" + "=" * 60, flush=True)
    print("Results saved to %s" % out_path, flush=True)
    print("=" * 60, flush=True)

    if len(df_out) > 0:
        avg = df_out.groupby(["block", "method"]).agg(
            avg_tv=("tv_prob", "mean"),
            avg_time=("time_s", "mean"),
        ).reset_index()
        print("\nAverage per block per method:", flush=True)
        print(avg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
