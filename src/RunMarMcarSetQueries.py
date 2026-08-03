"""
Run set queries on MAR and MCAR datasets using:
  1. Section 4: ranking (no-opt skipped — has attribute issues with MAR/MCAR)
  2. Section 5.2: nonAgg_direct (run_direct_per_tuple)

Handles reconnection on DB failures.
Reports TV_prob, time per approach per query.
Output: mar_mcar_set_results.csv
"""

import os, json, re, time
import numpy as np
import pandas as pd
import psycopg2

from RankingQueryExecuter import QueryExecutorRanking as RankExecutor
from RunnerSetQueriy import (
    to_executor_csv_queries,
    build_maps_from_lists, replace_csv_with_tables,
)
from nonAgg_direct import run_direct_per_tuple, _load_table

CONN_PARAMS = dict(host="localhost", port=5433, dbname="mydb",
                   user="alzamill", password=os.environ.get("PGPASSWORD", ""))

RANKING_FRACTION = 0.5
TIMEOUT_RANKING = 120  # seconds — skip if ranking takes longer


def _norm_tuple(r):
    def _nv(v):
        s = str(v)
        if s in ("nan", "None", ""):
            return s
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return s
        except (ValueError, TypeError):
            return s
    return tuple(_nv(x) for x in r)


def _tv_on_sets(pred_set, gt_set):
    if not pred_set and not gt_set:
        return 0.0
    p = 1.0 / len(pred_set) if pred_set else 0.0
    q = 1.0 / len(gt_set) if gt_set else 0.0
    keys = pred_set | gt_set
    return 0.5 * sum(
        abs((p if t in pred_set else 0.0) - (q if t in gt_set else 0.0))
        for t in keys
    )


def get_conn():
    c = psycopg2.connect(**CONN_PARAMS)
    c.autocommit = False
    return c


def ensure_conn(conn):
    """Return conn if alive, else reconnect."""
    try:
        conn.rollback()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return conn
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return get_conn()


def main():
    # Clean DB first
    try:
        c = psycopg2.connect(**CONN_PARAMS)
        c.autocommit = True
        cur = c.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename='alzamill' AND pid<>pg_backend_pid()")
        c.close()
    except Exception:
        pass

    conn = get_conn()
    results = []

    for miss_type, json_path in [("MAR", "configs/mar_set_queries.json"),
                                  ("MCAR", "configs/mcar_set_queries.json")]:
        print("\n" + "#" * 60, flush=True)
        print("  %s SET QUERIES" % miss_type, flush=True)
        print("#" * 60, flush=True)

        with open(json_path) as f:
            cfg = json.load(f)

        for group_key in cfg:
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

                print("\n" + "=" * 60, flush=True)
                print("Block: %s / %s [%s]" % (group_key, block_key, miss_type), flush=True)
                print("=" * 60, flush=True)

                csv_path = csvs[0]
                table_name = tables[0]
                gt_csv = complete_csvs[0] if complete_csvs else None
                gt_table = complete_tables[0] if complete_tables else None

                # Ensure connection is alive
                conn = ensure_conn(conn)

                # Load tables
                try:
                    _load_table(conn, csv_path, table_name, force=True)
                    if gt_csv and gt_table:
                        _load_table(conn, gt_csv, gt_table)
                    print("  Loaded: %s" % os.path.basename(csv_path), flush=True)
                except Exception as e:
                    print("  Load error: %s" % str(e)[:60], flush=True)
                    conn = ensure_conn(conn)
                    try:
                        _load_table(conn, csv_path, table_name, force=True)
                        if gt_csv and gt_table:
                            _load_table(conn, gt_csv, gt_table)
                        print("  Loaded (retry): %s" % os.path.basename(csv_path), flush=True)
                    except Exception as e2:
                        print("  Load failed: %s" % str(e2)[:60], flush=True)
                        continue

                # Prepare ranking executor
                csv_queries = to_executor_csv_queries(
                    block_key, csvs, tables, complete_csvs, complete_tables)

                ordering_T = ordering
                missing_T = missing_attrs

                nonjoin_qs = [q for q in queries if "JOIN" not in q.upper()]

                for qi, query in enumerate(nonjoin_qs):
                    print("\n  Q%d: %s" % (qi + 1, query[:80]), flush=True)

                    # Build SQL
                    full_map, base_map = build_maps_from_lists(csvs, tables)
                    pred_sql = replace_csv_with_tables(query, full_map, base_map)

                    # GT SQL
                    mnar_to_gt = {}
                    mnar_to_gt_base = {}
                    for cp, gt in zip(csvs, complete_tables or tables):
                        mnar_to_gt[cp] = gt
                        bn = os.path.basename(cp)
                        mnar_to_gt_base[bn] = gt
                        stem = os.path.splitext(bn)[0]
                        dn = os.path.basename(os.path.dirname(cp)) or os.path.dirname(cp)
                        mnar_to_gt_base["%s/%s" % (dn, bn)] = gt
                        mnar_to_gt_base["%s/%s" % (dn, stem)] = gt
                    gt_sql = replace_csv_with_tables(query, mnar_to_gt, mnar_to_gt_base)

                    # GT set
                    conn = ensure_conn(conn)
                    gt_flat = set()
                    try:
                        conn.rollback()
                        cur = conn.cursor()
                        cur.execute(gt_sql)
                        gt_flat = {_norm_tuple(r) for r in cur.fetchall()}
                        cur.close()
                        conn.commit()
                    except Exception as e:
                        print("    GT error: %s" % str(e)[:60], flush=True)
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                    row_base = {"miss_type": miss_type, "group": group_key,
                                "block": block_key, "query_idx": qi + 1}

                    # ── Ranking ──
                    conn = ensure_conn(conn)
                    print("    ranking...", end="", flush=True)
                    try:
                        rank_exec = RankExecutor(conn, csv_queries, skip_prepare=True)
                        rank_exec._ordering_T = ordering_T
                        rank_exec._missing_T = missing_T
                        rank_exec.FRACTION = RANKING_FRACTION

                        import signal

                        class TimeoutError(Exception):
                            pass

                        def handler(signum, frame):
                            raise TimeoutError("ranking timeout")

                        old_handler = signal.signal(signal.SIGALRM, handler)
                        signal.alarm(TIMEOUT_RANKING)
                        try:
                            t0 = time.time()
                            flat_rows, iv_width, time_pred, has_bounds = rank_exec.run_flat(
                                pred_sql, ordering_T, missing_T, None, None)
                            elapsed = time.time() - t0
                            signal.alarm(0)
                        except TimeoutError:
                            signal.alarm(0)
                            raise Exception("timeout after %ds" % TIMEOUT_RANKING)
                        finally:
                            signal.signal(signal.SIGALRM, old_handler)

                        tail_n = 4 if has_bounds else 2
                        pred_flat_r = {_norm_tuple(r[:-tail_n]) for r in flat_rows}
                        pred_scores = {}
                        for r in flat_rows:
                            pay = _norm_tuple(r[:-tail_n])
                            sc = max(0, float(r[-(tail_n)] or 0))
                            if pay not in pred_scores or sc > pred_scores[pay]:
                                pred_scores[pay] = sc
                        z_p = sum(pred_scores.values())
                        z_g = max(len(gt_flat), 1)
                        tv_prob_r = 0.0
                        for t in set(pred_scores.keys()) | gt_flat:
                            p_p = pred_scores.get(t, 0) / z_p if z_p > 1e-12 else 0
                            p_g = (1.0 / z_g) if t in gt_flat else 0
                            tv_prob_r += abs(p_p - p_g)
                        tv_prob_r *= 0.5
                        print(" TVprob=%.4f t=%.2fs" % (tv_prob_r, elapsed), flush=True)
                    except Exception as e:
                        tv_prob_r = float("nan")
                        elapsed = 0
                        print(" ERR:%s" % str(e)[:40], flush=True)
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                    results.append({**row_base, "method": "ranking",
                                    "tv_prob": tv_prob_r, "time_s": elapsed})

                    # ── Direct 5.2 ──
                    conn = ensure_conn(conn)
                    print("    direct-5.2...", end="", flush=True)
                    try:
                        conn.rollback()
                        r5 = run_direct_per_tuple(
                            conn, query, table_name, gt_table or table_name,
                            missing_attrs, ordering)
                        if r5.get("error"):
                            tv5 = float("nan")
                            print(" ERR:%s" % str(r5["error"])[:40], flush=True)
                        else:
                            tv5 = r5["tv_prob"]
                            print(" TV=%.4f dw=%.4f t=%.3fs" % (
                                tv5, r5.get("delta_w", 0), r5.get("sql_time_s", 0)), flush=True)
                    except Exception as e:
                        tv5 = float("nan")
                        print(" ERR:%s" % str(e)[:40], flush=True)
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                    results.append({**row_base, "method": "direct-5.2",
                                    "tv_prob": tv5,
                                    "time_s": r5.get("sql_time_s", 0) if not np.isnan(tv5) else 0})

    conn.close()

    df_out = pd.DataFrame(results)
    out_path = "mar_mcar_set_results.csv"
    df_out.to_csv(out_path, index=False)

    print("\n" + "=" * 60, flush=True)
    print("Results saved to %s (%d rows)" % (out_path, len(df_out)), flush=True)
    print("=" * 60, flush=True)

    if len(df_out) > 0:
        good = df_out.dropna(subset=["tv_prob"])
        if len(good) > 0:
            avg = good.groupby(["miss_type", "block", "method"])["tv_prob"].mean()
            print("\nSummary:", flush=True)
            print(avg.to_string(), flush=True)


if __name__ == "__main__":
    main()
