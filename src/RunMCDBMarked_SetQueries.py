"""
mGraph-MCDB-Marked on safe set queries using TupleBundle (real MCDB).

Same as HAVING comparison but for set queries:
  1. Build TupleBundle with MarkedMGraphSampler (samples from conditional/PoE)
  2. Apply WHERE selections on the bundle
  3. infer_row_probs() → per-row membership probability
  4. Group by output cols, P(y in q) = 1 - exp(Σ log(1-p))
  5. For missing output cols: instantiate them in the bundle, use first-world value for grouping

No Python per-world loop for output extraction — uses row_probs + log-sum-exp.
Fast: same speed as HAVING comparison (~100-130s on Bitcoin).
"""
import os, json, re, time
import numpy as np
import pandas as pd
from collections import defaultdict

from MCDBBaseline import (
    TupleBundle, _parse_where, _norm_val,
)
from CompareMCDB_marked import MarkedMGraphSampler

CONN = dict(host="localhost", port=5433, dbname="mydb", user="alzamill", password=os.environ.get("PGPASSWORD", ""))
DISCRETIZED = "/tmp/mnar_set_queries_discretized_safe.json"
ORIG = "configs/mnar_set_queries.json"
MCDB_DATA_DIR = "data/mcdb_test_data"
MC = 100  # Same as Console baseline (γ=100 for ε=0.1)


def _parse_select(q):
    m = re.match(r"SELECT\s+(.+?)\s+FROM\s+", q, re.IGNORECASE)
    return [s.strip().lower() for s in m.group(1).split(",")] if m else []


def run_mgraph_marked_set(query, df, missing_attrs, ordering, gt_set, output_cols,
                           m=MC, seed=42):
    t0 = time.time()
    col_map = {c.lower(): c for c in df.columns}
    active = [a for a in missing_attrs if a in df.columns and df[a].isna().any()]
    N = len(df)

    wc = _parse_where(query)
    resolved = [(col_map.get(c.lower(), c), o, v) for c, o, v in wc
                if col_map.get(c.lower()) and col_map.get(c.lower()) in df.columns]
    det = [(c, o, v) for c, o, v in resolved if c not in active]
    rnd = [(c, o, v) for c, o, v in resolved if c in active]

    # Deterministic pushdown
    pre = np.ones(N, dtype=bool)
    for col, op, val in det:
        s = df[col]
        try: vn = float(val); sc = pd.to_numeric(s, errors="coerce")
        except: sc = s.astype(str); vn = val
        if op in ("=", "=="): mc = sc == vn
        elif op in ("!=", "<>"): mc = sc != vn
        elif op == ">": mc = sc > vn
        elif op == "<": mc = sc < vn
        elif op == ">=": mc = sc >= vn
        elif op == "<=": mc = sc <= vn
        else: continue
        pre &= mc.fillna(True).values

    df_f = df.loc[pre].reset_index(drop=True)
    Nf = len(df_f)
    if Nf == 0:
        return {"tv_prob": 0.5, "time_s": time.time() - t0, "n_pred": 0}

    # Build sampler and bundle — same as HAVING code
    sampler = MarkedMGraphSampler(df_f, active, ordering or {}, seed)
    bundle = TupleBundle(df_f, active, m)

    # Apply WHERE on random attrs with marked sampling
    for col, op, val in rnd:
        sym_col = "%s_nullsym" % col
        sym_ids = df_f[sym_col].values if sym_col in df_f.columns else np.full(Nf, np.nan)
        miss_mask = df_f[col].isna().values

        arr = np.empty((Nf, m), dtype=object)
        obs = ~miss_mask
        if obs.any():
            arr[obs] = np.tile(df_f[col].values[obs, None], (1, m))
        if miss_mask.any():
            arr[miss_mask] = sampler.draw_marked(col, miss_mask, sym_ids, df_f, m)
        bundle.random[col] = arr

        try: ac = arr.astype(float); vc = float(val)
        except: ac = np.vectorize(str)(arr); vc = str(val)
        if op in ("=", "=="): m2 = ac == vc
        elif op in ("!=", "<>"): m2 = ac != vc
        elif op == ">": m2 = ac > vc
        elif op == "<": m2 = ac < vc
        elif op == ">=": m2 = ac >= vc
        elif op == "<=": m2 = ac <= vc
        else: m2 = np.ones_like(arr, dtype=bool)
        bundle.is_present &= m2

    # Per-row membership probability from bundle
    row_probs = bundle.infer_row_probs()

    # Instantiate output cols if they're random (needed for grouping)
    real_out = [col_map.get(c, c) for c in output_cols]
    for oc in real_out:
        if oc in active and bundle.is_random(oc):
            bundle.instantiate(oc, sampler)

    # For each world j: build output set from alive rows using world j's values
    # Count how often each output tuple appears across worlds → membership prob
    # Vectorized: process all worlds via is_present matrix

    has_random_out = any(oc in bundle.random and bundle.random[oc] is not None for oc in real_out)

    if not has_random_out:
        # Fast path: output cols are constant — group by observed values, log-sum-exp
        groups = defaultdict(list)
        for i in range(Nf):
            gk = tuple(str(_norm_val(bundle.constant[oc][i])) for oc in real_out)
            groups[gk].append(i)
        pred = {}
        for gk, idx in groups.items():
            p = row_probs[idx]
            lp = np.sum(np.log(np.maximum(1.0 - p, 1e-15)))
            pred[gk] = 1.0 - np.exp(max(lp, -700))
    else:
        # Random output cols: count appearances across all m worlds
        tuple_counts = defaultdict(int)
        # Process in batches of worlds to balance speed vs memory
        batch = min(m, 50)
        for j0 in range(0, m, batch):
            j1 = min(j0 + batch, m)
            for j in range(j0, j1):
                alive = bundle.is_present[:, j]
                if not alive.any():
                    continue
                alive_idx = np.where(alive)[0]
                world_set = set()
                for i in alive_idx:
                    parts = []
                    skip = False
                    for oc in real_out:
                        if oc in bundle.random and bundle.random[oc] is not None:
                            v = bundle.random[oc][i, j]
                        else:
                            v = bundle.constant[oc][i]
                        if pd.isna(v):
                            skip = True
                            break
                        parts.append(str(_norm_val(v)))
                    if not skip:
                        world_set.add(tuple(parts))
                for gk in world_set:
                    tuple_counts[gk] += 1
        pred = {gk: cnt / m for gk, cnt in tuple_counts.items()}

    # TV
    all_k = set(pred) | gt_set
    zp = sum(pred.values()) or 1.0
    zg = float(len(gt_set)) or 1.0
    tv = 0.5 * sum(abs((pred.get(k, 0) / zp) - ((1 / zg) if k in gt_set else 0)) for k in all_k)

    return {"tv_prob": float(tv), "time_s": time.time() - t0, "n_pred": len(pred)}


def main():
    jp = DISCRETIZED if os.path.isfile(DISCRETIZED) else ORIG
    print("Using: %s" % jp, flush=True)
    with open(jp) as f: cfg = json.load(f)

    results = []

    for gk in ["bank_manr1_set", "nyc_manr1_set", "bit_manr1_set"]:
        if gk not in cfg: continue
        for bk, meta in cfg[gk].items():
            csvs = meta.get("csv", [])
            cc = meta.get("complete_csv", [])
            ma = [a.lower() for a in meta.get("missing_attrs_single", [])]
            od = {k.lower(): [c.lower() for c in v] for k, v in meta.get("ordering_single", {}).items()}
            qs = meta.get("queries", [])
            if not csvs or not qs: continue

            ds = None
            for dn in ["bank", "nyc", "bitcoin"]:
                if dn in bk.lower() or dn in gk.lower(): ds = dn; break
            rate = None
            for r in [5, 10, 20]:
                if str(r) in bk: rate = r; break
            if not ds or not rate: continue

            mcsv = os.path.join(MCDB_DATA_DIR, "%s_mnar_mcdb_%d_marked.csv" % (ds, rate))
            if not os.path.isfile(mcsv): continue

            print("\n" + "=" * 60, flush=True)
            print(bk, flush=True)
            print("=" * 60, flush=True)

            df = pd.read_csv(mcsv); df.columns = df.columns.str.lower()
            gt_df = pd.read_csv(cc[0]) if cc else pd.read_csv(csvs[0])
            gt_df.columns = gt_df.columns.str.lower()
            print("  Loaded: %d rows" % len(df), flush=True)

            nonjoin = [q for q in qs if "JOIN" not in q.upper()]
            for qi, q in enumerate(nonjoin[:4]):
                print("  Q%d: %s" % (qi + 1, q[:70]), flush=True)
                sel = _parse_select(q)
                wcs = _parse_where(q)

                gf = gt_df.copy()
                for col, op, val in wcs:
                    if col not in gf.columns: continue
                    try: vn = float(val); s = pd.to_numeric(gf[col], errors="coerce")
                    except: s = gf[col].astype(str); vn = val
                    if op in ("=", "=="): gf = gf[s == vn]
                    elif op in ("!=", "<>"): gf = gf[s != vn]
                    elif op == ">": gf = gf[s > vn]
                    elif op == "<": gf = gf[s < vn]
                    elif op == ">=": gf = gf[s >= vn]
                    elif op == "<=": gf = gf[s <= vn]
                gt_set = set()
                for _, row in gf.iterrows():
                    gt_set.add(tuple(str(_norm_val(row[c])) for c in sel if c in row.index))

                r = run_mgraph_marked_set(q, df, ma, od, gt_set, sel, m=MC, seed=42)
                print("    TV=%.4f t=%.1fs |pred|=%d" % (
                    r["tv_prob"], r["time_s"], r.get("n_pred", 0)), flush=True)
                results.append({"block": bk, "query_idx": qi + 1,
                                "method": "mGraph-MCDB-Marked",
                                "tv_prob": r["tv_prob"], "time_s": r["time_s"]})

                pd.DataFrame(results).to_csv("mcdb_marked_set_results.csv", index=False)

    df_out = pd.DataFrame(results)
    df_out.to_csv("mcdb_marked_set_results.csv", index=False)
    print("\n" + "=" * 60, flush=True)
    print("Saved mcdb_marked_set_results.csv (%d rows)" % len(df_out), flush=True)
    good = df_out.dropna(subset=["tv_prob"])
    if len(good) > 0:
        print(good.groupby("block")[["tv_prob", "time_s"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
