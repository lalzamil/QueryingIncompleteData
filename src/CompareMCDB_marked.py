"""
Compare Naive-MCDB-Marked vs mGraph-MCDB-Marked on unsafe HAVING queries.

Both methods operate under marked-null semantics: each null symbol ⊥_k
must take a single value across all cells it occupies.

Naive-Marked: draws from marginal P(Y | observed), one draw per symbol.
mGraph-Marked: draws from PoE ∏_j P(Y=v | X_j=x_j), one draw per symbol.
"""

import os, json, re, time, math
import numpy as np
import pandas as pd
import psycopg2
from io import StringIO
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from MCDBBaseline import (
    _bin_column, _parse_where, _norm_val, _Z_95, N_BINS,
    TupleBundle, NaiveSampler, MGraphSampler, TIMEOUT_S,
    _compute_row_survival_prob,
)

CONN_PARAMS = dict(host="localhost", port=5433, dbname="mydb",
                   user="alzamill", password=os.environ.get("PGPASSWORD", ""))

JSON_PATH = "configs/unsafe_mnar_set_queries.json"
MCDB_DATA_DIR = "data/mcdb_test_data"
MC_SAMPLES = 738

DATASETS_TO_RUN = [
    "bank_manr1_set",
    "nyc_manr1_set",
    "bit_manr1_set",
]


# ═══════════════════════════════════════════════════════════════
#  Marked-null samplers
# ═══════════════════════════════════════════════════════════════

class MarkedNaiveSampler(NaiveSampler):
    """
    Marked-null variant of NaiveSampler.
    For each symbol ⊥_k, draws ONE value from marginal and replicates
    across all cells sharing that symbol.
    """

    def draw_marked(self, attr: str, mask: np.ndarray, symbol_ids: np.ndarray,
                    m: int) -> np.ndarray:
        """
        mask: (N,) bool — which rows are missing for this attr
        symbol_ids: (N,) float — symbol ID per row (NaN for observed)
        Returns (n_miss, m) array with marked-null constraint.
        """
        vals, probs = self._marginals.get(attr, (np.array([]), np.array([])))
        n_miss = int(mask.sum())
        if len(vals) == 0 or n_miss == 0:
            return np.full((n_miss, m), np.nan, dtype=object)

        miss_positions = np.where(mask)[0]
        miss_syms = symbol_ids[miss_positions]

        result = np.empty((n_miss, m), dtype=object)

        unique_syms = np.unique(miss_syms[~np.isnan(miss_syms)])
        # One draw per symbol per world
        sym_draws = {}
        for sym in unique_syms:
            sym_draws[sym] = self.rng.choice(vals, size=m, p=probs)

        for i, pos in enumerate(miss_positions):
            sym = miss_syms[i]
            if np.isnan(sym):
                result[i, :] = self.rng.choice(vals, size=m, p=probs)
            else:
                result[i, :] = sym_draws[sym]

        return result


class MarkedMGraphSampler(MGraphSampler):
    """
    Marked-null variant with Product-of-Experts.
    For each symbol ⊥_k spanning strata {x_1,...,x_J}, computes:
      P̃(v) ∝ ∏_j P(Y=v | X_j=x_j)
    and draws ONE value from P̃.
    """

    def draw_marked(self, attr: str, mask: np.ndarray, symbol_ids: np.ndarray,
                    df: pd.DataFrame, m: int) -> np.ndarray:
        n_miss = int(mask.sum())
        vals_fb, probs_fb = self._fallback.get(attr, (np.array([]), np.array([])))
        if n_miss == 0:
            return np.empty((0, m), dtype=object)

        cond = self._conditionals.get(attr, {})
        sep = self._sep_cols.get(attr, [])

        miss_positions = np.where(mask)[0]
        miss_syms = symbol_ids[miss_positions]

        # Build stratum key for each missing row
        if sep and attr in self._binned:
            binned_df = self._binned[attr]
            miss_keys = []
            for pos in miss_positions:
                key = tuple(str(binned_df.iloc[pos][s]) for s in sep)
                miss_keys.append(key)
        else:
            miss_keys = [("__global__",)] * n_miss

        result = np.empty((n_miss, m), dtype=object)

        # Group by symbol
        sym_to_rows = defaultdict(list)
        for i, pos in enumerate(miss_positions):
            sym = miss_syms[i]
            if np.isnan(sym):
                sym_to_rows[("_indep_", i)].append(i)
            else:
                sym_to_rows[sym].append(i)

        # Get all unique values across all conditionals for PoE domain
        all_vals = set()
        for dist_vals, _ in cond.values():
            all_vals.update(dist_vals)
        if vals_fb is not None and len(vals_fb) > 0:
            all_vals.update(vals_fb)
        all_vals = sorted(all_vals, key=str)

        if not all_vals:
            result[:, :] = np.nan
            return result

        val_to_idx = {v: i for i, v in enumerate(all_vals)}
        n_vals = len(all_vals)

        for sym, row_indices in sym_to_rows.items():
            if isinstance(sym, tuple) and sym[0] == "_indep_":
                # Independent row (no symbol) — use per-stratum conditional
                i = row_indices[0]
                key = miss_keys[i]
                dist = cond.get(key)
                if dist is None or len(dist[0]) == 0:
                    dist = (vals_fb, probs_fb) if vals_fb is not None and len(vals_fb) > 0 else None
                if dist is None:
                    result[i, :] = np.nan
                else:
                    result[i, :] = self.rng.choice(dist[0], size=m, p=dist[1])
                continue

            # Collect strata for this symbol's rows
            strata_keys = [miss_keys[i] for i in row_indices]

            # Product of Experts across strata
            log_poe = np.zeros(n_vals)
            for key in strata_keys:
                dist = cond.get(key)
                if dist is None or len(dist[0]) == 0:
                    # Fallback: use marginal (contributes uniform-ish factor)
                    if vals_fb is not None and len(vals_fb) > 0:
                        dist = (vals_fb, probs_fb)
                    else:
                        continue
                # Map dist to the full domain
                dist_map = dict(zip(dist[0], dist[1]))
                for vi, v in enumerate(all_vals):
                    p = dist_map.get(v, 1e-10)
                    log_poe[vi] += np.log(max(p, 1e-10))

            # Normalize
            log_poe -= log_poe.max()
            poe = np.exp(log_poe)
            poe_sum = poe.sum()
            if poe_sum < 1e-15:
                poe = np.ones(n_vals) / n_vals
            else:
                poe /= poe_sum

            # Draw m values from PoE
            draws = self.rng.choice(all_vals, size=m, p=poe)

            # Assign same draw to all rows of this symbol
            for i in row_indices:
                result[i, :] = draws

        return result


# ═══════════════════════════════════════════════════════════════
#  Marked-null HAVING evaluation
# ═══════════════════════════════════════════════════════════════

def run_mcdb_marked_having(query: str, df_incomplete: pd.DataFrame,
                           missing_attrs: List[str],
                           ordering: Optional[Dict[str, List[str]]],
                           gt_groups_set: set,
                           group_cols: List[str],
                           having_threshold: int,
                           m: int = 738,
                           method: str = "mgraph",
                           seed: int = 42) -> Dict[str, Any]:
    """
    Marked-null HAVING evaluation via Bernoulli simulation.
    For each missing attr, uses marked samplers that respect symbol constraints.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    col_map = {c.lower(): c for c in df_incomplete.columns}
    active_missing = [a for a in missing_attrs
                      if a in df_incomplete.columns and df_incomplete[a].isna().any()]

    where_clauses = _parse_where(query)
    resolved_where = []
    for col_raw, op, val in where_clauses:
        col_name = col_map.get(col_raw.lower())
        if col_name and col_name in df_incomplete.columns:
            resolved_where.append((col_name, op, val))

    det_clauses = [(c, o, v) for c, o, v in resolved_where if c not in active_missing]
    rnd_clauses = [(c, o, v) for c, o, v in resolved_where if c in active_missing]

    # Deterministic pushdown
    pre_mask = np.ones(len(df_incomplete), dtype=bool)
    for col, op, val in det_clauses:
        series = df_incomplete[col]
        try:
            val_n = float(val)
            series_cmp = pd.to_numeric(series, errors="coerce")
        except (ValueError, TypeError):
            series_cmp = series.astype(str)
            val_n = val
        if op in ("=", "=="):    mc = series_cmp == val_n
        elif op in ("!=", "<>"): mc = series_cmp != val_n
        elif op == ">":          mc = series_cmp > val_n
        elif op == "<":          mc = series_cmp < val_n
        elif op == ">=":         mc = series_cmp >= val_n
        elif op == "<=":         mc = series_cmp <= val_n
        else: continue
        pre_mask &= mc.fillna(True).values

    df_filtered = df_incomplete.loc[pre_mask].reset_index(drop=True)
    N = len(df_filtered)

    real_grp = [col_map.get(c.lower(), c) for c in group_cols]
    grp_has_missing = any(gc in active_missing for gc in real_grp)

    if grp_has_missing:
        # Full bundle path for missing group cols
        if method == "naive":
            sampler = MarkedNaiveSampler(df_filtered, active_missing, seed)
        else:
            sampler = MarkedMGraphSampler(df_filtered, active_missing,
                                          ordering or {}, seed)
        bundle = TupleBundle(df_filtered, active_missing, m)

        for col, op, val in rnd_clauses:
            attr = col
            sym_col = "%s_nullsym" % attr
            if sym_col in df_filtered.columns:
                sym_ids = df_filtered[sym_col].values
            else:
                sym_ids = np.full(N, np.nan)

            miss_mask = df_filtered[attr].isna().values
            if bundle.is_random(attr):
                # Marked draw
                if method == "naive":
                    arr = np.empty((N, m), dtype=object)
                    obs_mask = ~miss_mask
                    arr[obs_mask] = np.tile(df_filtered[attr].values[obs_mask, None], (1, m))
                    if miss_mask.any():
                        arr[miss_mask] = sampler.draw_marked(attr, miss_mask, sym_ids, m)
                    bundle.random[attr] = arr
                else:
                    arr = np.empty((N, m), dtype=object)
                    obs_mask = ~miss_mask
                    arr[obs_mask] = np.tile(df_filtered[attr].values[obs_mask, None], (1, m))
                    if miss_mask.any():
                        arr[miss_mask] = sampler.draw_marked(attr, miss_mask, sym_ids, df_filtered, m)
                    bundle.random[attr] = arr

                # Apply selection
                try:
                    arr_cmp = arr.astype(float)
                    val_cmp = float(val)
                except (ValueError, TypeError):
                    arr_cmp = np.vectorize(str)(arr)
                    val_cmp = str(val)
                if op in ("=", "=="):    mask_2d = arr_cmp == val_cmp
                elif op in ("!=", "<>"): mask_2d = arr_cmp != val_cmp
                elif op == ">":          mask_2d = arr_cmp > val_cmp
                elif op == "<":          mask_2d = arr_cmp < val_cmp
                elif op == ">=":         mask_2d = arr_cmp >= val_cmp
                elif op == "<=":         mask_2d = arr_cmp <= val_cmp
                else: mask_2d = np.ones_like(arr, dtype=bool)
                bundle.is_present &= mask_2d

        # Instantiate group cols
        for gc in real_grp:
            if gc in active_missing and bundle.is_random(gc):
                sym_col = "%s_nullsym" % gc
                sym_ids = df_filtered[sym_col].values if sym_col in df_filtered.columns else np.full(N, np.nan)
                miss_mask = df_filtered[gc].isna().values
                if method == "naive":
                    arr = np.empty((N, m), dtype=object)
                    obs = ~miss_mask
                    arr[obs] = np.tile(df_filtered[gc].values[obs, None], (1, m))
                    if miss_mask.any():
                        arr[miss_mask] = sampler.draw_marked(gc, miss_mask, sym_ids, m)
                    bundle.random[gc] = arr
                else:
                    arr = np.empty((N, m), dtype=object)
                    obs = ~miss_mask
                    arr[obs] = np.tile(df_filtered[gc].values[obs, None], (1, m))
                    if miss_mask.any():
                        arr[miss_mask] = sampler.draw_marked(gc, miss_mask, sym_ids, df_filtered, m)
                    bundle.random[gc] = arr

        group_counts: Dict[tuple, int] = defaultdict(int)
        group_probs = {}
        group_ci = {}
        for j in range(m):
            alive = bundle.is_present[:, j]
            if not alive.any():
                continue
            keys = []
            for gc in real_grp:
                if gc in bundle.random and bundle.random[gc] is not None:
                    keys.append(bundle.random[gc][alive, j])
                else:
                    keys.append(bundle.constant[gc][alive])
            n_alive = int(alive.sum())
            world_groups: Dict[tuple, int] = defaultdict(int)
            for ri in range(n_alive):
                gk = tuple(_norm_val(keys[gi][ri]) for gi in range(len(real_grp)))
                world_groups[gk] += 1
            for gk, cnt in world_groups.items():
                if cnt > having_threshold:
                    group_counts[gk] += 1
        for gk, cnt in group_counts.items():
            p = cnt / m
            group_probs[gk] = p
            group_ci[gk] = 2 * _Z_95 * math.sqrt(p * (1 - p) / m)
    else:
        # Marked Bernoulli path with PoE-based symbol probabilities.
        # For each symbol ⊥_k: compute P(⊥_k satisfies pred) using PoE
        # (mGraph) or marginal (naive), then one Bernoulli per symbol.

        # Step 1: Build per-attr conditional/marginal distributions for predicate eval
        # and compute per-row base probabilities for observed (non-null) rows
        row_probs = np.ones(N, dtype=np.float64)

        for col, op, val in rnd_clauses:
            miss_mask = df_filtered[col].isna().values
            obs_mask = ~miss_mask

            # Observed rows: deterministic 0 or 1
            if obs_mask.any():
                series = df_filtered.loc[obs_mask, col]
                try:
                    val_n = float(val)
                    series_cmp = pd.to_numeric(series, errors="coerce")
                except (ValueError, TypeError):
                    series_cmp = series.astype(str)
                    val_n = val
                if op in ("=", "=="):    ok = series_cmp == val_n
                elif op in ("!=", "<>"): ok = series_cmp != val_n
                elif op == ">":          ok = series_cmp > val_n
                elif op == "<":          ok = series_cmp < val_n
                elif op == ">=":         ok = series_cmp >= val_n
                elif op == "<=":         ok = series_cmp <= val_n
                else:                    ok = pd.Series(True, index=series.index)
                row_probs[obs_mask] *= ok.fillna(False).values.astype(np.float64)

            if not miss_mask.any():
                continue

            # Build value domain and predicate satisfaction per value
            observed = df_filtered.loc[obs_mask, col].dropna()
            if len(observed) == 0:
                row_probs[miss_mask] = 0.0
                continue

            all_vals = observed.unique()
            try:
                val_n = float(val)
                all_vals_num = pd.to_numeric(pd.Series(all_vals), errors="coerce")
                if op in ("=", "=="):    sat = (all_vals_num == val_n).values
                elif op in ("!=", "<>"): sat = (all_vals_num != val_n).values
                elif op == ">":          sat = (all_vals_num > val_n).values
                elif op == "<":          sat = (all_vals_num < val_n).values
                elif op == ">=":         sat = (all_vals_num >= val_n).values
                elif op == "<=":         sat = (all_vals_num <= val_n).values
                else:                    sat = np.ones(len(all_vals), dtype=bool)
            except (ValueError, TypeError):
                all_vals_str = np.array([str(v) for v in all_vals])
                if op in ("=", "=="):    sat = all_vals_str == str(val)
                elif op in ("!=", "<>"): sat = all_vals_str != str(val)
                else:                    sat = np.ones(len(all_vals), dtype=bool)

            sat = sat.astype(np.float64)
            val_to_sat = dict(zip(all_vals, sat))

            # Marginal distribution
            vc = observed.value_counts(normalize=True)
            marginal_vals = vc.index.values
            marginal_probs = vc.values.astype(np.float64)
            marginal_p_sat = sum(val_to_sat.get(v, 0.0) * p
                                 for v, p in zip(marginal_vals, marginal_probs))

            # Per-stratum conditionals (for mGraph)
            sep = ordering.get(col, []) if ordering else []
            valid_sep = [s for s in sep if s in df_filtered.columns]
            stratum_cond = {}  # binned_key → {val: prob}
            if method == "mgraph" and valid_sep:
                binned = pd.DataFrame(index=df_filtered.index)
                for s in valid_sep:
                    binned[s] = _bin_column(df_filtered[s])
                obs_binned = binned.loc[obs_mask]
                obs_df_tmp = df_filtered.loc[obs_mask].copy()
                obs_df_tmp["_bk"] = obs_binned[valid_sep].apply(
                    lambda r: tuple(str(v) for v in r), axis=1)
                for bk, grp in obs_df_tmp.groupby("_bk"):
                    grp_vals = grp[col].dropna()
                    if len(grp_vals) < 5:
                        continue
                    vc_g = grp_vals.value_counts(normalize=True)
                    stratum_cond[bk] = dict(zip(vc_g.index.values, vc_g.values))

            # Get symbol assignments
            sym_col_name = "%s_nullsym" % col
            sym_ids = df_filtered[sym_col_name].values if sym_col_name in df_filtered.columns else np.full(N, np.nan)

            # Group missing rows by symbol
            miss_positions = np.where(miss_mask)[0]
            sym_to_miss = defaultdict(list)
            indep_miss = []
            for pos in miss_positions:
                sid = sym_ids[pos]
                if np.isnan(sid):
                    indep_miss.append(pos)
                else:
                    sym_to_miss[int(sid)].append(pos)

            # Independent missing rows: use per-row conditional/marginal
            for pos in indep_miss:
                if method == "mgraph" and valid_sep:
                    bk = tuple(str(binned.iloc[pos][s]) for s in valid_sep)
                    cd = stratum_cond.get(bk)
                    if cd:
                        p = sum(val_to_sat.get(v, 0.0) * cd.get(v, 0.0) for v in cd)
                        row_probs[pos] *= p
                        continue
                row_probs[pos] *= marginal_p_sat

            # Symbol groups: compute PoE-based P(satisfies pred) per symbol
            for sid, positions in sym_to_miss.items():
                if method == "naive" or not valid_sep or not stratum_cond:
                    # Naive: marginal-based probability, same for all in symbol
                    for pos in positions:
                        row_probs[pos] *= marginal_p_sat
                else:
                    # mGraph: PoE across strata of this symbol's rows
                    strata_keys = []
                    for pos in positions:
                        bk = tuple(str(binned.iloc[pos][s]) for s in valid_sep)
                        strata_keys.append(bk)

                    # Compute PoE: P̃(v) ∝ ∏_j P(v | X_j=x_j)
                    log_poe = np.zeros(len(all_vals))
                    n_factors = 0
                    for bk in strata_keys:
                        cd = stratum_cond.get(bk)
                        if cd is None:
                            continue
                        n_factors += 1
                        for vi, v in enumerate(all_vals):
                            p_v = cd.get(v, 1e-10)
                            log_poe[vi] += np.log(max(p_v, 1e-10))

                    if n_factors > 0:
                        log_poe -= log_poe.max()
                        poe = np.exp(log_poe)
                        poe /= poe.sum() + 1e-15
                        p_sat = sum(sat[vi] * poe[vi] for vi in range(len(all_vals)))
                    else:
                        p_sat = marginal_p_sat

                    for pos in positions:
                        row_probs[pos] *= p_sat

        # Assign rows to groups
        unique_keys = {}
        row_gid = np.empty(N, dtype=np.int32)
        for i in range(N):
            gk = tuple(_norm_val(df_filtered[gc].iloc[i]) for gc in real_grp)
            if gk not in unique_keys:
                unique_keys[gk] = len(unique_keys)
            row_gid[i] = unique_keys[gk]

        n_groups = len(unique_keys)
        gid_to_key = {v: k for k, v in unique_keys.items()}

        # Simulate m worlds: per-symbol correlated Bernoulli draws
        survives = np.zeros((N, m), dtype=bool)
        det_mask = row_probs >= 1.0
        survives[det_mask, :] = True

        # Collect all symbol groups for correlated draws
        all_sym_groups = defaultdict(list)
        for col, op, val in rnd_clauses:
            sym_col_name = "%s_nullsym" % col
            if sym_col_name in df_filtered.columns:
                sv = df_filtered[sym_col_name].values
                for i in range(N):
                    if not np.isnan(sv[i]):
                        all_sym_groups[(col, int(sv[i]))].append(i)

        # Independent stochastic rows: draw independently
        in_sym = set()
        for indices in all_sym_groups.values():
            in_sym.update(indices)
        indep_stoch = [i for i in range(N) if row_probs[i] > 0 and row_probs[i] < 1 and i not in in_sym]
        if indep_stoch:
            idx_arr = np.array(indep_stoch)
            survives[idx_arr, :] = rng.random((len(idx_arr), m)) < row_probs[idx_arr, None]

        # Symbol groups: ONE draw per symbol, shared across all rows
        for (attr, sid), indices in all_sym_groups.items():
            p = row_probs[indices[0]]  # all rows in symbol have same prob (set above)
            group_draw = rng.random(m) < p
            for idx in indices:
                survives[idx, :] = group_draw

        # Count per output group per world
        group_probs = {}
        group_ci = {}
        for gid in range(n_groups):
            mask_g = row_gid == gid
            counts_per_world = survives[mask_g, :].sum(axis=0)
            survive_count = int((counts_per_world > having_threshold).sum())
            if survive_count > 0:
                p = survive_count / m
                gk = gid_to_key[gid]
                group_probs[gk] = p
                group_ci[gk] = 2 * _Z_95 * math.sqrt(p * (1 - p) / m)

    elapsed = time.time() - t0

    all_groups = set(group_probs.keys()) | gt_groups_set
    z_pred = sum(group_probs.values()) if group_probs else 0.0
    z_gt = float(len(gt_groups_set)) if gt_groups_set else 0.0
    tv = 0.0
    for g in all_groups:
        p_p = (group_probs.get(g, 0.0) / z_pred) if z_pred > 0 else 0.0
        p_g = (1.0 / z_gt) if (g in gt_groups_set and z_gt > 0) else 0.0
        tv += abs(p_p - p_g)
    tv /= 2.0

    return {
        "tv_prob": tv,
        "time_s": elapsed,
        "m": m,
        "n_gt": len(gt_groups_set),
        "n_pred": len(group_probs),
    }


# ═══════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════

def _load_table(conn, csv_path, table_name):
    cur = conn.cursor()
    cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=%s)",
                (table_name,))
    if cur.fetchone()[0]:
        return
    df = pd.read_csv(csv_path)
    cols = []
    for c in df.columns:
        if df[c].dtype.kind in ("f", "i"):
            cols.append('"%s" DOUBLE PRECISION' % c)
        else:
            cols.append('"%s" TEXT' % c)
    cur.execute('DROP TABLE IF EXISTS "%s"' % table_name)
    cur.execute('CREATE TABLE "%s" (%s)' % (table_name, ", ".join(cols)))
    buf = StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_from(buf, table_name, sep=",", null="\\N", columns=list(df.columns))
    conn.commit()
    cur.execute('ANALYZE "%s"' % table_name)
    conn.commit()


def main():
    conn = psycopg2.connect(**CONN_PARAMS)
    conn.autocommit = False

    with open(JSON_PATH) as f:
        cfg = json.load(f)

    results = []

    for group_key in DATASETS_TO_RUN:
        if group_key not in cfg:
            continue

        blocks = cfg[group_key]
        for block_key, meta in blocks.items():
            csvs = meta.get("csv", [])
            complete_csvs = meta.get("complete_csv", [])
            complete_tables = meta.get("complete_table", [])
            missing_attrs = meta.get("missing_attrs_single", [])
            ordering = meta.get("ordering_single", {})
            queries = meta.get("queries", [])

            if not csvs or not queries:
                continue

            # Determine the marked CSV path
            csv_path = csvs[0]
            ds_name = None
            for dn in ["bank", "nyc", "bitcoin"]:
                if dn in csv_path.lower():
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
                print("Skipping %s: %s not found" % (block_key, marked_csv))
                continue

            gt_csv = complete_csvs[0] if complete_csvs else None
            gt_table = complete_tables[0] if complete_tables else None

            print("\n" + "=" * 60, flush=True)
            print("Block: %s / %s" % (group_key, block_key), flush=True)
            print("=" * 60, flush=True)

            try:
                conn.rollback()
            except Exception:
                pass

            if gt_csv and gt_table:
                _load_table(conn, gt_csv, gt_table)

            df_incomplete = pd.read_csv(marked_csv)
            print("  Loaded: %s (%d rows)" % (os.path.basename(marked_csv), len(df_incomplete)),
                  flush=True)

            for qi, query in enumerate(queries):
                print("\n  Q%d/%d: %s" % (qi + 1, len(queries), query[:95]),
                      flush=True)

                row_base = {
                    "group": group_key, "block": block_key,
                    "query_idx": qi + 1, "query": query[:120],
                }

                hav_m = re.search(r"HAVING\s+count\s*\(\s*\*\s*\)\s*>\s*(\d+)", query, re.IGNORECASE)
                having_threshold = int(hav_m.group(1)) if hav_m else 0

                gb_m = re.search(r"GROUP\s+BY\s+(.+?)(?:\s+HAVING|$)", query, re.IGNORECASE)
                group_cols_parsed = [c.strip().lower() for c in gb_m.group(1).split(",")] if gb_m else []

                gt_groups_set = set()
                if gt_csv and gt_table:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        wm = re.search(r"WHERE\s+(.+?)(?:\s+GROUP\s+BY|$)", query, re.IGNORECASE)
                        w_str = wm.group(1).strip() if wm else "TRUE"
                        gk_sql = ", ".join('"%s"' % c for c in group_cols_parsed)
                        gt_sql = 'SELECT %s, COUNT(*) AS cnt FROM %s WHERE %s GROUP BY %s HAVING COUNT(*) > %d' % (
                            gk_sql, gt_table, w_str, gk_sql, having_threshold)
                        cur = conn.cursor()
                        cur.execute(gt_sql)
                        for r in cur.fetchall():
                            gt_groups_set.add(tuple(str(_norm_val(r[i])) for i in range(len(group_cols_parsed))))
                        conn.commit()
                    except Exception:
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                for mc_method, mc_label in [("naive", "Naive-MCDB-Marked"),
                                             ("mgraph", "mGraph-MCDB-Marked")]:
                    print("    %s..." % mc_label, end="", flush=True)

                    result = run_mcdb_marked_having(
                        query, df_incomplete, missing_attrs,
                        ordering=ordering if mc_method == "mgraph" else None,
                        gt_groups_set=gt_groups_set,
                        group_cols=group_cols_parsed,
                        having_threshold=having_threshold,
                        m=MC_SAMPLES, method=mc_method, seed=42,
                    )
                    print(" TV=%.4f  t=%.1fs  |GT|=%d |Pred|=%d" % (
                        result["tv_prob"], result["time_s"],
                        result.get("n_gt", 0), result.get("n_pred", 0)),
                        flush=True)
                    results.append({
                        **row_base, "method": mc_label,
                        "tv_prob": result["tv_prob"],
                        "time_s": result["time_s"],
                        "worlds": result["m"],
                    })

    conn.close()

    df_out = pd.DataFrame(results)
    out_path = "mcdb_marked_results.csv"
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
