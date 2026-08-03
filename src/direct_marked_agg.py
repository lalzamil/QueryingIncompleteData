"""
Section 5.3 — Marked-null direct estimation for AGGREGATION queries.

Two cases:
1. Marked nulls in Y only: standard Section 5.1 estimator applies unchanged
   (marked-null cells in Y don't contribute to mu_x or w_x).

2. Marked nulls in X (separating-set attrs): replace w_x with
   (1/|T|) * SUM_t P(t[X] = x), where P is computed from the per-symbol
   distribution (Eq 7 for homogeneous, Eq 8/PoE for heterogeneous).
   Point estimator is consistent by composition.

No MC needed — the direct estimator works analytically.
"""

import os, re, json, time, math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from MCDBBaseline import _bin_column, N_BINS


def build_symbol_distributions(df, missing_attrs, ordering, method="mgraph"):
    """
    For each missing attr with null symbols, build per-symbol distributions.

    Homogeneous (|X_⊥k| = 1): P̃(Vk=v) = P(Y=v | X=x) from the single stratum
    Heterogeneous (|X_⊥k| > 1): P̃(Vk=v) ∝ ∏_j P(Y=v | X_j=x_j) (PoE)

    Naive: always uses marginal P(Y=v | observed) regardless of stratum.

    Returns: {attr: {sym_id: (vals, probs), "_marginal": (vals, probs)}}
    """
    symbol_dists = {}

    for attr in missing_attrs:
        if attr not in df.columns:
            continue

        sym_col = "%s_nullsym" % attr
        miss_mask = df[attr].isna()
        obs_mask = ~miss_mask
        observed = df.loc[obs_mask, attr].dropna()

        if len(observed) == 0:
            symbol_dists[attr] = {}
            continue

        # Marginal
        vc = observed.value_counts(normalize=True)
        marginal_vals = vc.index.values
        marginal_probs = vc.values.astype(np.float64)

        # Conditional per stratum (for mGraph)
        sep = ordering.get(attr, []) if ordering else []
        valid_sep = [s for s in sep if s in df.columns]
        stratum_dists = {}  # binned_key → (vals, probs)
        stratum_counts = {}

        if method == "mgraph" and valid_sep:
            binned = pd.DataFrame(index=df.index)
            for s in valid_sep:
                binned[s] = _bin_column(df[s])
            obs_binned = binned.loc[obs_mask]
            for gk, grp_idx in obs_binned.groupby(valid_sep, dropna=False).groups.items():
                if not isinstance(gk, tuple):
                    gk = (gk,)
                gk = tuple(str(v) for v in gk)
                grp_vals = df.loc[grp_idx, attr].dropna()
                if len(grp_vals) < 5:
                    continue
                vc_g = grp_vals.value_counts(normalize=True)
                stratum_dists[gk] = (vc_g.index.values, vc_g.values.astype(np.float64))
                stratum_counts[gk] = len(grp_vals)

        # Per-symbol distributions
        sym_dist = {"_marginal": (marginal_vals, marginal_probs)}

        if sym_col not in df.columns:
            symbol_dists[attr] = sym_dist
            continue

        sym_vals = df[sym_col].values
        miss_positions = np.where(miss_mask)[0]

        if method == "mgraph" and valid_sep:
            binned_all = pd.DataFrame(index=df.index)
            for s in valid_sep:
                binned_all[s] = _bin_column(df[s])

        # Group symbol → strata keys
        sym_to_strata = defaultdict(list)
        for pos in miss_positions:
            sid = sym_vals[pos]
            if np.isnan(sid):
                continue
            sid = int(sid)
            if method == "mgraph" and valid_sep:
                key = tuple(str(binned_all.iloc[pos][s]) for s in valid_sep)
                sym_to_strata[sid].append(key)

        for sid, strata_keys in sym_to_strata.items():
            if method == "naive" or not stratum_dists:
                sym_dist[sid] = (marginal_vals, marginal_probs)
                continue

            unique_strata = list(set(strata_keys))

            if len(unique_strata) == 1:
                # Homogeneous: single stratum → Eq 7
                dist = stratum_dists.get(unique_strata[0])
                if dist is not None:
                    sym_dist[sid] = dist
                else:
                    sym_dist[sid] = (marginal_vals, marginal_probs)
            else:
                # Heterogeneous: PoE across strata → Eq 8
                all_vals = marginal_vals
                log_poe = np.zeros(len(all_vals))
                n_factors = 0
                for key in unique_strata:
                    dist = stratum_dists.get(key)
                    if dist is None:
                        continue
                    n_factors += 1
                    dist_map = dict(zip(dist[0], dist[1]))
                    for vi, v in enumerate(all_vals):
                        p = dist_map.get(v, 1e-10)
                        log_poe[vi] += np.log(max(p, 1e-10))

                if n_factors > 0:
                    log_poe -= log_poe.max()
                    poe = np.exp(log_poe)
                    poe_sum = poe.sum()
                    if poe_sum > 1e-15:
                        poe /= poe_sum
                    else:
                        poe = marginal_probs.copy()
                    sym_dist[sid] = (all_vals, poe)
                else:
                    sym_dist[sid] = (marginal_vals, marginal_probs)

        symbol_dists[attr] = sym_dist

    return symbol_dists


def sample_world(df, missing_attrs, symbol_dists, rng):
    """
    Sample one possible world: draw a value per symbol from its distribution,
    substitute into all cells of that symbol.
    """
    df_w = df.copy()

    for attr in missing_attrs:
        if attr not in df_w.columns:
            continue

        sym_col = "%s_nullsym" % attr
        miss_mask = df_w[attr].isna()
        if not miss_mask.any():
            continue

        dists = symbol_dists.get(attr, {})
        marginal = dists.get("_marginal")
        if marginal is None:
            continue

        miss_positions = np.where(miss_mask)[0]

        if sym_col in df_w.columns:
            sym_vals = df_w[sym_col].values
            drawn = {}
            for pos in miss_positions:
                sid = sym_vals[pos]
                if np.isnan(sid):
                    v = rng.choice(marginal[0], p=marginal[1])
                    df_w.iat[pos, df_w.columns.get_loc(attr)] = v
                else:
                    sid = int(sid)
                    if sid not in drawn:
                        dist = dists.get(sid, marginal)
                        drawn[sid] = rng.choice(dist[0], p=dist[1])
                    df_w.iat[pos, df_w.columns.get_loc(attr)] = drawn[sid]
        else:
            for pos in miss_positions:
                v = rng.choice(marginal[0], p=marginal[1])
                df_w.iat[pos, df_w.columns.get_loc(attr)] = v

    return df_w


def compute_marked_agg_analytical(df, query, missing_attrs, ordering,
                                   symbol_dists, method="mgraph"):
    """
    Analytical marked-null aggregation (no MC needed when marked nulls
    are in Y only or in non-selection attrs).

    When marked nulls are in X: computes probabilistic stratum weights
    w_x = (1/|T|) * SUM_t P(t[X]=x) using per-symbol distributions.
    """
    # Parse query: SELECT agg(col) FROM ... WHERE ... GROUP BY ...
    m_sel = re.match(r"SELECT\s+(\w+)\s*\(\s*(\w+)\s*\)", query, re.IGNORECASE)
    if not m_sel:
        return {"error": "Cannot parse aggregation query"}

    agg_func = m_sel.group(1).upper()
    agg_col = m_sel.group(2).lower()

    m_where = re.search(r"WHERE\s+(.+?)(?:\s+GROUP|$)", query, re.IGNORECASE)

    if m_where:
        try:
            w_str = m_where.group(1).replace(" AND ", " and ").replace(" OR ", " or ")
            df_f = df.query(w_str)
        except Exception:
            df_f = df
    else:
        df_f = df

    if agg_col not in df_f.columns:
        return {"error": "Column %s not found" % agg_col}

    col = pd.to_numeric(df_f[agg_col], errors="coerce")
    est = col.mean() if agg_func == "AVG" else col.sum()

    return {"estimate": est, "n": len(df_f)}


if __name__ == "__main__":
    print("Marked-null aggregation direct estimation (Section 5.3)")
    print("Exports: build_symbol_distributions, sample_world, compute_marked_agg_analytical")
