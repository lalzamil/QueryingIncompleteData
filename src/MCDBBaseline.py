"""
Monte Carlo baselines for query evaluation on incomplete data,
implementing the MCDB tuple-bundle execution model.

Naive-MCDB:  samples missing cells from marginal P(Y | Y observed).
             Uses tuple bundles, isPresent bit matrix, lazy instantiation,
             and deterministic predicate pushdown.

mGraph-MCDB: inherits all Naive-MCDB machinery.  Overrides the sampling
             distribution to P(Y | X = t[X]) via ordered separating sets
             (Algorithm 1).  Adds per-pattern batching: rows with the same
             missingness vector m(t) and binned X=x share a single draw.

Reference:  Jampani et al., "MCDB: A Monte Carlo Approach to Managing
            Uncertain Data", SIGMOD 2008.
"""

import re, time, math, signal, warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from io import StringIO

warnings.filterwarnings("ignore", category=FutureWarning)

_Z_95 = 1.96
TIMEOUT_S = 300
N_BINS = 20


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


def _norm_tuple(t):
    return tuple(_norm_val(v) for v in t)


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("timeout")


# ═══════════════════════════════════════════════════════════════════
#  TupleBundle  — faithful analog of MCDB Section 7
# ═══════════════════════════════════════════════════════════════════

class TupleBundle:
    """
    Represents N rows across m Monte Carlo worlds.

    constant[attr]  : np.array(N,)       — observed attrs, shared across worlds
    random[attr]    : np.array(N, m) | None — missing attrs, lazy-instantiated
    is_present      : np.array(N, m) bool — which rows survive in each world
    missing_mask[attr]: np.array(N,) bool — which rows have missing values
    """

    def __init__(self, df: pd.DataFrame, missing_attrs: List[str], m: int):
        self.n_rows = len(df)
        self.m = m
        self.columns = list(df.columns)

        self.constant: Dict[str, np.ndarray] = {}
        self.random: Dict[str, Optional[np.ndarray]] = {}
        self.missing_mask: Dict[str, np.ndarray] = {}
        self.is_present = np.ones((self.n_rows, m), dtype=bool)

        for col in df.columns:
            vals = df[col].values.copy()
            if col in missing_attrs and df[col].isna().any():
                self.missing_mask[col] = df[col].isna().values
                self.constant[col] = vals
                self.random[col] = None      # lazy — not yet instantiated
            else:
                self.constant[col] = vals     # constant attribute

    def instantiate(self, attr: str, sampler: "BaseSampler"):
        """Late materialization (MCDB Section 6.2).
        Only called when an operator actually needs this random attribute."""
        if attr not in self.random or self.random[attr] is not None:
            return
        mask = self.missing_mask[attr]
        n_miss = int(mask.sum())
        arr = np.empty((self.n_rows, self.m), dtype=object)
        obs_vals = self.constant[attr][~mask]
        arr[~mask, :] = obs_vals[:, np.newaxis]
        if n_miss > 0:
            arr[mask, :] = sampler.draw(attr, mask, self.m)
        self.random[attr] = arr

    def get_column(self, attr: str) -> np.ndarray:
        """Return (N,) for constant or (N,m) for instantiated random."""
        if attr in self.random and self.random[attr] is not None:
            return self.random[attr]
        return self.constant.get(attr)

    def is_random(self, attr: str) -> bool:
        return attr in self.random

    # ── Selection operators ──

    def select_constant(self, attr: str, op: str, val):
        """Selection on a constant attribute — single comparison, broadcast.
        Updates is_present without instantiation."""
        col = self.constant[attr]
        try:
            col_cmp = col.astype(float)
            val_cmp = float(val)
        except (ValueError, TypeError):
            col_cmp = np.array([str(v) for v in col])
            val_cmp = str(val)

        if op in ("=", "=="):    mask_1d = col_cmp == val_cmp
        elif op in ("!=", "<>"): mask_1d = col_cmp != val_cmp
        elif op == ">":          mask_1d = col_cmp > val_cmp
        elif op == "<":          mask_1d = col_cmp < val_cmp
        elif op == ">=":         mask_1d = col_cmp >= val_cmp
        elif op == "<=":         mask_1d = col_cmp <= val_cmp
        else: return

        nan_mask = pd.isna(col)
        mask_1d[nan_mask] = True
        self.is_present &= mask_1d[:, np.newaxis]

    def select_random(self, attr: str, op: str, val, sampler: "BaseSampler"):
        """Selection on a random attribute — requires instantiation first.
        Element-wise comparison over (N, m)."""
        self.instantiate(attr, sampler)
        arr = self.random[attr]  # (N, m)
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
        else: return

        self.is_present &= mask_2d

    # ── Inference operator (MCDB Section 8.4) ──

    def infer_row_probs(self) -> np.ndarray:
        """p_hat[i] = fraction of worlds where row i is present."""
        return self.is_present.sum(axis=1) / self.m

    def infer_ci_half(self) -> np.ndarray:
        p = self.infer_row_probs()
        return _Z_95 * np.sqrt(p * (1 - p) / self.m)


# ═══════════════════════════════════════════════════════════════════
#  Samplers
# ═══════════════════════════════════════════════════════════════════

class BaseSampler:
    def __init__(self, df: pd.DataFrame, missing_attrs: List[str], seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.missing_attrs = missing_attrs

    def draw(self, attr: str, mask: np.ndarray, m: int) -> np.ndarray:
        raise NotImplementedError


class NaiveSampler(BaseSampler):
    """Samples from marginal P(Y | R_Y=0) — ignores missingness mechanism."""

    def __init__(self, df: pd.DataFrame, missing_attrs: List[str], seed: int = 42):
        super().__init__(df, missing_attrs, seed)
        self._marginals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for attr in missing_attrs:
            if attr not in df.columns:
                continue
            observed = df[attr].dropna()
            if len(observed) == 0:
                self._marginals[attr] = (np.array([]), np.array([]))
                continue
            vc = observed.value_counts(normalize=True)
            self._marginals[attr] = (vc.index.values, vc.values.astype(np.float64))

    def draw(self, attr: str, mask: np.ndarray, m: int) -> np.ndarray:
        """Return (n_miss, m) array drawn iid from marginal."""
        vals, probs = self._marginals.get(attr, (np.array([]), np.array([])))
        n_miss = int(mask.sum())
        if len(vals) == 0:
            return np.full((n_miss, m), np.nan, dtype=object)
        indices = self.rng.choice(len(vals), size=(n_miss, m), p=probs)
        return vals[indices]


def _bin_column(series: pd.Series, n_bins: int = N_BINS) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0 or numeric.nunique() <= n_bins:
            return series.astype(str)
        return pd.qcut(numeric, q=n_bins, duplicates="drop").astype(str)
    except (ValueError, TypeError):
        return series.astype(str)


class MGraphSampler(BaseSampler):
    """Samples from P(Y | X = t[X]) via ordered separating sets.
    Uses per-pattern batching: rows with same (m(t), X_binned=x) share one draw."""

    def __init__(self, df: pd.DataFrame, missing_attrs: List[str],
                 ordering: Dict[str, List[str]], seed: int = 42):
        super().__init__(df, missing_attrs, seed)
        self.ordering = ordering
        self._conditionals: Dict[str, Dict[tuple, Tuple[np.ndarray, np.ndarray]]] = {}
        self._fallback: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._sep_cols: Dict[str, List[str]] = {}
        self._binned: Dict[str, pd.DataFrame] = {}
        self._pattern_groups: Dict[str, List[Tuple[tuple, np.ndarray]]] = {}

        for attr in missing_attrs:
            if attr not in df.columns:
                continue
            sep = [c for c in ordering.get(attr, []) if c in df.columns]
            self._sep_cols[attr] = sep
            mask_obs = df[attr].notna()
            observed = df.loc[mask_obs]

            # conditional distributions with binned separating-set columns
            cond: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
            if sep and len(observed) > 0:
                binned_cols = {sc: _bin_column(df[sc]) for sc in sep}
                binned_df = pd.DataFrame(binned_cols, index=df.index)
                self._binned[attr] = binned_df
                obs_binned = binned_df.loc[mask_obs]
                for gk, grp_idx in obs_binned.groupby(sep, dropna=False).groups.items():
                    if not isinstance(gk, tuple):
                        gk = (gk,)
                    gk = tuple(str(v) for v in gk)
                    obs_vals = df.loc[grp_idx, attr].dropna()
                    if len(obs_vals) == 0:
                        continue
                    vc = obs_vals.value_counts(normalize=True)
                    cond[gk] = (vc.index.values, vc.values.astype(np.float64))
            if not cond and len(observed) > 0:
                vc = observed[attr].value_counts(normalize=True)
                cond[("__global__",)] = (vc.index.values, vc.values.astype(np.float64))
            self._conditionals[attr] = cond

            fb_obs = df[attr].dropna()
            if len(fb_obs) > 0:
                vc = fb_obs.value_counts(normalize=True)
                self._fallback[attr] = (vc.index.values, vc.values.astype(np.float64))

            # per-pattern groups for missing rows
            miss_mask = df[attr].isna().values
            if miss_mask.any() and sep and attr in self._binned:
                binned_df = self._binned[attr]
                miss_binned = binned_df.loc[miss_mask, sep]
                groups = []
                for gk, gk_idx in miss_binned.groupby(sep, dropna=False).groups.items():
                    if not isinstance(gk, tuple):
                        gk = (gk,)
                    gk = tuple(str(v) for v in gk)
                    local_positions = np.where(miss_mask)[0]
                    global_idx = binned_df.index.get_indexer(gk_idx)
                    local_map = {v: i for i, v in enumerate(np.where(miss_mask)[0])}
                    local_idx = np.array([local_map[binned_df.index[gi]]
                                          for gi in global_idx
                                          if binned_df.index[gi] in local_map])
                    if len(local_idx) > 0:
                        groups.append((gk, local_idx))
                self._pattern_groups[attr] = groups

    def draw(self, attr: str, mask: np.ndarray, m: int) -> np.ndarray:
        """Return (n_miss, m) array using per-pattern conditional sampling."""
        n_miss = int(mask.sum())
        result = np.empty((n_miss, m), dtype=object)
        fb = self._fallback.get(attr)
        cond = self._conditionals.get(attr, {})
        groups = self._pattern_groups.get(attr, [])

        if groups:
            filled = np.zeros(n_miss, dtype=bool)
            for gk, local_idx in groups:
                dist = cond.get(gk)
                if dist is None or len(dist[0]) == 0:
                    dist = fb
                if dist is None or len(dist[0]) == 0:
                    result[local_idx, :] = np.nan
                else:
                    block = self.rng.choice(
                        dist[0], size=(len(local_idx), m), p=dist[1])
                    result[local_idx, :] = block
                filled[local_idx] = True
            if not filled.all():
                unfilled = np.where(~filled)[0]
                if fb and len(fb[0]) > 0:
                    result[unfilled, :] = self.rng.choice(
                        fb[0], size=(len(unfilled), m), p=fb[1])
        else:
            if fb and len(fb[0]) > 0:
                result[:, :] = self.rng.choice(fb[0], size=(n_miss, m), p=fb[1])
            else:
                result[:, :] = np.nan
        return result


# ═══════════════════════════════════════════════════════════════════
#  Query parsing and evaluation
# ═══════════════════════════════════════════════════════════════════

def _parse_where(query: str):
    """Parse WHERE clauses into list of (col, op, val)."""
    m = re.search(r"WHERE\s+(.+?)(?:\s+GROUP\s+BY|$)", query, re.IGNORECASE)
    if not m:
        return []
    clauses = re.split(r"\s+AND\s+", m.group(1), flags=re.IGNORECASE)
    result = []
    for clause in clauses:
        m_op = re.match(
            r"(\w+)\s*(!=|<>|>=|<=|>|<|=)\s*['\"]?([^'\"]*?)['\"]?\s*$",
            clause.strip())
        if m_op:
            result.append((m_op.group(1), m_op.group(2), m_op.group(3)))
    return result


def _parse_select(query: str):
    m = re.match(r"SELECT\s+(.+?)\s+FROM\s+", query, re.IGNORECASE)
    if not m:
        return []
    return [s.strip() for s in m.group(1).split(",")]


def _parse_groupby(query: str):
    m = re.search(r"GROUP\s+BY\s+(.+)$", query, re.IGNORECASE)
    if not m:
        return []
    return [g.strip() for g in m.group(1).split(",")]


def run_mcdb_bundled(query: str, df_incomplete: pd.DataFrame,
                     missing_attrs: List[str],
                     ordering: Optional[Dict[str, List[str]]],
                     gt_indicators: np.ndarray,
                     m: int = 738,
                     method: str = "mgraph",
                     seed: int = 42,
                     timeout: int = TIMEOUT_S) -> Dict[str, Any]:
    """
    Run Naive-MCDB or mGraph-MCDB using tuple-bundle execution.
    Returns row-level TV, delta_w, CI width, timing.
    """
    t0 = time.time()
    col_map = {c.lower(): c for c in df_incomplete.columns}
    active_missing = [a for a in missing_attrs
                      if a in df_incomplete.columns and df_incomplete[a].isna().any()]

    # Parse query
    where_clauses = _parse_where(query)
    resolved_where = []
    for col_raw, op, val in where_clauses:
        col_name = col_map.get(col_raw.lower())
        if col_name and col_name in df_incomplete.columns:
            resolved_where.append((col_name, op, val))

    # Classify WHERE clauses
    det_clauses = [(c, o, v) for c, o, v in resolved_where if c not in active_missing]
    rnd_clauses = [(c, o, v) for c, o, v in resolved_where if c in active_missing]

    # Deterministic predicate pushdown: filter BEFORE building bundle
    df_pre = df_incomplete.copy()
    pre_mask = np.ones(len(df_pre), dtype=bool)
    for col, op, val in det_clauses:
        series = df_pre[col]
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

    # Map original row indices for gt comparison
    original_indices = np.arange(len(df_incomplete))
    surviving_indices = original_indices[pre_mask]
    df_filtered = df_pre.loc[pre_mask].reset_index(drop=True)

    # Build sampler
    if method == "naive":
        sampler = NaiveSampler(df_filtered, active_missing, seed)
    else:
        sampler = MGraphSampler(df_filtered, active_missing,
                                ordering or {}, seed)

    # Build TupleBundle on filtered data
    bundle = TupleBundle(df_filtered, active_missing, m)

    # Apply random-attribute selections (lazy instantiation happens here)
    for col, op, val in rnd_clauses:
        if bundle.is_random(col):
            bundle.select_random(col, op, val, sampler)
        elif col in bundle.constant:
            bundle.select_constant(col, op, val)

    elapsed = time.time() - t0

    # Inference: per-row membership probability
    row_probs_filtered = bundle.infer_row_probs()
    ci_half_filtered = bundle.infer_ci_half()

    # Map back to original row indices
    row_probs = np.zeros(len(df_incomplete))
    ci_half = np.zeros(len(df_incomplete))
    row_probs[surviving_indices] = row_probs_filtered
    ci_half[surviving_indices] = ci_half_filtered
    # Rows eliminated by deterministic pushdown have p=0 in all worlds
    # (they were filtered before bundle creation — correct by MCDB semantics)

    n = len(df_incomplete)
    tv = 0.5 * np.sum(np.abs(row_probs - gt_indicators)) / n
    mean_ci = float(np.mean(2 * ci_half))
    mean_prob = float(np.mean(row_probs))
    dw = mean_ci / mean_prob if mean_prob > 1e-12 else float("inf")

    return {
        "tv_rowlevel": float(tv),
        "delta_w": dw,
        "mean_ci": mean_ci,
        "time_s": elapsed,
        "m": m,
        "timed_out": False,
        "n_filtered": len(df_filtered),
    }


# ═══════════════════════════════════════════════════════════════════
#  HAVING query evaluation via tuple bundles (fast, no SQL per world)
# ═══════════════════════════════════════════════════════════════════

def _compute_row_survival_prob(df: pd.DataFrame, attr: str, op: str, val: str,
                               method: str, ordering: Optional[Dict[str, List[str]]],
                               n_bins: int = N_BINS) -> np.ndarray:
    """
    For each row, compute P(attr satisfies op val) without materialization.
    Observed rows: deterministic 0 or 1.
    Missing rows: marginal or conditional probability from the distribution.
    Returns (N,) float array of probabilities.
    """
    N = len(df)
    probs = np.ones(N, dtype=np.float64)
    miss_mask = df[attr].isna().values
    obs_mask = ~miss_mask

    # Observed rows: deterministic check
    if obs_mask.any():
        series = df.loc[obs_mask, attr]
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
        probs[obs_mask] = ok.fillna(False).values.astype(np.float64)

    if not miss_mask.any():
        return probs

    observed = df.loc[obs_mask, attr].dropna()
    if len(observed) == 0:
        probs[miss_mask] = 0.0
        return probs

    if method == "naive" or not ordering or attr not in ordering:
        # Marginal probability
        try:
            val_n = float(val)
            obs_vals = pd.to_numeric(observed, errors="coerce").dropna()
            if op in ("=", "=="):    p = (obs_vals == val_n).mean()
            elif op in ("!=", "<>"): p = (obs_vals != val_n).mean()
            elif op == ">":          p = (obs_vals > val_n).mean()
            elif op == "<":          p = (obs_vals < val_n).mean()
            elif op == ">=":         p = (obs_vals >= val_n).mean()
            elif op == "<=":         p = (obs_vals <= val_n).mean()
            else:                    p = 1.0
        except (ValueError, TypeError):
            obs_str = observed.astype(str)
            if op in ("=", "=="):    p = (obs_str == str(val)).mean()
            elif op in ("!=", "<>"): p = (obs_str != str(val)).mean()
            else:                    p = 1.0
        probs[miss_mask] = p
    else:
        # Conditional probability per stratum
        sep = ordering.get(attr, [])
        if not sep:
            probs[miss_mask] = 0.5
            return probs

        valid_sep = [s for s in sep if s in df.columns]
        if not valid_sep:
            probs[miss_mask] = 0.5
            return probs

        # Bin separating set columns
        binned = pd.DataFrame(index=df.index)
        for s in valid_sep:
            binned[s] = _bin_column(df[s])

        miss_idx = np.where(miss_mask)[0]
        miss_keys = binned.iloc[miss_idx]

        # Per-stratum conditional probabilities
        obs_df = df.loc[obs_mask].copy()
        obs_binned = binned.loc[obs_mask]
        obs_df["_binned_key"] = obs_binned[valid_sep].apply(
            lambda r: tuple(str(v) for v in r), axis=1)

        try:
            val_n = float(val)
            obs_attr = pd.to_numeric(obs_df[attr], errors="coerce")
            if op in ("=", "=="):    obs_df["_sat"] = obs_attr == val_n
            elif op in ("!=", "<>"): obs_df["_sat"] = obs_attr != val_n
            elif op == ">":          obs_df["_sat"] = obs_attr > val_n
            elif op == "<":          obs_df["_sat"] = obs_attr < val_n
            elif op == ">=":         obs_df["_sat"] = obs_attr >= val_n
            elif op == "<=":         obs_df["_sat"] = obs_attr <= val_n
            else:                    obs_df["_sat"] = True
        except (ValueError, TypeError):
            obs_str = obs_df[attr].astype(str)
            if op in ("=", "=="):    obs_df["_sat"] = obs_str == str(val)
            elif op in ("!=", "<>"): obs_df["_sat"] = obs_str != str(val)
            else:                    obs_df["_sat"] = True

        grp = obs_df.groupby("_binned_key")["_sat"]
        stratum_prob = grp.mean().to_dict()
        stratum_n = grp.count().to_dict()
        global_p = obs_df["_sat"].mean() if len(obs_df) > 0 else 0.5

        MIN_NX = 5
        miss_binned_keys = miss_keys[valid_sep].apply(
            lambda r: tuple(str(v) for v in r), axis=1)
        for i, (idx, key) in enumerate(zip(miss_idx, miss_binned_keys)):
            nx = stratum_n.get(key, 0)
            if nx >= MIN_NX:
                probs[idx] = stratum_prob.get(key, global_p)
            else:
                probs[idx] = global_p

    return probs


def run_mcdb_bundled_having(query: str, df_incomplete: pd.DataFrame,
                            missing_attrs: List[str],
                            ordering: Optional[Dict[str, List[str]]],
                            gt_groups_set: set,
                            group_cols: List[str],
                            having_threshold: int,
                            m: int = 738,
                            method: str = "mgraph",
                            seed: int = 42,
                            timeout: int = TIMEOUT_S) -> Dict[str, Any]:
    """
    Evaluate HAVING queries via probability-based Bernoulli simulation.
    Avoids materializing full (N, m) arrays. Instead computes per-row survival
    probability and simulates group counts with vectorized Binomial draws.
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
        # ── FULL BUNDLE PATH: group cols are random, need per-world sampling ──
        if method == "naive":
            sampler = NaiveSampler(df_filtered, active_missing, seed)
        else:
            sampler = MGraphSampler(df_filtered, active_missing,
                                    ordering or {}, seed)
        bundle = TupleBundle(df_filtered, active_missing, m)

        for col, op, val in rnd_clauses:
            if bundle.is_random(col):
                bundle.select_random(col, op, val, sampler)
            elif col in bundle.constant:
                bundle.select_constant(col, op, val)

        # Instantiate group columns if not yet materialized
        for gc in real_grp:
            if gc in active_missing and bundle.is_random(gc):
                bundle.instantiate(gc, sampler)

        from scipy import sparse

        group_probs = {}
        group_ci = {}

        # Per-world: build group keys from (possibly random) columns + count
        group_counts: Dict[tuple, int] = defaultdict(int)
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
        # ── FAST BERNOULLI PATH: group cols are constant ──
        row_probs = np.ones(N, dtype=np.float64)
        mc_method = method
        for col, op, val in rnd_clauses:
            p_col = _compute_row_survival_prob(
                df_filtered, col, op, val, mc_method, ordering)
            row_probs *= p_col

        unique_keys = {}
        row_gid = np.empty(N, dtype=np.int32)
        for i in range(N):
            gk = tuple(_norm_val(df_filtered[gc].iloc[i]) for gc in real_grp)
            if gk not in unique_keys:
                unique_keys[gk] = len(unique_keys)
            row_gid[i] = unique_keys[gk]

        n_groups = len(unique_keys)
        gid_to_key = {v: k for k, v in unique_keys.items()}

        group_probs = {}
        group_ci = {}

        for gid in range(n_groups):
            mask_g = row_gid == gid
            p_g = row_probs[mask_g]

            det_count = int((p_g == 1.0).sum())
            stoch_mask = (p_g > 0.0) & (p_g < 1.0)
            stoch_probs = p_g[stoch_mask]

            if len(stoch_probs) == 0:
                if det_count > having_threshold:
                    gk = gid_to_key[gid]
                    group_probs[gk] = 1.0
                    group_ci[gk] = 0.0
                continue

            stoch_draws = rng.random((len(stoch_probs), m)) < stoch_probs[:, None]
            stoch_counts = stoch_draws.sum(axis=0)
            total_counts = det_count + stoch_counts

            survive_count = int((total_counts > having_threshold).sum())
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

    widths = list(group_ci.values()) if group_ci else [0.0]
    probs = list(group_probs.values()) if group_probs else [0.0]
    mean_ci = float(np.mean(widths))
    mean_prob = float(np.mean(probs)) if probs else 0.0
    dw = mean_ci / mean_prob if mean_prob > 1e-12 else float("inf")

    return {
        "tv_prob": tv,
        "delta_w": dw,
        "mean_ci": mean_ci,
        "time_s": elapsed,
        "m": m,
        "timed_out": False,
        "n_gt": len(gt_groups_set),
        "n_pred": len(group_probs),
    }


# ═══════════════════════════════════════════════════════════════════
#  Unsafe query evaluation (full SQL per world via PostgreSQL)
# ═══════════════════════════════════════════════════════════════════

def run_mcdb_unsafe(conn, query: str, df_incomplete: pd.DataFrame,
                    csv_paths: List[str], gt_tables: List[str],
                    table_names: List[str],
                    missing_attrs: List[str],
                    ordering: Optional[Dict[str, List[str]]],
                    m: int = 738,
                    method: str = "mgraph",
                    seed: int = 42,
                    timeout: int = TIMEOUT_S) -> Dict[str, Any]:
    """
    Run Naive/mGraph-MCDB on unsafe queries using full SQL per world.
    Handles single-table and join queries. For joins, samples each table
    independently and loads both into temp tables.
    """
    t0 = time.time()
    cur = conn.cursor()

    tables_in_query = []
    for cp in csv_paths:
        if cp in query or cp.split("/")[-1] in query:
            tables_in_query.append(cp)
    if not tables_in_query:
        tables_in_query = [csv_paths[0]]

    bundles = {}
    samplers = {}
    for cp in tables_in_query:
        df = pd.read_csv(cp)
        active = [a for a in missing_attrs if a in df.columns and df[a].isna().any()]
        if method == "naive":
            samp = NaiveSampler(df, active, seed)
        else:
            samp = MGraphSampler(df, active, ordering or {}, seed)
        bndl = TupleBundle(df, active, m)
        for attr in active:
            bndl.instantiate(attr, samp)
        bundles[cp] = (bndl, df)
        samplers[cp] = samp

    def _make_world_sql(s_idx):
        """Build SQL for world s by rewriting table references to temp tables."""
        q = query
        for i, cp in enumerate(tables_in_query):
            tmp_name = "mcdb_world_%d" % i
            q = q.replace(cp, tmp_name)
            q = q.replace(cp.split("/")[-1], tmp_name)
        return q

    def _make_gt_sql():
        q = query
        for cp, tnames in zip(csv_paths, gt_tables):
            q = q.replace(cp, tnames)
            q = q.replace(cp.split("/")[-1], tnames)
        return q

    def _load_world(s_idx):
        for i, cp in enumerate(tables_in_query):
            tmp_name = "mcdb_world_%d" % i
            bndl, df_orig = bundles[cp]
            world_df = pd.DataFrame(index=range(bndl.n_rows))
            for col in df_orig.columns:
                if col in bndl.random and bndl.random[col] is not None:
                    world_df[col] = bndl.random[col][:, s_idx]
                else:
                    world_df[col] = bndl.constant[col]

            try:
                cur.execute("DROP TABLE IF EXISTS %s" % tmp_name)
                conn.commit()
            except Exception:
                conn.rollback()

            cols_sql = []
            for c in df_orig.columns:
                if df_orig[c].dtype.kind in ("f", "i"):
                    cols_sql.append('"%s" DOUBLE PRECISION' % c)
                else:
                    cols_sql.append('"%s" TEXT' % c)
            cur.execute("CREATE TEMP TABLE %s (%s)" % (tmp_name, ", ".join(cols_sql)))
            buf = StringIO()
            world_df.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            cur.copy_from(buf, tmp_name, sep=",", null="\\N",
                          columns=list(df_orig.columns))
            conn.commit()

    def _cleanup_world():
        for i in range(len(tables_in_query)):
            try:
                cur.execute("DROP TABLE IF EXISTS mcdb_world_%d" % i)
                conn.commit()
            except Exception:
                conn.rollback()

    # For HAVING queries: track which GROUP BY keys survive per world
    # (strip the aggregate count column to get the group key)
    # Detect: if query has HAVING, the last column is the aggregate count
    has_having = "HAVING" in query.upper()

    group_counts: Dict[tuple, int] = defaultdict(int)  # group_key -> # worlds it appeared
    tuple_counts: Dict[tuple, int] = defaultdict(int)   # full tuple -> # worlds
    worlds_done = 0

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        for s in range(m):
            _load_world(s)
            q_sql = _make_world_sql(s)
            try:
                cur.execute(q_sql)
                rows = cur.fetchall()
                seen_groups = set()
                for r in rows:
                    full_key = _norm_tuple(r)
                    tuple_counts[full_key] += 1
                    # Strip count column for group-level tracking
                    if has_having and len(r) > 1:
                        gk = _norm_tuple(r[:-1])
                    else:
                        gk = full_key
                    seen_groups.add(gk)
                for gk in seen_groups:
                    group_counts[gk] += 1
            except Exception:
                conn.rollback()
            _cleanup_world()
            worlds_done += 1
    except _TimeoutError:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        _cleanup_world()

    elapsed = time.time() - t0
    actual_m = max(worlds_done, 1)

    # Group-level probabilities: P(group survives HAVING)
    group_probs = {gk: cnt / actual_m for gk, cnt in group_counts.items()}
    group_ci = {gk: 2 * _Z_95 * math.sqrt(p * (1 - p) / actual_m)
                for gk, p in group_probs.items()}

    try:
        gt_sql = _make_gt_sql()
        cur.execute(gt_sql)
        gt_rows = cur.fetchall()
        if has_having:
            gt_set = set(_norm_tuple(r[:-1]) for r in gt_rows)  # strip count
        else:
            gt_set = set(_norm_tuple(r) for r in gt_rows)
    except Exception:
        conn.rollback()
        gt_set = set()

    # TV over group keys (not full tuples with count)
    all_groups = set(group_probs.keys()) | gt_set
    z_pred = sum(group_probs.values()) if group_probs else 0.0
    z_gt = float(len(gt_set)) if gt_set else 0.0
    tv = 0.0
    for g in all_groups:
        p_p = (group_probs.get(g, 0.0) / z_pred) if z_pred > 0 else 0.0
        p_g = (1.0 / z_gt) if (g in gt_set and z_gt > 0) else 0.0
        tv += abs(p_p - p_g)
    tv /= 2.0

    widths = list(group_ci.values()) if group_ci else [0.0]
    probs = list(group_probs.values()) if group_probs else [0.0]
    mean_ci = float(np.mean(widths))
    mean_prob = float(np.mean(probs)) if probs else 0.0
    dw = mean_ci / mean_prob if mean_prob > 1e-12 else float("inf")

    return {
        "tv_prob": tv,
        "delta_w": dw,
        "mean_ci": mean_ci,
        "time_s": elapsed,
        "m": actual_m,
        "timed_out": worlds_done < m,
        "n_gt": len(gt_set),
        "n_pred": len(group_probs),
    }


# ═══════════════════════════════════════════════════════════════════
#  Metrics (unchanged)
# ═══════════════════════════════════════════════════════════════════

def tv_prob(pred_probs: Dict[tuple, float], gt_tuples: set) -> float:
    all_tuples = set(pred_probs.keys()) | gt_tuples
    if not all_tuples:
        return 0.0
    z_pred = sum(pred_probs.values())
    z_gt = float(len(gt_tuples))
    if z_pred <= 0 and z_gt <= 0:
        return 0.0
    tv = 0.0
    for t in all_tuples:
        p_pred = (pred_probs.get(t, 0.0) / z_pred) if z_pred > 0 else 0.0
        p_gt = (1.0 / z_gt) if (t in gt_tuples and z_gt > 0) else 0.0
        tv += abs(p_pred - p_gt)
    return tv / 2.0


def delta_w(ci_widths: Dict[tuple, float], tuple_probs: Dict[tuple, float]) -> float:
    if not ci_widths:
        return 0.0
    widths = list(ci_widths.values())
    probs = [tuple_probs.get(t, 0.0) for t in ci_widths]
    mean_w = np.mean(widths)
    mean_p = np.mean(probs)
    if mean_p <= 1e-12:
        return float("inf")
    return float(mean_w / mean_p)
