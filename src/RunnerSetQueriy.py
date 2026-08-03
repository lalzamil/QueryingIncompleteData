# RunnerSetQueriy.py
import os, re, json, time, csv, io
from typing import Dict, List, Tuple, Any, Optional
import psycopg2
import myDataAnalyzer
# ---- your executor/rewriter ----
from SetQueryRewriterExecuter import QueryExecutor  # make sure path/name is correct


output_dir = "psql_results"

# ========================== Basic SQL utils ==========================

def strip_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";")

def is_join(sql: str) -> bool:
    return bool(re.search(r"\bJOIN\b", sql, re.IGNORECASE) and
                re.search(r"\bUSING\s*\(", sql, re.IGNORECASE))

def build_maps_from_lists(csv_list: List[str], table_list: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return (full_path_map, basename_map). Supports lookups by:
      - full path
      - basename
      - without .csv
      - DirName/stem
    """
    full, base = {}, {}
    for p, t in zip(csv_list, table_list):
        full[p] = t
        base_name = os.path.basename(p)
        base[base_name] = t

        # also allow lookups without .csv
        if p.lower().endswith('.csv'):
            full[p[:-4]] = t
        if base_name.lower().endswith('.csv'):
            base[base_name[:-4]] = t

        # Dir/stem support (e.g., Mnar1JoinsData/bank_agg_mnar1_5_join1)
        stem = os.path.splitext(os.path.basename(p))[0]
        dir_name = os.path.basename(os.path.dirname(p)) or os.path.dirname(p)
        base[f"{dir_name}/{stem}"] = t
        base[f"{dir_name}\\{stem}"] = t
        # Dir/basename so "Mnar1JoinsData/bank_mnar1_5_join1.csv" is replaced as a whole
        base[f"{dir_name}/{base_name}"] = t
        base[f"{dir_name}\\{base_name}"] = t
    return full, base

def replace_csv_with_tables(sql: str, full_map: dict, base_map: dict) -> str:
    """Replace paths/basenames (with/without .csv) in SQL with real table names."""
    if not isinstance(full_map, dict):
        raise TypeError(f"full_map must be dict, got {type(full_map).__name__}")
    if not isinstance(base_map, dict):
        raise TypeError(f"base_map must be dict, got {type(base_map).__name__}")

    out = sql
    # Replace longer strings first
    for key, tbl in sorted(full_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = re.sub(re.escape(key), tbl, out, flags=re.IGNORECASE)
        if key.lower().endswith('.csv'):
            out = re.sub(re.escape(key[:-4]), tbl, out, flags=re.IGNORECASE)
    # Process keys that contain .csv first (so "path/file.csv" is replaced before "path/file")
    # then by length descending, so we never leave ".csv" behind.
    def _base_sort(kv):
        k = kv[0]
        return (".csv" not in k.lower(), -len(k))
    for key, tbl in sorted(base_map.items(), key=_base_sort):
        out = re.sub(rf"\b{re.escape(key)}\b", tbl, out, flags=re.IGNORECASE)
        # Only replace stem (key without .csv) for bare basenames, not path/stem (avoids leaving .csv)
        if key.lower().endswith('.csv') and "/" not in key and "\\" not in key:
            out = re.sub(rf"\b{re.escape(key[:-4])}\b", tbl, out, flags=re.IGNORECASE)
    return out

def _fix_group_by_select(sql: str) -> str:
    """
    If a query has GROUP BY but SELECT lists unaggregated bare columns not present
    in GROUP BY, append those missing columns to GROUP BY.
    Only touches the GROUP BY list; never touches WHERE.
    """
    m = re.search(r"\bSELECT\s+(?P<select>.+?)\s+FROM\s+(?P<rest>.+)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return sql
    sel_part  = m.group("select")
    rest_part = m.group("rest")
    rest_abs  = m.start("rest")

    g = re.search(r"\bGROUP\s+BY\s+(?P<gb>.+?)(?=(\bHAVING\b|\bORDER\b|$))", rest_part, re.IGNORECASE | re.DOTALL)
    if not g:
        return sql

    gb_abs_start = rest_abs + g.start("gb")
    gb_abs_end   = rest_abs + g.end("gb")
    gb_text = g.group("gb")

    sel_cols = [c.strip() for c in sel_part.split(",")]
    sel_bare = [c for c in sel_cols if "(" not in c and ")" not in c and "*" not in c and c != ""]
    gb_cols  = [c.strip() for c in gb_text.split(",") if c.strip()]
    missing  = [c for c in sel_bare if c not in gb_cols]
    if not missing:
        return sql

    new_gb = ", ".join(gb_cols + missing)
    return sql[:gb_abs_start] + new_gb + sql[gb_abs_end:]


# ==================== GT building (group-level, quoted) ====================

def _parse_sql_basic(sql: str) -> Optional[dict]:
    """
    Pull out: select list, FROM..JOIN..USING, WHERE, GROUP BY, JOIN keys.
    """
    s = sql.strip().rstrip(';')
    m = re.search(
        r"SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<from>.+?)(?:\s+WHERE\s+(?P<where>.+?))?(?:\s+GROUP\s+BY\s+(?P<groupby>.+?))?(?=\s+ORDER\s+BY|$)",
        s, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None
    sel = [c.strip() for c in m.group('select').split(',')]
    frm = m.group('from').strip()
    wh  = (m.group('where') or '').strip()
    gb  = [c.strip() for c in (m.group('groupby') or '').split(',') if c and c.strip()]

    is_join = bool(re.search(r"\bJOIN\b", frm, re.IGNORECASE))
    join_keys = []
    u = re.search(r"\bUSING\s*\((?P<keys>[^)]*)\)", frm, re.IGNORECASE)
    if u:
        join_keys = [k.strip().strip('"') for k in u.group('keys').split(',') if k.strip()]
    return {'select': sel, 'from': frm, 'where': wh, 'groupby': gb, 'is_join': is_join, 'join_keys': join_keys}

def _get_table_cols(cur, table_name: str) -> set:
    cur.execute("""
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema='public' AND table_name=%s
    """, (table_name.strip('"'),))
    return {r[0] for r in cur.fetchall()}

def _quote_ident(name: str) -> str:
    """
    Quote identifiers safely for SQL.
    - If qualified like t.col, keep table part and quote the column.
    - Otherwise, quote the whole name.
    """
    name = name.strip()
    if '.' in name:
        t, c = name.split('.', 1)
        c = c.replace('"', '')
        return f'{t}."{c}"'
    else:
        name = name.replace('"', '')
        return f'"{name}"'

def _qualify_cols(cols: List[str], tname: str, sname: str, tcols: set, scols: set) -> List[str]:
    """
    For each bare column in cols, qualify to the table that has it; quote identifiers.
    If both have it, prefer left (tname). If already qualified, just quote the column part.
    """
    out = []
    for col in cols:
        col = col.strip()
        if not col:
            continue
        if '.' in col:
            t, c = col.split('.', 1)
            c = c.replace('"', '')
            out.append(f'{t}."{c}"')
        else:
            in_t = col in tcols
            in_s = col in scols
            col_clean = col.replace('"', '')
            if in_t and not in_s:
                out.append(f'{tname}."{col_clean}"')
            elif in_s and not in_t:
                out.append(f'{sname}."{col_clean}"')
            elif in_t and in_s:
                out.append(f'{tname}."{col_clean}"')  # prefer left if duplicated
            else:
                out.append(f'"{col_clean}"')
    return out

def _derive_group_cols_single(ordering_T: dict, missing_T: list) -> List[str]:
    """
    Choose GT GROUP BY columns mirroring your rewriter strata:
    union (preserving order) of conditioning columns for the missing attributes.
    """
    cols = []
    for attr in (missing_T or []):
        cols.extend(ordering_T.get(attr, []))
    seen, out = set(), []
    for c in cols:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out

def _build_gt_group_sql_grouped(cur, parsed_gt: dict, ordering_T: dict, missing_T: list) -> Tuple[str, List[str]]:
    """
    Build GT **group-level** SQL mirroring your rewriter strata:
      - group cols = union of conditioning cols for missing_T
      - for JOINs, also include USING keys
      - qualify columns to the table that actually has them
      - quote identifiers (e.g., "default")
    Returns (gt_sql, group_cols_used).
    """
    sel_cols = [c.strip() for c in parsed_gt['select']]
    frm      = parsed_gt['from']
    where    = parsed_gt['where']
    is_j     = parsed_gt['is_join']
    join_keys= parsed_gt['join_keys'] or []

    # Match rewriter behavior: only missing attrs that are query-relevant
    # (appear in SELECT/WHERE) should determine strata for GT grouping.
    query_text = " ".join(sel_cols + ([where] if where else []))
    relevant_missing = [
        a for a in (missing_T or [])
        if re.search(rf"\b{re.escape(a)}\b", query_text, re.IGNORECASE)
    ]
    group_cols = _derive_group_cols_single(ordering_T, relevant_missing)

    tname = sname = None
    tcols = scols = set()
    if is_j:
        m = re.search(r"^\s*(?P<T>\S+)\s+JOIN\s+(?P<S>\S+)\s+USING\s*\(", frm, re.IGNORECASE)
        if m:
            tname = m.group('T').strip('"')
            sname = m.group('S').strip('"')
            tcols = _get_table_cols(cur, tname)
            scols = _get_table_cols(cur, sname)
        for k in join_keys:
            if k not in group_cols:
                group_cols.append(k)

    # qualify + quote
    if is_j and tname and sname:
        sel_q = _qualify_cols(sel_cols, tname, sname, tcols, scols)
        gb_q  = _qualify_cols(group_cols, tname, sname, tcols, scols)
    else:
        sel_q = [_quote_ident(c) for c in sel_cols]
        gb_q  = [_quote_ident(c) for c in group_cols]

    sel_expr = ', '.join(sel_q) if sel_q else '*'
    if gb_q:
        gb_expr = ', '.join(gb_q)
        sql = f"SELECT {gb_expr}, ARRAY_AGG(ROW({sel_expr})) AS gt_tuples FROM {frm} "
        if where: sql += f"WHERE {where} "
        sql += f"GROUP BY {gb_expr}"
        return sql, gb_q
    else:
        sql = f"SELECT ARRAY_AGG(ROW({sel_expr})) AS gt_tuples FROM {frm} "
        if where: sql += f"WHERE {where}"
        return sql, []


def _build_gt_group_total_sql(parsed_gt: dict, ordering_T: dict, missing_T: list,
                               cur=None) -> Tuple[str, List[str]]:
    """
    Build a COUNT(*) per group query on the full table WITHOUT the WHERE predicate.
    This gives |{t in T^full : t[X]=x}| for each separator group x,
    which is the correct denominator for the oracle p_x.
    """
    frm = parsed_gt['from']
    sel_cols = [c.strip() for c in parsed_gt['select']]
    is_j = parsed_gt['is_join']
    join_keys = parsed_gt['join_keys'] or []

    query_text = " ".join(sel_cols + ([parsed_gt['where']] if parsed_gt['where'] else []))
    relevant_missing = [
        a for a in (missing_T or [])
        if re.search(rf"\b{re.escape(a)}\b", query_text, re.IGNORECASE)
    ]
    group_cols = _derive_group_cols_single(ordering_T, relevant_missing)

    tname = sname = None
    tcols = scols = set()
    if is_j:
        m = re.search(r"^\s*(?P<T>\S+)\s+JOIN\s+(?P<S>\S+)\s+USING\s*\(", frm, re.IGNORECASE)
        if m:
            tname = m.group('T').strip('"')
            sname = m.group('S').strip('"')
            if cur:
                tcols = _get_table_cols(cur, tname)
                scols = _get_table_cols(cur, sname)
        for k in join_keys:
            if k not in group_cols:
                group_cols.append(k)

    if is_j and tname and sname:
        gb_q = _qualify_cols(group_cols, tname, sname, tcols, scols)
    else:
        gb_q = [_quote_ident(c) for c in group_cols]

    if gb_q:
        gb_expr = ', '.join(gb_q)
        sql = f"SELECT {gb_expr}, COUNT(*) AS group_total FROM {frm} GROUP BY {gb_expr}"
        return sql, gb_q
    else:
        sql = f"SELECT COUNT(*) AS group_total FROM {frm}"
        return sql, []


def _extract_gt_group_totals(cur, group_cols_used: List[str]) -> Dict[tuple, int]:
    """Extract {group_key: total_count} from the result of _build_gt_group_total_sql."""
    rows = cur.fetchall()
    out: Dict[tuple, int] = {}
    n_gc = len(group_cols_used)
    for row in rows:
        if n_gc > 0:
            gk = _normalize_tuple_for_comparison(tuple(str(v) for v in row[:n_gc]))
        else:
            gk = tuple()
        total = int(row[n_gc]) if row[n_gc] is not None else 0
        out[gk] = out.get(gk, 0) + total
    return out


# ==================== Prediction/GT extraction & metrics ====================

def _parse_pg_record_or_array(s: str) -> tuple:
    """
    Parse PostgreSQL text form of a record (24) or array of one record {(24)}, {()}, {(a,b)}.
    Returns a single tuple (the payload); for array form we take the first (only) element.
    """
    s = s.strip()
    if not s:
        return ()
    # Array of records: {(24)} or {(24,foo)} or {()} or {}
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        inner = s[1:-1].strip()
        if not inner:
            return ()
        # One record inside: (24) or (24,foo) or ("x",3)
        if inner[0] == "(" and inner[-1] == ")":
            inner = inner[1:-1]
            if not inner.strip():
                return ()
            try:
                row = next(csv.reader(io.StringIO(inner)))
            except StopIteration:
                return ()
            return tuple(row)
    # Plain record: (24) or (24,foo)
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1]
        if not inner.strip():
            return ()
        try:
            row = next(csv.reader(io.StringIO(inner)))
        except StopIteration:
            return ()
        return tuple(row)
    return (s,)


def _split_pg_array_top_level(inner: str) -> List[str]:
    """
    Split PostgreSQL array text payload at top-level commas.
    Handles nested records and quoted strings.
    """
    if not inner.strip():
        return []
    out: List[str] = []
    cur: List[str] = []
    depth = 0
    in_quotes = False
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == '"' and (i == 0 or inner[i - 1] != "\\"):
            in_quotes = not in_quotes
            cur.append(ch)
        elif not in_quotes and ch == "(":
            depth += 1
            cur.append(ch)
        elif not in_quotes and ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif not in_quotes and depth == 0 and ch == ",":
            tok = "".join(cur).strip()
            if tok:
                out.append(tok)
            cur = []
        else:
            cur.append(ch)
        i += 1
    tok = "".join(cur).strip()
    if tok:
        out.append(tok)
    return out


def _iter_payload_tuples(arr: Any) -> List[tuple]:
    """
    Normalize payload carrier (record[], record text, list/tuple) to a list of tuples.
    """
    if arr is None:
        return []
    if isinstance(arr, (list, tuple)):
        return [_as_tuple(x) for x in arr]
    if isinstance(arr, str):
        s = arr.strip()
        # PostgreSQL array text: {...}
        if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
            inner = s[1:-1].strip()
            if not inner:
                return []
            toks = _split_pg_array_top_level(inner)
            out: List[tuple] = []
            for tok in toks:
                t = tok.strip()
                if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
                    t = t[1:-1].replace('\\"', '"')
                out.append(_as_tuple(t))
            return out
    return [_as_tuple(arr)]


def _as_tuple(x):
    """Normalize a predicted/GT array element (ROW text/tuple/other) to a Python tuple."""
    if isinstance(x, tuple): return x
    if isinstance(x, list):  return tuple(x)
    if isinstance(x, str):
        s = x.strip()
        if (len(s) >= 2 and (s[0] == "{" or s[0] == "(")):
            parsed = _parse_pg_record_or_array(x)
            return parsed  # may be () for "{()}"
        if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
            inner = s[1:-1]
            row = next(csv.reader(io.StringIO(inner)))
            return tuple(row)
        return (s,)
    return (x,)


def _normalize_tuple_for_comparison(t: tuple) -> tuple:
    """Canonical form so pred and GT tuples match (e.g. Decimal vs int, 16.0 vs 16, trailing spaces)."""
    def _norm_cell(x):
        if x is None:
            return ""
        s = str(x).strip()
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return s
        except ValueError:
            return s
    return tuple(_norm_cell(x) for x in t)

def _extract_pred_groups(pred_rows: List[tuple], has_intervals: bool = False) -> Dict[tuple, Tuple[set, set]]:
    """
    Convert rewriter rows to: dict[group_key_tuple] -> (certain_set, possible_set)
    Row shape without intervals: [ group_keys..., global_probability, missing_tuples, complete_tuples ]
    Row shape with intervals:    [ ..., global_probability, prob_lower, prob_upper, missing_tuples, complete_tuples ]
    """
    n_tail = 5 if has_intervals else 3
    groups: Dict[tuple, Tuple[set, set]] = {}
    for row in pred_rows:
        if len(row) < n_tail + 1:
            continue
        gk = _normalize_tuple_for_comparison(tuple(row[:-n_tail]))
        missing_arr = row[-2]
        complete_arr = row[-1]
        cert, poss = groups.get(gk, (set(), set()))
        # certain
        if complete_arr:
            for t in _iter_payload_tuples(complete_arr):
                cert.add(_normalize_tuple_for_comparison(_as_tuple(t)))
        # possible
        if missing_arr:
            for t in _iter_payload_tuples(missing_arr):
                poss.add(_normalize_tuple_for_comparison(_as_tuple(t)))
        groups[gk] = (cert, poss)
    return groups

def _extract_gt_groups(cur, group_cols_used: List[str]) -> Dict[tuple, set]:
    """
    Read GT rows (after running the GT group SQL) into:
      dict[group_key_tuple] -> set_of_payload_tuples
    Tolerates scalar/composite-string instead of array; normalizes so comparison with pred matches.
    """
    out: Dict[tuple, set] = {}
    rows = cur.fetchall()
    if group_cols_used:
        for r in rows:
            gk = _normalize_tuple_for_comparison(tuple(r[:-1]))
            arr = r[-1]
            s = set()
            for elem in _iter_payload_tuples(arr):
                s.add(_normalize_tuple_for_comparison(_as_tuple(elem)))
            out[gk] = s
    else:
        s = set()
        if rows:
            arr = rows[0][0]
            for elem in _iter_payload_tuples(arr):
                s.add(_normalize_tuple_for_comparison(_as_tuple(elem)))
        out[tuple()] = s
    return out

def _extract_pred_group_prob(pred_rows: List[tuple], has_intervals: bool = False) -> Dict[tuple, float]:
    """
    Extract per-group estimated P̂(φ|X=x) from rewriter output.
    This is the separator-conditioned probability the rewriter computes.
    """
    n_tail = 5 if has_intervals else 3
    out: Dict[tuple, float] = {}
    for row in pred_rows:
        if len(row) < n_tail + 1:
            continue
        gk = _normalize_tuple_for_comparison(tuple(row[:-n_tail]))
        prob_idx = -n_tail  # -3 without intervals, -5 with
        try:
            p = float(row[prob_idx]) if row[prob_idx] is not None else 0.0
        except Exception:
            p = 0.0
        if gk in out:
            out[gk] = max(out[gk], p)
        else:
            out[gk] = p
    return out


def _extract_pred_group_bounds(pred_rows: List[tuple]) -> Dict[tuple, Tuple[float, float]]:
    """
    Extract per-group interval bounds [lower, upper] from rewriter output.
    Row shape with intervals: [ ..., global_probability, prob_lower, prob_upper, missing_tuples, complete_tuples ]
    """
    out: Dict[tuple, Tuple[float, float]] = {}
    for row in pred_rows:
        if len(row) < 6:
            continue
        gk = _normalize_tuple_for_comparison(tuple(row[:-5]))
        try:
            lo = max(0.0, float(row[-4])) if row[-4] is not None else 0.0
            hi = min(1.0, float(row[-3])) if row[-3] is not None else 1.0
        except (TypeError, ValueError):
            lo, hi = 0.0, 1.0
        if lo > hi:
            lo, hi = hi, lo
        if gk not in out:
            out[gk] = (lo, hi)
        else:
            out[gk] = (min(out[gk][0], lo), max(out[gk][1], hi))
    return out


def _mass_weighted_width(pred_rows: List[tuple]) -> float:
    """
    Mass-weighted dataset-level interval width:
      W = Σ_t P̄(t) · (U_t - L_t)
    where P̄(t) = P̃(t) / Z,  Z = Σ_t P̃(t).

    Row format with intervals:
      [...cols..., global_prob, prob_lower, prob_upper, missing_count, complete_count]
    """
    z = 0.0
    weighted_sum = 0.0
    for row in pred_rows:
        if len(row) < 6:
            continue
        try:
            p = float(row[-5]) if row[-5] is not None else 0.0
            lo = max(0.0, float(row[-4])) if row[-4] is not None else 0.0
            hi = min(1.0, float(row[-3])) if row[-3] is not None else 1.0
        except (TypeError, ValueError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        p = max(0.0, p)
        z += p
        weighted_sum += p * (hi - lo)
    if z <= 1e-12:
        return 0.0
    return weighted_sum / z


def _normalized_interval_width(
    pred_bounds: Dict[tuple, Tuple[float, float]],
    pred_prob: Dict[tuple, float],
) -> float:
    """Return mean interval width divided by the absolute mean point estimate."""
    widths = []
    estimates = []
    for key, (lo, hi) in pred_bounds.items():
        widths.append(max(0.0, hi - lo))
        estimates.append(abs(float(pred_prob.get(key, 0.0))))
    if not widths:
        return 0.0
    mean_estimate = sum(estimates) / len(estimates)
    return (sum(widths) / len(widths)) / mean_estimate if mean_estimate > 1e-12 else 0.0


def _interval_metrics(
    pred_bounds: Dict[tuple, Tuple[float, float]],
    pred_prob: Dict[tuple, float],
    gt_groups: Dict[tuple, set],
    gt_totals: Dict[tuple, int],
    alpha: float,
    pred_rows: Optional[List[tuple]] = None,
) -> Dict[str, float]:
    """
    Compute interval evaluation metrics:
      - empirical coverage: fraction of groups where p_oracle in [lower, upper]
      - mean width: mass-weighted dataset-level width Σ P̄(t)·(U_t - L_t)
      - Winkler score: penalizes width + missing the truth (group-level avg)
    Oracle p_x = #{qualifying in T^full for group x} / #{total in T^full for group x}.
    Only evaluated over groups that have predicted bounds.
    """
    if not pred_bounds:
        return {
            "interval_coverage": 0.0,
            "interval_width": 0.0,
            "normalized_interval_width": 0.0,
            "winkler_score": 0.0,
        }

    pred_key_len = len(next(iter(pred_bounds)))
    gt_groups = _truncate_gt_keys(gt_groups, pred_key_len)
    gt_totals = _truncate_gt_totals(gt_totals, pred_key_len)

    n = 0
    covered = 0
    total_winkler = 0.0
    for gk, (lo, hi) in pred_bounds.items():
        gt_qualifying = len(gt_groups.get(gk, set()))
        group_total = gt_totals.get(gk, 0)
        p_oracle = (gt_qualifying / group_total) if group_total > 0 else 0.0

        w = max(0.0, hi - lo)
        if lo <= p_oracle <= hi:
            covered += 1
        winkler = w
        if alpha > 0:
            if p_oracle < lo:
                winkler += (2.0 / alpha) * (lo - p_oracle)
            elif p_oracle > hi:
                winkler += (2.0 / alpha) * (p_oracle - hi)
        total_winkler += winkler
        n += 1

    mass_width = _mass_weighted_width(pred_rows) if pred_rows is not None else 0.0
    normalized_width = _normalized_interval_width(pred_bounds, pred_prob)

    return {
        "interval_coverage": covered / n if n else 0.0,
        "interval_width": mass_width,
        "normalized_interval_width": normalized_width,
        "winkler_score": total_winkler / n if n else 0.0,
    }


def _truncate_gt_keys(gt_groups: Dict[tuple, set], pred_key_len: int) -> Dict[tuple, set]:
    """Merge GT groups whose keys are longer than pred keys by truncating to pred key length."""
    if not gt_groups or pred_key_len <= 0:
        return gt_groups
    sample_key = next(iter(gt_groups))
    if len(sample_key) == pred_key_len:
        return gt_groups
    merged: Dict[tuple, set] = {}
    for gk, s in gt_groups.items():
        short = gk[:pred_key_len]
        if short in merged:
            merged[short] |= s
        else:
            merged[short] = set(s)
    return merged


def _tv_prob_conditional(
    pred_prob: Dict[tuple, float],
    gt_groups: Dict[tuple, set],
    gt_totals: Dict[tuple, int],
    pred_groups: Optional[Dict[tuple, Tuple[set, set]]] = None,
) -> float:
    """
    Proper total variation distance: TV = (1/2) Σ_t |P̃(t)/Z̃ - P*(t)/Z*|.

    Both P̃ and P* assign group-level conditional probabilities to each tuple:
      - cert tuple (observed, qualifying):  P̃(t) = 1,     P*(t) = 1
      - poss tuple (missing attr):          P̃(t) = p̂_x,   P*(t) = p*_x

    After normalization by Z̃ = Σ P̃(t), Z* = Σ P*(t), both become proper
    distributions summing to 1, so TV ∈ [0, 1] and TV = 0 when p̂_x = p*_x.
    """
    if pred_prob:
        pred_key_len = len(next(iter(pred_prob)))
        gt_groups = _truncate_gt_keys(gt_groups, pred_key_len)
        if gt_totals:
            gt_totals = _truncate_gt_totals(gt_totals, pred_key_len)

    all_keys = set(pred_prob.keys()) | set(gt_groups.keys()) | set(gt_totals.keys())
    if not all_keys:
        return 0.0

    eps = 1e-12

    groups_info = []
    z_pred = 0.0
    z_oracle = 0.0
    for gk in all_keys:
        p_hat = pred_prob.get(gk, 0.0)
        gt_qualifying = len(gt_groups.get(gk, set()))
        group_total = gt_totals.get(gk, 0)
        p_oracle = gt_qualifying / group_total if group_total > 0 else 0.0

        if pred_groups is not None:
            cert, poss = pred_groups.get(gk, (set(), set()))
            n_cert = len(cert)
            n_poss = len(poss)
        else:
            n_cert = 0
            n_poss = group_total

        z_pred += n_cert + n_poss * p_hat
        z_oracle += n_cert + n_poss * p_oracle
        groups_info.append((n_cert, n_poss, p_hat, p_oracle))

    if z_pred <= eps and z_oracle <= eps:
        return 0.0

    tv_sum = 0.0
    inv_zp = (1.0 / z_pred) if z_pred > eps else 0.0
    inv_zo = (1.0 / z_oracle) if z_oracle > eps else 0.0

    for n_cert, n_poss, p_hat, p_oracle in groups_info:
        tv_sum += n_cert * abs(inv_zp - inv_zo)
        tv_sum += n_poss * abs(p_hat * inv_zp - p_oracle * inv_zo)

    return 0.5 * tv_sum


def _truncate_gt_totals(gt_totals: Dict[tuple, int], pred_key_len: int) -> Dict[tuple, int]:
    """Merge GT total counts whose keys are longer than pred keys by truncating."""
    if not gt_totals or pred_key_len <= 0:
        return gt_totals
    sample_key = next(iter(gt_totals))
    if len(sample_key) == pred_key_len:
        return gt_totals
    merged: Dict[tuple, int] = {}
    for gk, cnt in gt_totals.items():
        short = gk[:pred_key_len]
        merged[short] = merged.get(short, 0) + cnt
    return merged


def _extract_pred_group_mass(pred_rows: List[tuple], pred_groups: Dict[tuple, tuple], has_intervals: bool = False) -> Dict[tuple, float]:
    """Predicted mass per group = global_probability * |tuples_in_group|."""
    prob_map = _extract_pred_group_prob(pred_rows, has_intervals=has_intervals)
    out: Dict[tuple, float] = {}
    for gk, (cert, poss) in pred_groups.items():
        n = len(cert) + len(poss)
        p = prob_map.get(gk, 1.0)
        out[gk] = p * n
    return out


def _gt_group_mass_from_sets(gt_groups: Dict[tuple, set]) -> Dict[tuple, float]:
    return {gk: float(len(v)) for gk, v in gt_groups.items()}


def _tv_prob_maps(pred_mass: Dict[tuple, float], gt_mass: Dict[tuple, float], eps: float = 1e-12) -> float:
    """TV distance between two normalized mass distributions."""
    keys = set(pred_mass.keys()) | set(gt_mass.keys())
    if not keys:
        return 0.0
    sp = sum(max(0.0, float(pred_mass.get(k, 0.0))) for k in keys)
    sg = sum(max(0.0, float(gt_mass.get(k, 0.0))) for k in keys)
    if sp <= eps and sg <= eps:
        return 0.0
    tv = 0.0
    for k in keys:
        pp = (max(0.0, float(pred_mass.get(k, 0.0))) / sp) if sp > eps else 0.0
        pg = (max(0.0, float(gt_mass.get(k, 0.0))) / sg) if sg > eps else 0.0
        tv += abs(pp - pg)
    return 0.5 * tv


def _micro_metrics_over_groups(pred_groups: Dict[tuple, Tuple[set, set]],
                               gt_groups: Dict[tuple, set]) -> Tuple[float, float, float, int, int]:
    """
    Micro-averaged precision/recall. CSA (confidence set accuracy) = recall = fraction of GT
    contained in our reported set. Per the paper, confidence sets have coverage probability
    1-α (estimation), not 1; so we do not force CSA=1. Covered = certain exact match, or
    GT in a group where we report possible (stratum in our set). fn includes all GT not covered.
    """
    tp_prec = fp_prec = 0
    tp_covered = 0
    fn = 0
    size_gt = 0
    size_pred_all = 0

    for gk, (cert, poss) in pred_groups.items():
        pred = cert | poss
        gt = gt_groups.get(gk, set())
        size_pred_all += len(pred)
        size_gt += len(gt)
        tp_prec += len(cert & gt)
        fp_prec += len(cert - gt)
        if poss:
            tp_prec += 1 if gt else 0
            fp_prec += 1 if not gt else 0
        tp_covered += len(cert & gt)
        if poss:
            tp_covered += len(gt - cert)
        fn += len(gt) - (len(gt) if poss else len(cert & gt))

    for gk in gt_groups.keys():
        if gk not in pred_groups:
            gt = gt_groups[gk]
            size_gt += len(gt)
            fn += len(gt)

    precision = tp_prec / (tp_prec + fp_prec) if (tp_prec + fp_prec) else 0.0
    recall = tp_covered / (tp_covered + fn) if (tp_covered + fn) else 0.0
    accuracy = precision
    return precision, recall, accuracy, size_gt, size_pred_all


def evaluate_query(executor,
                   pred_sql: str,
                   gt_sql: str,
                   ordering_T: Dict[str, List[str]],
                   missing_T: List[str]) -> Dict[str, Any]:
    """
    - Runs predicted (via rewriter) and times it (pred only)
    - Builds GT **group-level** SQL mirroring your strata (single or join)
    - Extracts per-group sets and returns micro-averaged metrics + pred time
    """
    # predicted
    if is_join(pred_sql) and getattr(executor, "_ordering_S", None) and getattr(executor, "_missing_S", None):
        pred_rows = executor.run(pred_sql,
                                 ordering_T=executor._ordering_T,
                                 missing_T=executor._missing_T,
                                 ordering_S=executor._ordering_S,
                                 missing_S=executor._missing_S)
    else:
        pred_rows = executor.run(pred_sql,
                                 ordering_T=executor._ordering_T,
                                 missing_T=executor._missing_T)
    t_pred = getattr(executor, '_sql_elapsed', 0.0)

    has_iv = bool(getattr(executor, 'interval_mode', None))
    pred_groups = _extract_pred_groups(pred_rows, has_intervals=has_iv)

    # GT (group-level)
    parsed = _parse_sql_basic(gt_sql)
    if not parsed:
        raise ValueError("Unable to parse GT SQL for grouping.")
    gt_group_sql, group_cols_used = _build_gt_group_sql_grouped(executor.cur, parsed, ordering_T, missing_T)
    executor.cur.execute(gt_group_sql)
    gt_groups = _extract_gt_groups(executor.cur, group_cols_used)

    gt_total_sql, gt_total_cols = _build_gt_group_total_sql(parsed, ordering_T, missing_T, cur=executor.cur)
    executor.cur.execute(gt_total_sql)
    gt_totals = _extract_gt_group_totals(executor.cur, gt_total_cols)

    pred_prob = _extract_pred_group_prob(pred_rows, has_intervals=has_iv)
    pred_mass = _extract_pred_group_mass(pred_rows, pred_groups, has_intervals=has_iv)
    gt_mass = _gt_group_mass_from_sets(gt_groups)
    tv_prob = _tv_prob_maps(pred_mass, gt_mass)
    tv_cond = _tv_prob_conditional(pred_prob, gt_groups, gt_totals, pred_groups)

    precision, recall, accuracy, size_gt, size_pred_all = _micro_metrics_over_groups(pred_groups, gt_groups)

    return {
        "precision_pess": precision,
        "recall_pess":    recall,
        "precision_opt":  precision,
        "recall_opt":     recall,
        "accuracy":       accuracy,
        "size_gt":        size_gt,
        "size_pred_all":  size_pred_all,
        "time_pred_s":    t_pred,
        "tv_prob":        tv_prob,
        "tv_cond":        tv_cond,
    }


def evaluate_query_with_groups(executor,
                               pred_sql: str,
                               gt_sql: str,
                               ordering_T: Dict[str, List[str]],
                               missing_T: List[str]) -> Tuple[Dict[str, Any], Dict[tuple, Tuple[set, set]], Dict[tuple, set]]:
    """
    Same as evaluate_query but also returns (metrics_dict, pred_groups, gt_groups)
    for use in TV or other comparisons.
    """
    if is_join(pred_sql) and getattr(executor, "_ordering_S", None) and getattr(executor, "_missing_S", None):
        pred_rows = executor.run(pred_sql,
                                 ordering_T=executor._ordering_T,
                                 missing_T=executor._missing_T,
                                 ordering_S=executor._ordering_S,
                                 missing_S=executor._missing_S)
    else:
        pred_rows = executor.run(pred_sql,
                                 ordering_T=executor._ordering_T,
                                 missing_T=executor._missing_T)
    t_pred = getattr(executor, '_sql_elapsed', 0.0)

    has_iv = bool(getattr(executor, 'interval_mode', None))
    pred_groups = _extract_pred_groups(pred_rows, has_intervals=has_iv)

    parsed = _parse_sql_basic(gt_sql)
    if not parsed:
        raise ValueError("Unable to parse GT SQL for grouping.")
    gt_group_sql, group_cols_used = _build_gt_group_sql_grouped(executor.cur, parsed, ordering_T, missing_T)
    executor.cur.execute(gt_group_sql)
    gt_groups = _extract_gt_groups(executor.cur, group_cols_used)

    gt_total_sql, gt_total_cols = _build_gt_group_total_sql(parsed, ordering_T, missing_T, cur=executor.cur)
    executor.cur.execute(gt_total_sql)
    gt_totals = _extract_gt_group_totals(executor.cur, gt_total_cols)

    pred_prob = _extract_pred_group_prob(pred_rows, has_intervals=has_iv)
    pred_mass = _extract_pred_group_mass(pred_rows, pred_groups, has_intervals=has_iv)
    gt_mass = _gt_group_mass_from_sets(gt_groups)
    tv_prob = _tv_prob_maps(pred_mass, gt_mass)
    tv_cond = _tv_prob_conditional(pred_prob, gt_groups, gt_totals, pred_groups)

    precision, recall, accuracy, size_gt, size_pred_all = _micro_metrics_over_groups(pred_groups, gt_groups)
    metrics = {
        "precision_pess": precision, "recall_pess": recall,
        "precision_opt": precision, "recall_opt": recall,
        "accuracy": accuracy, "size_gt": size_gt, "size_pred_all": size_pred_all,
        "time_pred_s": t_pred, "tv_prob": tv_prob, "tv_cond": tv_cond,
    }

    if has_iv:
        iv_alpha = getattr(executor, 'interval_alpha', 0.05)
        pred_bounds = _extract_pred_group_bounds(pred_rows)
        iv_metrics = _interval_metrics(pred_bounds, pred_prob, gt_groups, gt_totals, iv_alpha, pred_rows=pred_rows)
        metrics.update(iv_metrics)

    return metrics, pred_groups, gt_groups


# =============================== Runner ===============================

def to_executor_csv_queries(block_key: str, csvs: List[str], tables: List[str],
                            c_csvs: Optional[List[str]], c_tabs: Optional[List[str]]) -> Dict[str, Any]:
    d = {block_key: {'csv': csvs, 'table': tables}}
    if c_csvs and c_tabs:
        d[block_key]['complete_csv'] = c_csvs
        d[block_key]['complete_table'] = c_tabs
    return d


def run_from_json(json_path: str,
                  group_order: List[str],
                  conn_params: Dict[str, Any]):
    conn = psycopg2.connect(**conn_params)
    try:
        with open(json_path) as f:
            cfg = json.load(f)

        # If group_order is empty, use all top-level keys from the JSON
        if not group_order:
            group_order = list(cfg.keys())

        for group_key in group_order:
            group = cfg.get(group_key, {})
            if not group:
                continue
            print(f"\n================ {group_key.upper()} ================\n")

            for block_key, meta in group.items():
                print(f"\n--- {block_key} ---")

                # 1) normalize lists
                csvs   = meta['csv']   if isinstance(meta['csv'], list)   else [meta['csv']]
                tables = meta['table'] if isinstance(meta['table'], list) else [meta['table']]
                if len(csvs) != len(tables):
                    raise ValueError(f"{block_key}: csv/table length mismatch.")

                complete_csvs = meta.get('complete_csv')
                complete_tabs = meta.get('complete_table')
                if complete_csvs and isinstance(complete_csvs, list):
                    if len(complete_csvs) != len(complete_tabs or []):
                        raise ValueError(f"{block_key}: complete_csv/complete_table length mismatch.")

                # 2) load tables for this subgroup
                csv_queries = to_executor_csv_queries(block_key, csvs, tables, complete_csvs, complete_tabs)
                executor = QueryExecutor(conn, csv_queries)

                # 3) stash orderings (single & join)
                ordering_single = meta.get('ordering_single', {})
                missing_single  = meta.get('missing_attrs_single', [])
                ordering_T      = meta.get('ordering_T', {}) or ordering_single
                missing_T       = meta.get('missing_attrs_T', []) or missing_single
                ordering_S      = meta.get('ordering_S')
                missing_S       = meta.get('missing_attrs_S')
                executor._ordering_T = ordering_T
                executor._missing_T  = missing_T
                executor._ordering_S = ordering_S
                executor._missing_S  = missing_S

                # 4) build maps (keep injected vs complete separate)
                inj_full_map, inj_base_map = build_maps_from_lists(csvs, tables)
                gt_full_map,  gt_base_map  = ({}, {})
                if complete_csvs and complete_tabs:
                    gt_full_map, gt_base_map = build_maps_from_lists(complete_csvs, complete_tabs)

                # index-aligned mapping: injected → complete table alias (robust for GT)
                inj_to_full = {}
                if complete_tabs:
                    for i in range(min(len(tables), len(complete_tabs))):
                        inj_tbl  = tables[i]            # e.g., mnar5b_bank0
                        full_tbl = complete_tabs[i]     # e.g., full_bank0
                        inj_to_full[inj_tbl] = full_tbl

                        inj_csv   = csvs[i]
                        inj_base  = os.path.basename(inj_csv)
                        inj_stem  = os.path.splitext(inj_base)[0]
                        inj_dir   = os.path.basename(os.path.dirname(inj_csv)) or os.path.dirname(inj_csv)
                        inj_to_full[inj_csv]                 = full_tbl
                        inj_to_full[inj_base]                = full_tbl
                        inj_to_full[inj_stem]                = full_tbl
                        inj_to_full[f"{inj_dir}/{inj_base}"] = full_tbl
                        inj_to_full[f"{inj_dir}/{inj_stem}"] = full_tbl

                subgroup_metrics: List[dict] = []

                # 5) per-query loop
                for q in meta.get('queries', []):
                    # predicted SQL over injected tables
                    qn = strip_ws(q)
                    qn = replace_csv_with_tables(qn, inj_full_map, inj_base_map)
                    qn = re.sub(r"\buisng\b", "USING", qn, flags=re.IGNORECASE)
                    qn = re.sub(r"GROUP\s*BY", "GROUP BY", qn, flags=re.IGNORECASE)
                    qn = _fix_group_by_select(qn)
                    for t in tables:
                        qn = re.sub(rf"\b{re.escape(t)}\.csv\b", t, qn, flags=re.IGNORECASE)

                    # ground-truth SQL over complete tables (group-level)
                    gt = strip_ws(q)
                    gt = replace_csv_with_tables(gt, gt_full_map, gt_base_map)
                    gt = re.sub(r"\buisng\b", "USING", gt, flags=re.IGNORECASE)
                    gt = re.sub(r"GROUP\s*BY", "GROUP BY", gt, flags=re.IGNORECASE)
                    gt = _fix_group_by_select(gt)
                    for k in sorted(inj_to_full, key=len, reverse=True):
                        gt = re.sub(re.escape(k), inj_to_full[k], gt, flags=re.IGNORECASE)
                    if complete_tabs:
                        for t in complete_tabs:
                            gt = re.sub(rf"\b{re.escape(t)}\.csv\b", t, gt, flags=re.IGNORECASE)

                    print("-------------------------------")
                    print(f"\nSQL (pred): {qn}")
                    print(f"SQL (gt)  : {gt}")

                    try:
                        m = evaluate_query(executor, qn, gt, ordering_T, missing_T)
                        print(f" GT size={m['size_gt']}  Pred all={m['size_pred_all']}")
                        print(f" Prec={m['precision_pess']:.3f}  Rec={m['recall_pess']:.3f}  Accuracy={m['accuracy']:.3f}")
                        print(f" Time: pred={m['time_pred_s']:.4f}s")
                        subgroup_metrics.append(m)
                    except Exception as e:
                        print(f"!! ERROR: {e}")
                        try:
                            conn.rollback()  # leave aborted state to proceed with next queries
                        except Exception:
                            pass

                # 6) subgroup (block) summary
                if subgroup_metrics:
                    analyzer = myDataAnalyzer.myDataAnalyzer(datasetName=block_key, output_dir="psql_results",out_file="a_real_mnar1_set_group_level_psql_injected.txt")
                    def _avg(key):
                        vals = [x[key] for x in subgroup_metrics]
                        return sum(vals)/len(vals) if vals else 0.0
                    print("\n--- Block summary ---")
                    print(f" Avg Prec: {_avg('precision_pess'):.3f}  Avg Rec: {_avg('recall_pess'):.3f}  Avg Accuracy: {_avg('accuracy'):.3f}")
                    print(f" Avg Pred Time: {_avg('time_pred_s'):.4f}s")
                    analyzer.add_set_queries_time(_avg('time_pred_s'))
                    analyzer.addNewLine()

                executor.cur.close()

    finally:
        conn.close()


# ============================ Entrypoint ============================

if __name__ == "__main__":
    JSON_PATH = "configs/mnar_set_queries.json"   # <- set your path
    # JSON_PATH = "configs/real_mnar_set_queries.json"
    # Leave empty to run all groups in the JSON; otherwise list group keys to run
    GROUP_ORDER = [
        # "bank_manr1_set",
        # "nyc_manr1_set",
        # "bit_manr1_set",
        # "student_mnar_set",
        # "plane_mnar_set",
        # "med_mnar_set",
    ]
    CONN = dict(host="localhost", port=5433, dbname="mydb", user="alzamill", password=os.environ.get("PGPASSWORD", ""))
    run_from_json(JSON_PATH, GROUP_ORDER, CONN)
    # Leave empty to run all groups in the JSON; otherwise list group keys to run
    GROUP_ORDER = [
        # "bank_manr1_set",
        # "nyc_manr1_set",
        # "bit_manr1_set",
        # "student_mnar_set",
        # "plane_mnar_set",
        # "med_mnar_set",
    ]
    CONN = dict(host="localhost", port=5433, dbname="mydb", user="alzamill", password=os.environ.get("PGPASSWORD", ""))
    run_from_json(JSON_PATH, GROUP_ORDER, CONN)
