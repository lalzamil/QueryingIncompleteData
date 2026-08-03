import os, re, json, time, math
from typing import Dict, List, Tuple, Any, Optional, Set
import psycopg2
import myDataAnalyzer
from bias_utils import BiasUtil
# If you import QueryExecutor elsewhere it's fine to keep this; not required here:
# from TopFractionPatternsQueryExecuter import QueryExecutor

# ========================= Config =========================
bias_analyzer = BiasUtil(dedup_joint=True, smooth_alpha=1e-8)
myCurrent_outfile = "a_mass_TV_certain_with_bias.txt"

# ========================= Regex/Parsers =========================
PROBLEM_IDENTS = ("count", "default", "user", "order", "group", "limit", "offset", "sum", "avg", "min", "max")

SELECT_RE = re.compile(
    r"^\s*select\s+(?P<select>.+?)\s+from\s+(?P<from>.+?)(?:\s+where\s+(?P<where>.+?))?(?:\s+group\s+by\s+(?P<groupby>.+?))?\s*$",
    re.IGNORECASE | re.DOTALL,
)
USING_RE = re.compile(r"\busing\s*\(\s*([a-zA-Z0-9_]+)\s*\)", re.IGNORECASE)
AND_SPLIT = re.compile(r"\s+and\s+", re.IGNORECASE)
PRED_RE   = re.compile(r"^\s*(?P<col>[a-zA-Z0-9_.]+)\s*(?P<op>=|!=|>=|<=|>|<)\s*(?P<val>'.*?'|\".*?\"|-?\s*\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
IDENT_RE  = re.compile(r"[a-zA-Z_][a-zA-Z0-9_\.]*")

SQL_WORDS = {
    "select","from","where","group","by","order","having","join","left","right","full","outer","inner",
    "on","using","and","or","not","in","between","like","ilike","is","null","distinct","as","case","when","then","else","end",
    "true","false","limit","offset","union","all"
}
JOIN_ON_RE    = re.compile(r"\bon\s+(?P<expr>.*?)(?=\bjoin\b|\bwhere\b|\bgroup\b|\border\b|$)", re.IGNORECASE | re.DOTALL)
EQUAL_PAIR_RE = re.compile(r"\b([A-Za-z_][\w\.]*)\s*=\s*([A-Za-z_][\w\.]*)\b")

# ========================= Set semantics =========================
DISTINCT_RE = re.compile(r"^\s*select\s+distinct\b", re.IGNORECASE)
AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)

def enforce_set_semantics(sql: str) -> str:
    """
    Deduplicate top-level results unless the query already has DISTINCT,
    uses GROUP BY, or is an aggregate query. Safe for SPJ workloads.
    """
    if DISTINCT_RE.search(sql):
        return sql

    m = SELECT_RE.match(sql.strip())
    if not m:
        return sql  # not our simple SELECT ... FROM ... shape

    select_part = m.group("select") or ""
    group_part  = (m.group("groupby") or "").strip()
    if group_part:
        return sql                 # GROUP BY yields 1 row per group
    if AGG_RE.search(select_part):
        return sql                 # aggregate query; do not add DISTINCT

    # Promote to set semantics
    return re.sub(r"^\s*select\s+", "SELECT DISTINCT ", sql, count=1, flags=re.IGNORECASE)

# ========================= Small helpers =========================
def strip_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";")

def rows_to_set(rows: List[tuple]) -> Set[tuple]:
    out: Set[tuple] = set()
    for r in rows:
        if isinstance(r, tuple):
            out.add(r)
        else:
            out.add(tuple(r if isinstance(r, list) else (r,)))
    return out

def _rows_to_counts(rows: List[tuple]) -> Tuple[Dict[tuple, int], int]:
    """Bag semantics: counts and total (for coverage)."""
    d: Dict[tuple, int] = {}
    total = 0
    for r in rows:
        k = r if isinstance(r, tuple) else tuple(r if isinstance(r, list) else (r,))
        d[k] = d.get(k, 0) + 1
        total += 1
    return d, total

def coverage_from_rows(pred_rows: List[tuple], gt_rows: List[tuple]) -> float:
    """
    Coverage = fraction of GT bag mass captured by the predicted *set* S.
    Coverage = (sum_{t in S} GT_count(t)) / (sum_{t} GT_count(t)).
    """
    S = rows_to_set(pred_rows)
    gt_counts, total_gt = _rows_to_counts(gt_rows)
    if total_gt == 0:
        return 1.0
    covered = sum(gt_counts.get(t, 0) for t in S)
    return covered / float(total_gt)

# ========================= Bias / distances on sets =========================
def _tv_js_on_sets(pred_rows, gt_rows, eps=1e-12):
    Pset = rows_to_set(pred_rows)
    Qset = rows_to_set(gt_rows)
    if not Pset and not Qset:
        return 0.0, 0.0, 0.0
    # uniform over sets
    p = 1.0 / len(Pset) if Pset else 0.0
    q = 1.0 / len(Qset) if Qset else 0.0
    keys = Pset | Qset
    tv = 0.5 * sum(abs((p if k in Pset else 0.0) - (q if k in Qset else 0.0)) for k in keys)
    # JSD (uniform over sets)
    js = 0.0
    for k in keys:
        pk = p if k in Pset else eps
        qk = q if k in Qset else eps
        m  = 0.5*(pk+qk)
        js += 0.5*(pk*math.log(pk/m) + qk*math.log(qk/m))
    return tv, js, math.sqrt(js)

# ========================= Identifier quoting / GBY fix =========================
def _quote_problem_idents(sql: str, idents=PROBLEM_IDENTS) -> str:
    """
    Quote problematic identifiers safely when used as column names (not function calls).
    - Qualified:   t.count  -> t."count"
    - Bare in SELECT/GBY: count -> "count"
    Never touches WHERE text; only edits SELECT list and GROUP BY list.
    """
    out = sql
    # Pass 1: qualified occurrences (t.name), not followed by '('
    for name in idents:
        out = re.sub(
            rf'\b([A-Za-z_][A-Za-z0-9_]*)\.({name})\b(?!\s*\()',
            lambda m: f'{m.group(1)}."{m.group(2)}"',
            out, flags=re.IGNORECASE
        )
    # Pass 2: bare tokens in SELECT (and GROUP BY if present)
    m = re.search(r'\bSELECT\s+(?P<select>.+?)\s+FROM\s+(?P<rest>.+)', out, re.IGNORECASE | re.DOTALL)
    if not m:
        return out
    sel_part  = m.group('select')
    rest_part = m.group('rest')
    rest_abs  = m.start('rest')

    def _quote_bare(text: str) -> str:
        for name in idents:
            text = re.sub(rf'(?<!")\b{name}\b(?!")\b(?!\s*\()',
                          f'"{name}"', text, flags=re.IGNORECASE)
        return text

    sel_quoted = _quote_bare(sel_part)

    g = re.search(r'\bGROUP\s+BY\s+(?P<gb>.+?)(?=(\bHAVING\b|\bORDER\b|$))',
                  rest_part, re.IGNORECASE | re.DOTALL)
    if g:
        gb_abs_start = rest_abs + g.start('gb')
        gb_abs_end   = rest_abs + g.end('gb')
        gb_part      = g.group('gb')
        gb_quoted    = _quote_bare(gb_part)
        out = out[:m.start('select')] + sel_quoted + out[m.end('select'):gb_abs_start] + gb_quoted + out[gb_abs_end:]
    else:
        out = out[:m.start('select')] + sel_quoted + out[m.end('select'):]
    return out

def _fix_group_by_select(sql: str) -> str:
    """
    If a query has GROUP BY but SELECT lists unaggregated columns not present
    in GROUP BY, append them to GROUP BY.
    """
    m = re.search(r"\bSELECT\s+(?P<select>.+?)\s+FROM\s+(?P<rest>.+)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return sql
    sel_part  = m.group("select")
    rest_part = m.group("rest")
    rest_abs  = m.start("rest")

    g = re.search(r"\bGROUP\s+BY\s+(?P<gb>.+?)(?=(\bHAVING\b|\bORDER\b|$))",
                  rest_part, re.IGNORECASE | re.DOTALL)
    if not g:
        return sql

    gb_abs_start = rest_abs + g.start("gb")
    gb_abs_end   = rest_abs + g.end("gb")
    gb_text = g.group("gb")

    sel_cols = [c.strip() for c in sel_part.split(",")]
    sel_bare = [c for c in sel_cols if "(" not in c and ")" not in c and "*" not in c and c]
    gb_cols  = [c.strip() for c in gb_text.split(",") if c.strip()]
    missing  = [c for c in sel_bare if c not in gb_cols]
    if not missing:
        return sql
    new_gb = ", ".join(gb_cols + missing)
    return sql[:gb_abs_start] + new_gb + sql[gb_abs_end:]

# ========================= Certain rewrite (strict) =========================
def parse_sql_shape(sql: str) -> Dict[str, Any]:
    m = SELECT_RE.match(sql.strip())
    if not m:
        return {"ok": False}
    return {
        "ok": True,
        "select": m.group("select") or "",
        "from": m.group("from") or "",
        "where": (m.group("where") or "").strip(),
        "groupby": (m.group("groupby") or "").strip(),
    }

def extract_predicate_columns(where_part: str) -> List[str]:
    """Simple a <op> c extraction (kept for compatibility with older queries)."""
    if not where_part:
        return []
    cols: List[str] = []
    parts = AND_SPLIT.split(where_part)
    for p in parts:
        pm = PRED_RE.match(p.strip())
        if pm:
            cols.append(pm.group("col").strip())
    return cols

def extract_using_keys(from_part: str) -> List[str]:
    keys = []
    for m in USING_RE.finditer(from_part or ""):
        keys.append(m.group(1).strip())
    return keys

def _strip_strings(s: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", s or "")

def extract_expr_identifiers(expr: str) -> Set[str]:
    """
    Collect column-like tokens from an arbitrary SQL expression
    (skips SQL keywords and function names).
    """
    expr = _strip_strings(expr)
    cols: Set[str] = set()
    for m in IDENT_RE.finditer(expr):
        tok = m.group(0)
        if tok.lower() in SQL_WORDS:
            continue
        j = m.end()
        if j < len(expr) and expr[j] == "(":
            continue
        cols.add(tok)
    return cols

def extract_on_equality_cols(from_part: str) -> Set[str]:
    """Both sides of equality predicates inside JOIN ... ON (...) clauses."""
    cols: Set[str] = set()
    for m in JOIN_ON_RE.finditer(from_part or ""):
        on_expr = m.group("expr") or ""
        for a, b in EQUAL_PAIR_RE.findall(on_expr):
            cols.add(a.strip()); cols.add(b.strip())
    return cols

def extract_select_columns(select_part: str) -> List[str]:
    out: List[str] = []
    if not select_part:
        return out
    for piece in select_part.split(","):
        piece = piece.strip()
        ids = IDENT_RE.findall(piece)
        if ids:
            out.append(ids[-1])
    return out

def add_is_not_null_guards(sql: str, strict_level: int = 2) -> str:
    """
    Strict certain-rewrite:
      - Always guard WHERE operands, USING keys, and JOIN-ON equality columns.
      - If strict_level >= 1: also guard SELECT/GROUP BY identifiers.
      - If strict_level >= 2: be maximally conservative (include expr identifiers from WHERE).
    """
    shape = parse_sql_shape(sql)
    if not shape["ok"]:
        return sql

    where_part  = shape["where"]
    from_part   = shape["from"]
    select_part = shape["select"]
    group_part  = shape["groupby"]

    # A) membership-critical identifiers
    pred_cols_simple = set(extract_predicate_columns(where_part))
    pred_cols_expr   = extract_expr_identifiers(where_part)      # catches IN/BETWEEN/LIKE/func args/etc.
    using_keys       = set(extract_using_keys(from_part))        # USING(...)
    on_cols          = extract_on_equality_cols(from_part)       # ON a=b -> {a,b}

    # B) payload-shaping columns
    select_cols = set(extract_select_columns(select_part))
    group_cols  = set(extract_select_columns(group_part)) if group_part else set()

    required: Set[str] = set()
    required |= using_keys | on_cols | pred_cols_simple
    if strict_level >= 1:
        required |= select_cols | group_cols
    if strict_level >= 2:
        required |= pred_cols_expr  # include all identifiers referenced in WHERE expressions

    if not required:
        return sql

    # keep only identifier-like tokens
    required = {c for c in required if IDENT_RE.fullmatch(c)}
    guards = " AND ".join(f"{c} IS NOT NULL" for c in sorted(required))

    # inject guards
    if where_part and where_part.strip():
        new_where = f"({where_part}) AND ({guards})"
        return re.sub(r"(?is)\bWHERE\b.+?(?=(\bGROUP\b|\bORDER\b|$))",
                      "WHERE " + new_where + " ", sql, count=1)
    else:
        m = re.search(r"(?is)\bFROM\b", sql)
        if not m:
            return sql
        head = sql[:m.end()]
        tail = sql[m.end():]
        cut = re.search(r"(?is)\b(GROUP|ORDER)\b", tail)
        if cut:
            return head + tail[:cut.start()] + f" WHERE {guards} " + tail[cut.start():]
        return head + f" WHERE {guards} " + tail

# ========================= Evaluation (SET + Coverage) =========================
def evaluate_certain(cur, pred_sql_injected: str, gt_sql_complete: str) -> Dict[str, Any]:
    """
    - Rewrite pred_sql_injected -> certain via IS NOT NULL guards (strict).
    - Quote problematic identifiers; fix GROUP BY.
    - Enforce set semantics (SELECT DISTINCT where safe).
    - Run pred (time), run gt (normalized the same way).
    - Compute set precision/recall & set-TV, plus **coverage** and **TV_mass_cov = 1 - coverage**.
    """
    # build certain-pred (set version keeps legacy TV_set behavior)
    pred_certain_bag = add_is_not_null_guards(pred_sql_injected, strict_level=2)
    pred_certain_bag = _quote_problem_idents(pred_certain_bag)
    pred_certain_bag = _fix_group_by_select(pred_certain_bag)
    pred_certain = enforce_set_semantics(pred_certain_bag)

    # run pred set
    t0 = time.perf_counter()
    cur.execute(pred_certain)
    pred_rows = cur.fetchall()
    t_pred = time.perf_counter() - t0
    pred_cols = [d[0] for d in cur.description]

    # build/normalize GT the same way (set semantics)
    gt_sql_bag = _quote_problem_idents(gt_sql_complete)
    gt_sql_bag = _fix_group_by_select(gt_sql_bag)
    gt_sql = enforce_set_semantics(gt_sql_bag)
    cur.execute(gt_sql)
    gt_rows = cur.fetchall()
    gt_cols = [d[0] for d in cur.description]

    # set metrics
    S_pred = rows_to_set(pred_rows)
    S_gt   = rows_to_set(gt_rows)
    inter  = S_pred & S_gt
    prec = (len(inter) / len(S_pred)) if S_pred else (1.0 if not S_gt else 0.0)
    rec  = (len(inter) / len(S_gt))   if S_gt   else (1.0 if not S_pred else 0.0)

    # set-based distances
    tv_set, js, js_sqrt = _tv_js_on_sets(pred_rows, gt_rows)

    # coverage (bag on GT, set on pred)
    coverage = coverage_from_rows(pred_rows, gt_rows)
    tv_mass_cov = 1.0 - coverage

    # --- Bag (row-level) precision/recall: exposes stronger certain-answer recall drop ---
    cur.execute(pred_certain_bag)
    pred_rows_bag = cur.fetchall()
    cur.execute(gt_sql_bag)
    gt_rows_bag = cur.fetchall()
    pred_counts, n_pred_bag = _rows_to_counts(pred_rows_bag)
    gt_counts, n_gt_bag = _rows_to_counts(gt_rows_bag)
    inter_bag = sum(min(pred_counts.get(k, 0), gt_counts.get(k, 0)) for k in (set(pred_counts) | set(gt_counts)))
    prec_bag = (inter_bag / n_pred_bag) if n_pred_bag else (1.0 if n_gt_bag == 0 else 0.0)
    rec_bag = (inter_bag / n_gt_bag) if n_gt_bag else (1.0 if n_pred_bag == 0 else 0.0)

    # distributional bias utility (still set payloads; uses your BiasUtil)
    try:
        bias = bias_analyzer.measure_bias(list(S_pred), list(S_gt), pred_cols, gt_cols)
    except Exception as e:
        print(f"BiasUtil failed: {e}")
        bias = {}

    # Group-level coverage skew: measures how unevenly certain answers
    # represent different groups. Groups with high missingness get fewer
    # certain tuples, creating skew.
    # ratio(g) = |certain_g| / |GT_g| for each group g defined by query columns.
    # bias_skew = CV of ratios; bias_jsd = JSD(ratio_dist, uniform_dist).
    group_bias_cv = 0.0
    group_bias_jsd = 0.0
    try:
        from collections import Counter
        pred_group_counts = Counter(t for t in S_pred)
        gt_group_counts = Counter(t for t in S_gt)
        if gt_group_counts:
            ratios = []
            for g, gt_n in gt_group_counts.items():
                pred_n = pred_group_counts.get(g, 0)
                ratios.append(pred_n / gt_n)
            if ratios:
                import numpy as np
                ratios_arr = np.array(ratios, dtype=float)
                mean_r = ratios_arr.mean()
                if mean_r > 1e-12:
                    group_bias_cv = float(ratios_arr.std() / mean_r)
                n = len(ratios_arr)
                uniform = np.ones(n) / n
                p_dist = ratios_arr / (ratios_arr.sum() + 1e-15)
                m = 0.5 * (p_dist + uniform)
                kl_pm = float(np.sum(p_dist * np.log((p_dist + 1e-15) / (m + 1e-15))))
                kl_um = float(np.sum(uniform * np.log((uniform + 1e-15) / (m + 1e-15))))
                group_bias_jsd = float(np.sqrt(max(0, 0.5 * kl_pm + 0.5 * kl_um)))
    except Exception:
        pass

    return {
        "precision": prec,
        "recall":    rec,
        "precision_bag": prec_bag,
        "recall_bag": rec_bag,
        "size_pred": len(S_pred),
        "size_gt":   len(S_gt),
        "size_pred_bag": n_pred_bag,
        "size_gt_bag": n_gt_bag,
        "time_pred_s": t_pred,
        "pred_sql_certain": pred_certain,
        "pred_sql_certain_bag": pred_certain_bag,
        "tv_set": tv_set,
        "js": js,
        "js_sqrt": js_sqrt,
        "coverage": coverage,
        "tv_mass_cov": tv_mass_cov,
        "f2_score": float(bias.get("f2_score", 0.0)),
        "avg_jsd": group_bias_jsd,
        "group_bias_cv": group_bias_cv,
        "joint_jsd_coverage": bias.get("joint_jsd_coverage", 0.0),
        "avg_wasserstein": bias.get("avg_wasserstein", 0.0),
        "distributional_details": bias.get("distributional_details", {}),
    }

# ========================= CSV→table mapping (unchanged) =========================
def build_maps_from_lists(csv_list: List[str], table_list: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (full_path_map, basename_map); supports without .csv and DirName/stem forms."""
    full, base = {}, {}
    for p, t in zip(csv_list, table_list):
        full[p] = t
        base_name = os.path.basename(p)
        base[base_name] = t
        if p.lower().endswith('.csv'):
            full[p[:-4]] = t
        if base_name.lower().endswith('.csv'):
            base[base_name[:-4]] = t
        stem = os.path.splitext(base_name)[0]
        dir_name = os.path.basename(os.path.dirname(p)) or os.path.dirname(p)
        base[f"{dir_name}/{stem}"] = t
        base[f"{dir_name}\\{stem}"] = t
    return full, base

def replace_csv_with_tables(sql: str, full_map: dict, base_map: dict) -> str:
    """
    Swap any path/basename tokens (with/without .csv) to table aliases.
    Removes any preceding directories so no "/" remains.
    """
    out = sql

    # (a) Full keys first (longest-first)
    for key, tbl in sorted(full_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        pattern_exact = re.escape(key)
        out = re.sub(pattern_exact, tbl, out, flags=re.IGNORECASE)
        if key.lower().endswith('.csv'):
            pattern_noext = re.escape(key[:-4])
            out = re.sub(pattern_noext, tbl, out, flags=re.IGNORECASE)
        else:
            pattern_withext = re.escape(key + '.csv')
            out = re.sub(pattern_withext, tbl, out, flags=re.IGNORECASE)

    # (b) dir/stem and (c) basename from base_map
    def _replace_dir_stem_tokens(out_text: str, k: str, tbl_name: str) -> str:
        k_esc = re.escape(k)
        pat = rf"(?:[A-Za-z0-9_\-\.]+[\/])*{k_esc}(?:\.csv)?"
        return re.sub(pat, tbl_name, out_text, flags=re.IGNORECASE)

    for key, tbl in sorted(base_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        out = _replace_dir_stem_tokens(out, key, tbl)
        if key.lower().endswith('.csv'):
            out = _replace_dir_stem_tokens(out, key[:-4], tbl)
        else:
            out = _replace_dir_stem_tokens(out, key + '.csv', tbl)

    return out

# ========================= Runner =========================
def run_from_json(json_path: str,
                  group_order: List[str],
                  conn_params: Dict[str, Any]):
    conn = psycopg2.connect(**conn_params)
    try:
        with open(json_path) as f:
            cfg = json.load(f)

        for group_key in group_order:
            group = cfg.get(group_key, {})
            if not group:
                continue
            print(f"\n================ {group_key.upper()} ================\n")

            for block_key, meta in group.items():
                print(f"\n--- {block_key} ---")

                # normalize lists (CSV ↔ tables)
                csvs   = meta['csv']   if isinstance(meta['csv'], list)   else [meta['csv']]
                tables = meta['table'] if isinstance(meta['table'], list) else [meta['table']]
                if len(csvs) != len(tables):
                    raise ValueError(f"{block_key}: csv/table length mismatch.")

                complete_csvs = meta.get('complete_csv')
                complete_tabs = meta.get('complete_table')
                if complete_csvs and isinstance(complete_csvs, list):
                    if len(complete_csvs) != len(complete_tabs or []):
                        raise ValueError(f"{block_key}: complete_csv/complete_table length mismatch.")

                inj_full_map, inj_base_map = build_maps_from_lists(csvs, tables)
                gt_full_map,  gt_base_map  = ({}, {})
                if complete_csvs and complete_tabs:
                    gt_full_map, gt_base_map = build_maps_from_lists(complete_csvs, complete_tabs)

                # mapping: injected token -> complete table alias (fallback)
                inj_to_full = {}
                if complete_tabs:
                    for i in range(min(len(tables), len(complete_tabs))):
                        inj_tbl  = tables[i]
                        full_tbll = complete_tabs[i]
                        inj_to_full[inj_tbl] = full_tbll
                        inj_csv   = csvs[i]
                        base_name = os.path.basename(inj_csv)
                        stem      = os.path.splitext(base_name)[0]
                        dir_name  = os.path.basename(os.path.dirname(inj_csv)) or os.path.dirname(inj_csv)
                        inj_to_full[inj_csv]                   = full_tbll
                        inj_to_full[base_name]                 = full_tbll
                        inj_to_full[stem]                      = full_tbll
                        inj_to_full[f"{dir_name}/{base_name}"] = full_tbll
                        inj_to_full[f"{dir_name}/{stem}"]      = full_tbll

                cur = conn.cursor()
                results = []

                for q in meta.get('queries', []):
                    # Build pred and gt SQL with table aliases
                    pred = strip_ws(q)
                    pred = replace_csv_with_tables(pred, inj_full_map, inj_base_map)
                    pred = re.sub(r"\buisng\b", "USING", pred, flags=re.IGNORECASE)

                    gt = strip_ws(q)
                    gt = replace_csv_with_tables(gt, gt_full_map, gt_base_map)
                    gt = re.sub(r"\buisng\b", "USING", gt, flags=re.IGNORECASE)
                    gt = _quote_problem_idents(gt)
                    gt = _fix_group_by_select(gt)
                    for k in sorted(inj_to_full, key=len, reverse=True):
                        gt = re.sub(re.escape(k), inj_to_full[k], gt, flags=re.IGNORECASE)

                    print("-------------------------------")
                    print(f"SQL (pred; certain): {add_is_not_null_guards(pred, strict_level=2)}")
                    print(f"SQL (gt)           : {gt}")

                    try:
                        m = evaluate_certain(cur, pred, gt)
                        print(f" GT size={m['size_gt']}  Pred size={m['size_pred']}")
                        print(f" Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
                        print(f" Time: pred={m['time_pred_s']:.4f}s")
                        print(f" TV_set={m['tv_set']:.3f}  TV_mass_cov={m['tv_mass_cov']:.3f}  coverage={m['coverage']:.3f}")
                        results.append(m)
                    except Exception as e:
                        print(f"!! ERROR: {e}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                # Block-level summary (averages over queries)
                if results:
                    analyzer = myDataAnalyzer.myDataAnalyzer(
                        datasetName=block_key, output_dir="psql_results", out_file=myCurrent_outfile
                    )
                    def _avg(k):
                        vals = [x[k] for x in results]
                        return (sum(vals)/len(vals)) if vals else 0.0

                    print("\n--- Block summary ---")
                    print(f" Avg Prec={_avg('precision'):.3f}  Avg Rec={_avg('recall'):.3f}")
                    print(f" Avg Pred Time={_avg('time_pred_s'):.4f}s")
                    print(f" Avg TV_set={_avg('tv_set'):.4f}  Avg TV_mass_cov={_avg('tv_mass_cov'):.4f}  Avg coverage={_avg('coverage'):.4f}")

                    # Keep your existing analyzer write-out as-is (doesn't include coverage fields)
                    analyzer.add_dominancePALL_queries_time(_avg('time_pred_s'), _avg('precision'), _avg('recall'), _avg('tv_mass_cov'))
                    analyzer.addNewLine()

                cur.close()

    finally:
        conn.close()

# ========================= Entrypoint =========================
if __name__ == "__main__":
    JSON_PATH = "configs/certain_answers_queries.json"
    GROUP_ORDER = [
        "bank_manr1_set",
        "nyc_manr1_set",
        "bit_manr1_set",
    ]
    CONN = dict(host="localhost", port=5433, dbname="mydb", user="alzamill", password=os.environ.get("PGPASSWORD", ""))
    run_from_json(JSON_PATH, GROUP_ORDER, CONN)
