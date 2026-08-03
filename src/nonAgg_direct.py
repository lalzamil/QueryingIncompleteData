"""
Direct estimation for non-aggregation (set) queries — Section 5.2.

Computes membership probability P(y in q(T)) for each candidate output
value y using log-linearization:

  P(y in q(T)) = 1 - exp( SUM_t log(1 - P_tilde(t)) )

where P_tilde(t) = PROD_i psi_hat_{t,A_i} is the per-tuple predicate-
satisfaction probability assembled from the q_gamma queries of Section 5.1
(implemented in QueryRewriterExecuter.py).

This is more efficient than Section 4 (SetQueryRewriterExecuter) because
it avoids per-tuple probability computation — everything is done via
SQL aggregation.
"""

import os, re, json, time, math, sys
import psycopg2
import pandas as pd
import numpy as np
from io import StringIO
from typing import Dict, List, Any, Optional, Tuple
from CadeDiscretization import materialized_cade_source, prepare_cade_bins
# NOTE: QueryRewriterExecuter.py has module-level code that runs on import.
# We reuse the q_gamma SQL patterns directly here instead of importing.

CONN_PARAMS = dict(host=os.environ.get("PGHOST", "localhost"),
                   port=int(os.environ.get("PGPORT", "5433")),
                   dbname=os.environ.get("PGDATABASE", "mydb"),
                   user=os.environ.get("PGUSER", "postgres"),
                   password=os.environ.get("PGPASSWORD", ""))

JSON_PATH = "configs/mnar_set_queries.json"
DISCRETIZED_JSON_PATH = "/tmp/mnar_set_queries_discretized_safe.json"


def _qn(c):
    """Quote column name for SQL."""
    return '"%s"' % c.lower()


def _parse_set_query(query: str):
    """Parse SELECT Y FROM T WHERE theta [GROUP BY ...]."""
    m_sel = re.match(r"SELECT\s+(.+?)\s+FROM\s+(\S+)", query, re.IGNORECASE)
    if not m_sel:
        return None
    sel_raw = [s.strip().lower() for s in m_sel.group(1).split(",")]
    from_token = m_sel.group(2)

    m_where = re.search(r"WHERE\s+(.+?)(?:\s+GROUP\s+BY|$)", query, re.IGNORECASE)
    where_str = m_where.group(1).strip() if m_where else None

    m_gb = re.search(r"GROUP\s+BY\s+(.+?)(?:;|$)", query, re.IGNORECASE)
    group_cols = [c.strip().lower() for c in m_gb.group(1).split(",")] if m_gb else []

    return {
        "select_cols": sel_raw,
        "from_token": from_token,
        "where_str": where_str,
        "group_cols": group_cols,
    }


def _parse_where_clauses(where_str: str):
    """Parse WHERE into list of (col, op, val)."""
    if not where_str:
        return []
    clauses = re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE)
    result = []
    for clause in clauses:
        m = re.match(
            r"(\w+)\s*(!=|<>|>=|<=|>|<|=)\s*['\"]?([^'\"]*?)['\"]?\s*$",
            clause.strip())
        if m:
            result.append((m.group(1).lower(), m.group(2), m.group(3)))
    return result


def _compute_bin_cuts(conn, table, col, n_bins=10, card_threshold=20):
    """Return ascending quantile cut points for a continuous separator column,
    or None if the column is low-cardinality/discrete or non-numeric.
    Binning continuous separators avoids singleton strata (which bias the
    conditional estimate and collapse CI coverage)."""
    cur = conn.cursor()
    try:
        cur.execute('SELECT COUNT(DISTINCT %s) FROM %s WHERE %s IS NOT NULL'
                    % (_qn(col), table, _qn(col)))
        nd = cur.fetchone()[0]
    except Exception:
        conn.rollback()
        return None
    if not nd or nd <= card_threshold:
        return None
    qs = [i / float(n_bins) for i in range(1, n_bins)]
    arr = "ARRAY[%s]" % ",".join("%f" % q for q in qs)
    try:
        cur.execute('SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY %s::numeric) '
                    'FROM %s WHERE %s IS NOT NULL'
                    % (arr, _qn(col), table, _qn(col)))
        cuts = cur.fetchone()[0]
    except Exception:
        conn.rollback()
        return None  # non-numeric (e.g. high-card string) -> leave unbinned
    if not cuts:
        return None
    cuts = sorted(set(float(c) for c in cuts if c is not None))
    return cuts if len(cuts) >= 1 else None


def _bin_ref(col, cuts, prefix=""):
    """SQL expression for a (possibly binned) separator column reference."""
    if not cuts:
        return "%s%s" % (prefix, _qn(col))
    column = "%s%s" % (prefix, _qn(col))
    clauses = " ".join(
        "WHEN %s::double precision <= %.17g THEN %d" % (
            column,
            float(boundary),
            index,
        )
        for index, boundary in enumerate(cuts)
    )
    return "CASE WHEN %s IS NULL THEN NULL %s ELSE %d END" % (
        column,
        clauses,
        len(cuts),
    )


def _build_qgamma_cte(attr: str, sep_set: List[str], table: str,
                       where_clauses: List[tuple], bin_map: dict = None) -> str:
    """Build the q_gamma CTE for one incomplete attribute.
    Returns SQL for: P_hat(phi_attr | X = x) per separating-set group.
    Continuous separators in bin_map are quantile-binned to avoid singleton
    strata."""
    bin_map = bin_map or {}
    # binned group expressions, aliased back to the column name so the join matches
    sep_sel = ", ".join("%s AS %s" % (_bin_ref(c, bin_map.get(c)), _qn(c)) for c in sep_set)
    sep_grp = ", ".join(_bin_ref(c, bin_map.get(c)) for c in sep_set)

    phi_conditions = []
    for col, op, val in where_clauses:
        if col == attr:
            try:
                float(val)
                phi_conditions.append('(%s %s %s)' % (_qn(col), op, val))
            except ValueError:
                phi_conditions.append("(%s %s '%s')" % (_qn(col), op, val))

    if phi_conditions:
        phi_expr = " AND ".join(phi_conditions)
        psi_expr = "AVG(CASE WHEN %s THEN 1.0 ELSE 0.0 END)" % phi_expr
    else:
        psi_expr = "1.0"

    not_null_conds = ["%s IS NOT NULL" % _qn(attr)]
    for c in sep_set:
        not_null_conds.append("%s IS NOT NULL" % _qn(c))
    where_sql = " AND ".join(not_null_conds)

    cte = (
        "qg_%s AS (\n"
        "  SELECT %s,\n"
        "    %s AS psi_hat,\n"
        "    COUNT(%s) AS n_x,\n"
        "    VAR_SAMP(CASE WHEN %s THEN 1.0 ELSE 0.0 END) AS var_psi\n"
        "  FROM %s\n"
        "  WHERE %s\n"
        "  GROUP BY %s\n"
        ")" % (
            attr,
            sep_sel,
            psi_expr,
            _qn(attr),
            phi_conditions[0] if phi_conditions else "TRUE",
            table,
            where_sql,
            sep_grp,
        )
    )
    return cte


def run_direct_set_query(conn, query: str, table: str, gt_table: str,
                          missing_attrs: List[str],
                          ordering: Dict[str, List[str]],
                          df_check: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Run Section 5.2 direct estimation on a set query.

    1. Build q_gamma CTEs for each incomplete attr with predicates
    2. JOIN base table with q_gamma stats
    3. Compute P_tilde(t) = PROD psi_hat per tuple
    4. Compute L(t) = LN(1 - P_tilde(t))
    5. GROUP BY select cols: S_y = SUM(L(t))
    6. P(y in q) = 1 - EXP(S_y)
    """
    t0 = time.time()
    cur = conn.cursor()
    parsed = _parse_set_query(query)
    if not parsed:
        return {"error": "Cannot parse query"}

    sel_cols = parsed["select_cols"]
    where_clauses = _parse_where_clauses(parsed["where_str"])
    group_cols = parsed["group_cols"]

    output_cols = group_cols if group_cols else sel_cols
    bin_map = prepare_cade_bins(conn, table, ordering)

    # Only include attrs that have WHERE predicates — NOT just because they're in SELECT
    involved_missing = []
    for attr in missing_attrs:
        if attr in ordering:
            for col, op, val in where_clauses:
                if col == attr:
                    involved_missing.append(attr)
                    break

    ctes = []
    join_clauses = []
    psi_factors = []

    for attr in involved_missing:
        sep_set = ordering.get(attr, [])
        if not sep_set:
            continue
        cte = _build_qgamma_cte(
            attr, sep_set, table, where_clauses, bin_map
        )
        ctes.append(cte)

        join_cond = " AND ".join(
            "%s = qg_%s.%s" % (
                _bin_ref(c, bin_map.get(c), "T."), attr, _qn(c)
            ) for c in sep_set
        )
        join_clauses.append(
            "LEFT JOIN qg_%s ON %s" % (attr, join_cond)
        )

        psi_factors.append(
            "CASE WHEN T.%s IS NULL THEN COALESCE(qg_%s.psi_hat, 0.0) "
            "ELSE CASE WHEN %s THEN 1.0 ELSE 0.0 END END" % (
                _qn(attr), attr,
                _build_phi_sql(attr, where_clauses),
            )
        )

    det_where = []
    for col, op, val in where_clauses:
        if col not in involved_missing:
            try:
                float(val)
                det_where.append("T.%s %s %s" % (_qn(col), op, val))
            except ValueError:
                det_where.append("T.%s %s '%s'" % (_qn(col), op, val))

    if not psi_factors:
        p_tilde_expr = "1.0"
    else:
        p_tilde_expr = " * ".join("(%s)" % f for f in psi_factors)

    l_expr = "LN(GREATEST(1e-15, 1.0 - (%s)))" % p_tilde_expr

    out_cols_sql = ", ".join("T.%s" % _qn(c) for c in output_cols)
    gb_sql = ", ".join("T.%s" % _qn(c) for c in output_cols)

    where_sql = ""
    if det_where:
        where_sql = "WHERE " + " AND ".join(det_where)

    with_clause = "WITH " + ",\n".join(ctes) if ctes else ""

    var_l_expr = "VAR_SAMP(%s)" % l_expr

    # Check if any output col is a missing attr (Y incomplete case from Section 5.2.2)
    all_missing_set = set(a.lower() for a in missing_attrs)
    if df_check is not None:
        y_incomplete = [c for c in output_cols if c in all_missing_set
                        and c in df_check.columns and df_check[c].isna().any()]
    else:
        y_incomplete = [c for c in output_cols if c in all_missing_set]

    if not y_incomplete:
        # Y-complete case: standard GROUP BY Y, SUM L(t)
        sql = (
            "%s\n"
            "SELECT %s,\n"
            "  1.0 - EXP(GREATEST(-700, SUM(%s))) AS membership_prob,\n"
            "  SUM(%s) AS sum_L,\n"
            "  COUNT(*) AS n_tuples,\n"
            "  COALESCE(%s, 0) AS var_L\n"
            "FROM %s T\n"
            "%s\n"
            "%s\n"
            "GROUP BY %s\n"
            "ORDER BY membership_prob DESC"
        ) % (
            with_clause,
            out_cols_sql,
            l_expr,
            l_expr,
            var_l_expr,
            table,
            "\n".join(join_clauses),
            where_sql,
            gb_sql,
        )
    else:
        # Y-incomplete case (Section 5.2.2) — stratum-level computation:
        # S_y^obs: GROUP BY Y over rows where Y IS NOT NULL
        # S_y^miss: aggregate at STRATUM level (not per-row!):
        #   For each stratum x: count n_miss_x of null-Y rows and their shared P_tilde_x
        #   Then S_y^miss = SUM_x n_miss_x * LN(1 - P(Y=y|X_Y=x) * P_tilde_x)
        y_col = y_incomplete[0]
        y_sep = ordering.get(y_col, [])

        # Use only the FIRST separating-set column for the Y-distribution
        # conditional. Using all columns causes sparsity (1 obs per stratum).
        # The first column is the most important cause of missingness.
        y_sep_eff = y_sep[:1] if y_sep else []
        bin_aliases = ["xb_strata"]

        if y_sep_eff:
            bin_exprs = "%s AS xb_strata" % _qn(y_sep_eff[0])
        else:
            bin_exprs = "1 AS xb_strata"

        y_not_null = "T.%s IS NOT NULL" % _qn(y_col)
        if y_sep_eff:
            y_not_null += " AND T.%s IS NOT NULL" % _qn(y_sep_eff[0])

        # CTE: P(Y=y | X_Y_1=x) from observed data — uses first sep column only
        xb_ref = (
            _bin_ref(y_sep_eff[0], bin_map.get(y_sep_eff[0]), "T.")
            if y_sep_eff else "1"
        )
        cte_ydist = (
            "qg_ydist AS (\n"
            "  SELECT %s AS xb_strata, T.%s AS yval,\n"
            "    COUNT(*)::FLOAT / SUM(COUNT(*)) OVER (PARTITION BY %s) AS p_y_given_x\n"
            "  FROM %s T\n"
            "  WHERE %s\n"
            "  GROUP BY %s, T.%s\n"
            ")" % (
                xb_ref, _qn(y_col),
                xb_ref,
                table,
                y_not_null,
                xb_ref, _qn(y_col),
            )
        )
        ctes.append(cte_ydist)

        # CTE: stratum-level stats for null-Y rows
        miss_where = "T.%s IS NULL" % _qn(y_col)
        if det_where:
            miss_where += " AND " + " AND ".join(det_where)

        xb_miss_ref = (
            _bin_ref(y_sep_eff[0], bin_map.get(y_sep_eff[0]), "T.")
            if y_sep_eff else "1"
        )
        cte_strata_miss = (
            "strata_miss AS (\n"
            "  SELECT %s AS xb_strata,\n"
            "    COUNT(*) AS n_miss_x,\n"
            "    AVG(%s) AS p_tilde_x\n"
            "  FROM %s T\n"
            "  %s\n"
            "  WHERE %s\n"
            "  GROUP BY %s\n"
            ")" % (
                xb_miss_ref,
                p_tilde_expr if p_tilde_expr != "1.0" else "1.0",
                table,
                "\n".join(join_clauses),
                miss_where,
                xb_miss_ref,
            )
        )
        ctes.append(cte_strata_miss)
        with_clause = "WITH " + ",\n".join(ctes)

        miss_join_cond = " AND ".join(
            "sm.%s = yd.%s" % (a, a) for a in bin_aliases
        )

        det_and = ""
        if det_where:
            det_and = "AND " + " AND ".join(det_where)

        obs_joins = "\n".join(join_clauses)

        sql = (
            "{with_clause}\n"
            "SELECT COALESCE(obs.yval, miss.yval) AS {y_qn},\n"
            "  1.0 - EXP(GREATEST(-700, COALESCE(obs.s_obs, 0) + COALESCE(miss.s_miss, 0))) AS membership_prob,\n"
            "  COALESCE(obs.s_obs, 0) + COALESCE(miss.s_miss, 0) AS sum_L,\n"
            "  COALESCE(obs.n_obs, 0) + COALESCE(miss.n_miss, 0) AS n_tuples,\n"
            "  0 AS var_L\n"
            "FROM (\n"
            "  SELECT T.{y_qn} AS yval,\n"
            "    SUM({l_expr}) AS s_obs,\n"
            "    COUNT(*) AS n_obs\n"
            "  FROM {table} T\n"
            "  {obs_joins}\n"
            "  WHERE T.{y_qn} IS NOT NULL {det_and}\n"
            "  GROUP BY T.{y_qn}\n"
            ") obs\n"
            "FULL OUTER JOIN (\n"
            "  SELECT yd.yval,\n"
            "    SUM(sm.n_miss_x * LN(GREATEST(1e-15, 1.0 - yd.p_y_given_x * GREATEST(0, LEAST(1, sm.p_tilde_x))))) AS s_miss,\n"
            "    SUM(sm.n_miss_x) AS n_miss\n"
            "  FROM strata_miss sm\n"
            "  JOIN qg_ydist yd ON {miss_join}\n"
            "  GROUP BY yd.yval\n"
            ") miss ON obs.yval = miss.yval\n"
            "WHERE 1.0 - EXP(GREATEST(-700, COALESCE(obs.s_obs, 0) + COALESCE(miss.s_miss, 0))) > 1e-10\n"
            "ORDER BY membership_prob DESC"
        ).format(
            with_clause=with_clause,
            y_qn=_qn(y_col),
            l_expr=l_expr,
            table=table,
            obs_joins=obs_joins,
            det_and=det_and,
            miss_join=miss_join_cond,
        )

    elapsed_sql = 0.0
    try:
        t_sql = time.perf_counter()
        cur.execute(sql)
        rows = cur.fetchall()
        elapsed_sql = time.perf_counter() - t_sql
    except Exception as e:
        conn.rollback()
        return {"error": str(e), "sql": sql, "time_s": time.time() - t0}

    def _nv(v):
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

    pred_probs = {}
    pred_ci_widths = {}
    z_alpha = 1.96
    for row in rows:
        n_out = len(output_cols)
        key = tuple(_nv(row[i]) for i in range(n_out))
        prob = float(row[n_out])
        if prob <= 0 or prob >= 1:
            ci_half = 0.0
        else:
            n_tuples = float(row[n_out + 2]) if len(row) > n_out + 2 else 1.0
            var_L = float(row[n_out + 3]) if len(row) > n_out + 3 and row[n_out + 3] is not None else 0.0
            var_prob = ((1.0 - prob) ** 2) * n_tuples * var_L / max(n_tuples, 1)
            ci_half = z_alpha * math.sqrt(max(var_prob, 0.0))
        pred_probs[key] = max(pred_probs.get(key, 0.0), prob)
        pred_ci_widths[key] = 2.0 * ci_half

    try:
        gt_sql = query
        gt_sql = re.sub(r"FROM\s+\S+", "FROM %s" % gt_table, gt_sql, count=1, flags=re.IGNORECASE)
        cur.execute(gt_sql)
        gt_rows = cur.fetchall()
        gt_set = set()
        for r in gt_rows:
            gt_set.add(tuple(_nv(v) for v in r))
    except Exception as e:
        conn.rollback()
        gt_set = set()

    all_tuples = set(pred_probs.keys()) | gt_set
    z_pred = sum(pred_probs.values()) if pred_probs else 0.0
    z_gt = float(len(gt_set)) if gt_set else 0.0
    tv = 0.0
    for t in all_tuples:
        p_p = (pred_probs.get(t, 0.0) / z_pred) if z_pred > 0 else 0.0
        p_g = (1.0 / z_gt) if (t in gt_set and z_gt > 0) else 0.0
        tv += abs(p_p - p_g)
    tv /= 2.0

    mean_ci = float(np.mean(list(pred_ci_widths.values()))) if pred_ci_widths else 0.0
    mean_prob = float(np.mean(list(pred_probs.values()))) if pred_probs else 0.0
    delta_w = mean_ci / mean_prob if mean_prob > 1e-12 else float("inf")

    elapsed = time.time() - t0
    return {
        "tv_prob": tv,
        "delta_w": delta_w,
        "time_s": elapsed,
        "sql_time_s": elapsed_sql,
        "n_pred": len(pred_probs),
        "n_gt": len(gt_set),
    }


def _build_phi_sql(attr, where_clauses):
    """Build the phi(attr) boolean expression from WHERE clauses."""
    parts = []
    for col, op, val in where_clauses:
        if col == attr:
            try:
                float(val)
                parts.append("T.%s %s %s" % (_qn(col), op, val))
            except ValueError:
                parts.append("T.%s %s '%s'" % (_qn(col), op, val))
    return " AND ".join(parts) if parts else "TRUE"


def _load_table(conn, csv_path, table_name, force=False):
    cur = conn.cursor()
    if not force:
        cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=%s)",
                    (table_name.lower(),))
        if cur.fetchone()[0]:
            return
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    cols = list(df.columns)
    type_map = {'int64': 'BIGINT', 'float64': 'NUMERIC'}
    ddl = ", ".join('"%s" %s' % (c, type_map.get(str(df[c].dtype), 'TEXT')) for c in cols)
    cur.execute('DROP TABLE IF EXISTS "%s"' % table_name)
    cur.execute('CREATE TABLE "%s" (%s)' % (table_name, ddl))
    buf = StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    col_list = ", ".join('"%s"' % c for c in cols)
    cur.copy_expert('COPY "%s"(%s) FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')' % (table_name, col_list), buf)
    conn.commit()
    cur.execute('ANALYZE "%s"' % table_name)
    conn.commit()


# ═══════════════════════════════════════════════════════════════
#  Per-tuple direct estimation (comparable to Section 4 / Table 5)
# ═══════════════════════════════════════════════════════════════

def run_direct_per_tuple(conn, query: str, table: str, gt_table: str,
                          missing_attrs: List[str],
                          ordering: Dict[str, List[str]],
                          return_groups: bool = False) -> Dict[str, Any]:
    """
    Section 5.2 per-tuple estimation: compute P_tilde(t) for each tuple
    using q_gamma from Section 5.1, then compare to GT via TV_prob.

    Faster than Section 4: single SQL with pre-computed q_gamma JOINs.
    Output format matches Table 5: TV_prob, delta_w, QT.
    """
    t0 = time.time()
    cur = conn.cursor()
    parsed = _parse_set_query(query)
    if not parsed:
        return {"error": "Cannot parse query"}

    sel_cols = parsed["select_cols"]
    where_clauses = _parse_where_clauses(parsed["where_str"])
    group_cols = parsed["group_cols"]
    output_cols = group_cols if group_cols else sel_cols
    working_table, effective_ordering = materialized_cade_source(
        conn, table, ordering
    )
    bin_map = {}

    # Only build q_gamma for missing attrs actually USED by this query
    # (in SELECT or WHERE). This avoids expensive JOINs on unused attrs.
    query_attrs = set(sel_cols + [c for c, _, _ in where_clauses])
    ctes = []
    join_clauses = []
    psi_factors = []
    relvar_factors = []

    for attr in missing_attrs:
        sep_set = effective_ordering.get(attr, [])
        if not sep_set:
            continue
        if attr not in query_attrs:
            continue

        cte = _build_qgamma_cte(
            attr, sep_set, working_table, where_clauses, bin_map
        )
        ctes.append(cte)

        join_cond = " AND ".join(
            "%s = qg_%s.%s" % (
                _bin_ref(c, bin_map.get(c), "T."), attr, _qn(c)
            ) for c in sep_set
        )
        join_clauses.append(
            "LEFT JOIN qg_%s ON %s" % (attr, join_cond)
        )

        phi_sql = _build_phi_sql(attr, where_clauses)
        psi_factors.append(
            "CASE WHEN T.%s IS NULL THEN COALESCE(qg_%s.psi_hat, 0.0) "
            "ELSE CASE WHEN %s THEN 1.0 ELSE 0.0 END END" % (
                _qn(attr), attr, phi_sql,
            )
        )
        # Relative variance for Delta method: var_psi / (n_x * psi_hat^2)
        # Agresti-Coull Delta method relative variance (same as SetQueryRewriterExecuter)
        z2 = 1.96 ** 2
        p_adj = "(qg_%s.psi_hat * qg_%s.n_x + %f) / (qg_%s.n_x + %f)" % (
            attr, attr, z2 / 2, attr, z2)
        n_adj = "(qg_%s.n_x + %f)" % (attr, z2)
        relvar_factors.append(
            "CASE WHEN T.%s IS NULL AND qg_%s.n_x > 0 THEN "
            "(1.0 - %s) / NULLIF(%s * %s, 0) "
            "ELSE 0 END" % (_qn(attr), attr, p_adj, n_adj, p_adj)
        )

    # Deterministic WHERE predicates (on complete attrs)
    det_where = []
    for col, op, val in where_clauses:
        if col not in [a for a in missing_attrs]:
            try:
                float(val)
                det_where.append("T.%s %s %s" % (_qn(col), op, val))
            except ValueError:
                det_where.append("T.%s %s '%s'" % (_qn(col), op, val))

    if psi_factors:
        p_tilde_expr = " * ".join("(%s)" % f for f in psi_factors)
    else:
        p_tilde_expr = "1.0"

    if relvar_factors:
        relvar_expr = " + ".join("(%s)" % f for f in relvar_factors)
    else:
        relvar_expr = "0"

    where_sql = "WHERE " + " AND ".join(det_where) if det_where else ""

    with_clause = "WITH " + ",\n".join(ctes) if ctes else ""

    out_cols_sql = ", ".join("T.%s" % _qn(c) for c in output_cols)
    gb_sql = ", ".join("T.%s" % _qn(c) for c in output_cols)

    # ── Section 5.2 estimation SQL ──
    # GROUP BY separating-set columns (X_1), compute:
    #   - p_hat = AVG(psi_hat product) within group (the q_gamma rate)
    #   - n_cert = count of certain tuples (all observed, predicate satisfied)
    #   - n_poss = count of possible tuples (has missing attr)
    #   - n_total = total rows in group
    # This matches Section 4's output structure for fair TV comparison.

    # Group by the query's output columns — this is what the set query returns.
    # Using separating-set columns would create too many groups on large datasets.
    group_key_cols = output_cols

    gk_sql = ", ".join("T.%s" % _qn(c) for c in group_key_cols)

    # Identify which rows have ANY relevant missing attr → "possible"
    relevant_missing = [
        a for a in missing_attrs
        if a in query_attrs and a in effective_ordering
    ]
    has_missing_expr = " OR ".join("T.%s IS NULL" % _qn(a) for a in relevant_missing) or "FALSE"

    sql = (
        "{with_clause}\n"
        "SELECT {gk_cols},\n"
        "  AVG({p_tilde}) AS p_hat,\n"
        "  SUM(CASE WHEN NOT ({has_miss}) AND ({p_tilde}) >= 1.0 THEN 1 ELSE 0 END) AS n_cert,\n"
        "  SUM(CASE WHEN ({has_miss}) THEN 1 ELSE 0 END) AS n_poss,\n"
        "  COUNT(*) AS n_total,\n"
        "  AVG(CASE WHEN ({has_miss}) THEN POWER({p_tilde}, 2) * ({relvar}) ELSE NULL END) AS avg_delta_var\n"
        "FROM {table} T\n"
        "{joins}\n"
        "{where}\n"
        "GROUP BY {gk_cols}"
    ).format(
        with_clause=with_clause,
        gk_cols=gk_sql,
        p_tilde=p_tilde_expr,
        has_miss=has_missing_expr,
        relvar=relvar_expr,
        table=working_table,
        joins="\n".join(join_clauses),
        where=where_sql,
    )

    elapsed_sql = 0.0
    try:
        try:
            conn.rollback()
        except Exception:
            pass
        t_sql = time.perf_counter()
        cur.execute(sql)
        pred_rows = cur.fetchall()
        elapsed_sql = time.perf_counter() - t_sql
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"error": str(e), "sql": sql, "time_s": time.time() - t0}

    def _nv(v):
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

    n_gk = len(group_key_cols)
    pred_prob = {}
    pred_mass = {}
    pred_ci = {}
    pred_ci_half = {}
    total_cert = 0
    total_poss = 0
    z_alpha = 1.96
    for row in pred_rows:
        gk = tuple(_nv(row[i]) for i in range(n_gk))
        p_hat = float(row[n_gk]) if row[n_gk] is not None else 0.0
        n_cert = int(row[n_gk + 1]) if row[n_gk + 1] else 0
        n_poss = int(row[n_gk + 2]) if row[n_gk + 2] else 0
        n_total = int(row[n_gk + 3]) if row[n_gk + 3] else 1
        avg_dv = float(row[n_gk + 4]) if row[n_gk + 4] is not None else 0.0
        pred_prob[gk] = p_hat
        pred_mass[gk] = p_hat * (n_cert + n_poss)
        # Delta method: CI_half = z * p_hat * sqrt(avg_sum_relvar)
        # avg_dv = avg(p_tilde^2 * sum_relvar) over uncertain rows
        # For CI width: sqrt(avg_dv) gives the std of p_tilde
        ci_half = z_alpha * math.sqrt(max(avg_dv, 0.0))
        pred_ci[gk] = 2.0 * ci_half
        pred_ci_half[gk] = ci_half
        total_cert += n_cert
        total_poss += n_poss

    gt_groups = {}
    gt_totals = {}   # gk -> n_total
    try:
        try:
            conn.rollback()
        except Exception:
            pass
        gt_gk_sql = ", ".join("%s" % _qn(c) for c in group_key_cols)
        from_token = parsed["from_token"]
        w = parsed["where_str"]
        gt_where = ("WHERE %s" % w) if w else ""

        gt_qual_sql = (
            "SELECT {gk}, COUNT(*) AS n_qual\n"
            "FROM {gt_table}\n"
            "{where}\n"
            "GROUP BY {gk}"
        ).format(gk=gt_gk_sql, gt_table=gt_table, where=gt_where)
        ss_cur = conn.cursor("gt_qual_cur")
        ss_cur.itersize = 5000
        ss_cur.execute(gt_qual_sql)
        while True:
            batch = ss_cur.fetchmany(5000)
            if not batch:
                break
            for row in batch:
                gk = tuple(_nv(row[i]) for i in range(n_gk))
                gt_groups[gk] = int(row[n_gk])
        ss_cur.close()
        conn.commit()

        gt_total_sql = (
            "SELECT {gk}, COUNT(*) AS n_total\n"
            "FROM {gt_table}\n"
            "GROUP BY {gk}"
        ).format(gk=gt_gk_sql, gt_table=gt_table)
        ss_cur2 = conn.cursor("gt_total_cur")
        ss_cur2.itersize = 5000
        ss_cur2.execute(gt_total_sql)
        while True:
            batch = ss_cur2.fetchmany(5000)
            if not batch:
                break
            for row in batch:
                gk = tuple(_nv(row[i]) for i in range(n_gk))
                gt_totals[gk] = int(row[n_gk])
        ss_cur2.close()
        conn.commit()
    except Exception as e:
        conn.rollback()
        gt_groups = {}
        gt_totals = {}
        print("    [GT ERROR] %s" % e, flush=True)

    gt_mass = {gk: float(n) for gk, n in gt_groups.items()}

    # ── TV_prob computation (same as RunnerSetQueriy._tv_prob_conditional) ──
    # Pre-index pred_rows by group key for O(1) lookup
    pred_row_map = {}
    for row in pred_rows:
        rk = tuple(_nv(row[i]) for i in range(n_gk))
        pred_row_map[rk] = row

    all_keys = set(pred_prob.keys()) | set(gt_groups.keys()) | set(gt_totals.keys())
    eps = 1e-12
    groups_info = []
    group_detail = []  # per-group records for coverage/bias (only when return_groups)
    z_pred = 0.0
    z_oracle = 0.0

    for gk in all_keys:
        p_hat = pred_prob.get(gk, 0.0)
        gt_qualifying = gt_groups.get(gk, 0)
        group_total = gt_totals.get(gk, 0)
        p_oracle = gt_qualifying / group_total if group_total > 0 else 0.0

        prow = pred_row_map.get(gk)
        n_cert_g = int(prow[n_gk + 1]) if prow and prow[n_gk + 1] else 0
        n_poss_g = int(prow[n_gk + 2]) if prow and prow[n_gk + 2] else 0

        z_pred += n_cert_g * 1.0 + n_poss_g * p_hat
        z_oracle += gt_qualifying * 1.0
        groups_info.append((n_cert_g, n_poss_g, p_hat, p_oracle))
        if return_groups:
            group_detail.append({
                "gk": gk,
                "p_hat": p_hat,
                "ci_half": pred_ci_half.get(gk, 0.0),  # Agresti-Coull Delta half-width
                "p_oracle": p_oracle,
                "n_cert": n_cert_g,
                "n_poss": n_poss_g,
                "gt_qualifying": gt_qualifying,
                "group_total": group_total,
            })

    inv_zp = (1.0 / z_pred) if z_pred > eps else 0.0
    inv_zo = (1.0 / z_oracle) if z_oracle > eps else 0.0

    tv_sum = 0.0
    for n_c, n_p, p_h, p_o in groups_info:
        tv_sum += n_c * abs(inv_zp - inv_zo)
        tv_sum += n_p * abs(p_h * inv_zp - p_o * inv_zo)
    tv = 0.5 * tv_sum

    # ── TV_set via mass distributions (same as _tv_prob_maps) ──
    all_mass_keys = set(pred_mass.keys()) | set(gt_mass.keys())
    sp = sum(max(0, v) for v in pred_mass.values())
    sg = sum(max(0, v) for v in gt_mass.values())
    tv_set = 0.0
    for k in all_mass_keys:
        pp = (max(0, pred_mass.get(k, 0)) / sp) if sp > eps else 0.0
        pg = (max(0, gt_mass.get(k, 0)) / sg) if sg > eps else 0.0
        tv_set += abs(pp - pg)
    tv_set *= 0.5

    mean_ci = float(np.mean(list(pred_ci.values()))) if pred_ci else 0.0
    mean_prob = float(np.mean(list(pred_prob.values()))) if pred_prob else 0.0
    delta_w = mean_ci / mean_prob if mean_prob > 1e-12 else float("inf")

    elapsed = time.time() - t0
    result = {
        "tv_prob": tv,
        "tv_set": tv_set,
        "delta_w": delta_w,
        "time_s": elapsed,
        "sql_time_s": elapsed_sql,
        "n_pred": total_cert + total_poss,
        "n_gt": sum(gt_groups.values()),
    }
    if return_groups:
        result["groups"] = group_detail
    return result


# ═══════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════

def main():
    conn = psycopg2.connect(**CONN_PARAMS)
    conn.autocommit = False

    json_path = JSON_PATH
    if os.path.isfile(DISCRETIZED_JSON_PATH):
        json_path = DISCRETIZED_JSON_PATH
        print("Using discretized JSON: %s" % json_path, flush=True)
    else:
        print("Using original JSON: %s (no discretized version found)" % json_path, flush=True)

    with open(json_path) as f:
        cfg = json.load(f)

    results = []

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

            # Load ALL tables (including join tables)
            csv_to_table = {}
            for cp, tn in zip(csvs, tables):
                _load_table(conn, cp, tn)
                csv_to_table[cp] = tn
                # Also map basename and without-extension
                csv_to_table[os.path.basename(cp)] = tn
                stem = os.path.splitext(os.path.basename(cp))[0]
                dir_name = os.path.basename(os.path.dirname(cp)) or os.path.dirname(cp)
                csv_to_table["%s/%s" % (dir_name, os.path.basename(cp))] = tn
                csv_to_table["%s/%s" % (dir_name, stem)] = tn
            gt_csv_to_table = {}
            for gc, gt in zip(complete_csvs, complete_tables):
                _load_table(conn, gc, gt)
                gt_csv_to_table[gc] = gt
                gt_csv_to_table[os.path.basename(gc)] = gt
                stem = os.path.splitext(os.path.basename(gc))[0]
                dir_name = os.path.basename(os.path.dirname(gc)) or os.path.dirname(gc)
                gt_csv_to_table["%s/%s" % (dir_name, os.path.basename(gc))] = gt
                gt_csv_to_table["%s/%s" % (dir_name, stem)] = gt
            # For GT join queries: map MNAR CSV paths → GT table names (by position)
            mnar_to_gt = {}
            for cp, gt in zip(csvs, complete_tables):
                mnar_to_gt[cp] = gt
                mnar_to_gt[os.path.basename(cp)] = gt
                stem = os.path.splitext(os.path.basename(cp))[0]
                dir_name = os.path.basename(os.path.dirname(cp)) or os.path.dirname(cp)
                mnar_to_gt["%s/%s" % (dir_name, os.path.basename(cp))] = gt
                mnar_to_gt["%s/%s" % (dir_name, stem)] = gt

            # Create indexes on separating-set columns for fast q_gamma JOINs
            try:
                cur = conn.cursor()
                for attr in missing_attrs:
                    sep = ordering.get(attr, [])
                    if sep:
                        idx_cols = ", ".join('"%s"' % c for c in sep)
                        idx_name = "idx_%s_%s" % (table_name, attr)
                        cur.execute('CREATE INDEX IF NOT EXISTS %s ON %s (%s)' % (
                            idx_name, table_name, idx_cols))
                cur.execute('ANALYZE %s' % table_name)
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass

            print("  Loaded: %s -> %s" % (os.path.basename(csv_path), table_name),
                  flush=True)

            df_check = pd.read_csv(csv_path)
            df_check.columns = df_check.columns.str.lower()

            for qi, query in enumerate(queries):
                # Replace CSV paths with table names for JOIN queries
                q_sql = query
                for csv_key in sorted(csv_to_table.keys(), key=len, reverse=True):
                    q_sql = q_sql.replace(csv_key, csv_to_table[csv_key])
                # Also build GT table name for join queries
                q_gt = query
                for csv_key in sorted(mnar_to_gt.keys(), key=len, reverse=True):
                    q_gt = q_gt.replace(csv_key, mnar_to_gt[csv_key])

                print("\n  Q%d/%d: %s" % (qi + 1, len(queries), query[:90]),
                      flush=True)

                is_join = "JOIN" in query.upper()
                if is_join:
                    # For join queries: run SQL directly after CSV→table replacement
                    # q_sql has MNAR table names, q_gt has GT table names
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        cur = conn.cursor()
                        t_sql = time.perf_counter()
                        cur.execute(q_sql)
                        pred_rows = cur.fetchall()
                        qt = time.perf_counter() - t_sql
                        conn.commit()
                        cur.execute(q_gt)
                        gt_rows = cur.fetchall()
                        conn.commit()
                        def _nv2(v):
                            s = str(v)
                            try:
                                f = float(s)
                                return str(int(f)) if f == int(f) else s
                            except (ValueError, TypeError):
                                return s
                        pred_set = set(tuple(_nv2(v) for v in r) for r in pred_rows)
                        gt_set = set(tuple(_nv2(v) for v in r) for r in gt_rows)
                        all_t = pred_set | gt_set
                        zp = float(len(pred_set)) if pred_set else 0.0
                        zg = float(len(gt_set)) if gt_set else 0.0
                        tv = 0.0
                        for t in all_t:
                            pp = (1.0 / zp) if t in pred_set and zp > 0 else 0.0
                            pg = (1.0 / zg) if t in gt_set and zg > 0 else 0.0
                            tv += abs(pp - pg)
                        tv /= 2.0
                        result = {"tv_prob": tv, "delta_w": 0.0,
                                  "sql_time_s": qt, "n_pred": len(pred_set),
                                  "n_gt": len(gt_set)}
                    except Exception as e:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        result = {"error": str(e), "tv_prob": 0.0, "delta_w": 0.0,
                                  "sql_time_s": 0.0, "n_pred": 0, "n_gt": 0}
                else:
                    result = run_direct_per_tuple(
                        conn, q_sql, table_name, gt_table or table_name,
                        missing_attrs, ordering,
                    )

                err = result.get("error")
                if err:
                    print("    ERROR: %s" % err, flush=True)
                    if "sql" in result:
                        print("    SQL: %s" % result["sql"][:200], flush=True)
                else:
                    print("    TVD=%.4f  dw=%.4f  QT=%.3fs  |pred|=%d  |GT|=%d" % (
                        result["tv_prob"], result.get("delta_w", 0),
                        result["sql_time_s"],
                        result["n_pred"], result["n_gt"]), flush=True)

                results.append({
                    "group": group_key, "block": block_key,
                    "query_idx": qi + 1,
                    "method": "Direct-5.2",
                    "tv_prob": result.get("tv_prob", np.nan),
                    "delta_w": result.get("delta_w", np.nan),
                    "QT": result.get("sql_time_s", np.nan),
                    "n_pred": result.get("n_pred", 0),
                    "n_gt": result.get("n_gt", 0),
                    "error": result.get("error", ""),
                })

    conn.close()

    df_out = pd.DataFrame(results)
    out_path = "nonAgg_direct_results.csv"
    df_out.to_csv(out_path, index=False)

    print("\n" + "=" * 60, flush=True)
    print("Results saved to %s" % out_path, flush=True)
    print("=" * 60, flush=True)

    if len(df_out) > 0:
        good = df_out[df_out["error"] == ""]
        avg = good.groupby("block").agg(
            TVD=("tv_prob", "mean"),
            delta_w=("delta_w", "mean"),
            QT=("QT", "mean"),
        ).reset_index()
        print("\n--- Table 5 format (Direct Estimation, Section 5.2) ---", flush=True)
        print(avg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
