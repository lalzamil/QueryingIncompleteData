"""
Base query rewriter and executor for incomplete data (no-opt).
Implements Lubna_VLDB_2026_cursor.pdf §4.2 (Answering Queries in the Presence
of Missing Data): ordered separating sets (Def. 4.1), cell-based distribution
estimation, selection (§4.2.1), projection/selection (§4.2.2), GROUP BY (§4.2.3),
and join rewriting (§4.2.4). No estimation-aware optimizations (§4.3).
"""
import math
import pandas as pd
import psycopg2
import re
from typing import Dict, List, Optional, Any
import psycopg2
import csv
from io import StringIO

# Z critical value for 95% two-sided (default); computed once.
_Z_TABLE = {0.01: 2.5758, 0.05: 1.9600, 0.10: 1.6449}

def _z_critical(alpha: float) -> float:
    if alpha in _Z_TABLE:
        return _Z_TABLE[alpha]
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2))

def _interval_bounds_sql(mode: str, cell_prob_expr: str, den_expr: str, alpha_f: float):
    """Return (lower_sql, upper_sql) for a cell probability interval."""
    if mode == "hoeffding":
        eps = f"SQRT(LN(2.0 / {alpha_f}) / (2.0 * {den_expr}))"
        return (f"GREATEST(0.0, {cell_prob_expr} - {eps})",
                f"LEAST(1.0, {cell_prob_expr} + {eps})")
    z = _z_critical(alpha_f)
    z2 = z * z
    if mode == "wilson":
        denom = f"(1.0 + {z2} / {den_expr})"
        center = f"(({cell_prob_expr} + {z2} / (2.0 * {den_expr})) / {denom})"
        half = (f"({z} * SQRT({cell_prob_expr} * (1.0 - {cell_prob_expr}) / {den_expr}"
                f" + {z2} / (4.0 * {den_expr} * {den_expr})) / {denom})")
        return (f"GREATEST(0.0, {center} - {half})",
                f"LEAST(1.0, {center} + {half})")
    # CLT / Wald
    eps = f"({z} * SQRT({cell_prob_expr} * (1.0 - {cell_prob_expr}) / {den_expr}))"
    return (f"GREATEST(0.0, {cell_prob_expr} - {eps})",
            f"LEAST(1.0, {cell_prob_expr} + {eps})")

def _delta_relvar_sql(cell_prob_expr: str, den_expr: str, alpha: float) -> str:
    """
    Agresti-Coull adjusted relative variance for the delta method.
    Non-zero even at p-hat in {0,1}, avoiding degenerate boundary intervals.
    """
    z2 = _z_critical(alpha) ** 2
    p_adj = f"(({cell_prob_expr} * {den_expr} + {z2 / 2}) / ({den_expr} + {z2}))"
    n_adj = f"({den_expr} + {z2})"
    return (
        f"CASE WHEN {den_expr} > 0 "
        f"THEN (1.0 - {p_adj}) / ({n_adj} * {p_adj}) "
        f"ELSE 0 END"
    )


class QueryExecutor:
    TYPE_MAP = {'int64':'BIGINT','float64':'NUMERIC'}

    def __init__(self, conn: psycopg2.extensions.connection, csv_queries: Dict[str, Any], skip_prepare: bool = False):
        """
        csv_queries: dict mapping a key to a dict with:
          - 'csv':  path or list of paths to CSV file(s)
          - 'table': name or list of table names to create
          - optionally 'complete_csv'/'complete_table' for ground-truth tables
        skip_prepare: if True, do not create/load tables (caller already did; reuse same table names).
        """
        self.conn = conn
        self.cur  = conn.cursor()
        self.csv_queries = csv_queries
        # Cache of columns already indexed per table in this executor lifetime.
        self._indexed_cols: Dict[str, set] = {}

        # map each CSV path -> its table name
        self.mar_table_for = {}
        for meta in csv_queries.values():
            paths  = meta['csv']   if isinstance(meta['csv'],   list) else [meta['csv']]
            tables = meta['table'] if isinstance(meta['table'], list) else [meta['table']]
            for p,t in zip(paths, tables):
                self.mar_table_for[p] = t

        # map any complete_csv -> complete_table (ground truth)
        self.full_table_for = {}
        self.groundTR = any(meta.get('complete_csv') for meta in csv_queries.values())
        for meta in csv_queries.values():
            if meta.get('complete_csv'):
                paths  = meta['complete_csv']   if isinstance(meta['complete_csv'],   list) else [meta['complete_csv']]
                tables = meta['complete_table'] if isinstance(meta['complete_table'], list) else [meta['complete_table']]
                for p,t in zip(paths, tables):
                    self.full_table_for[p] = t

        # actually create + load all tables (unless reusing tables from another executor)
        if not skip_prepare:
            self.prepareTables()

    def inferSchema(self, path: str) -> Dict[str,str]:
        # Sample more rows to avoid mistyping mixed columns (e.g. asin looks
        # numeric in first 100 rows but contains alphanumeric values later).
        df = pd.read_csv(path, nrows=1000, low_memory=False)
        df.columns = df.columns.str.lower()
        df = df.loc[:, ~df.columns.str.startswith('unnamed:')]
        return {col: self.TYPE_MAP.get(str(dt),'TEXT') for col,dt in df.dtypes.items()}

    def loadCSV(self, path: str, table_name: str, cols: List[str]):
        df = pd.read_csv(path, keep_default_na=True, na_values=['',' ','\\N'])
        df.columns = df.columns.str.lower()
        df = df.loc[:, ~df.columns.str.startswith('unnamed:')]
        df = df[cols]
        buf = StringIO()
        df.to_csv(buf, index=False, header=False, na_rep='\\N', quoting=csv.QUOTE_MINIMAL)
        buf.seek(0)
        col_list = ', '.join(f'"{c}"' for c in cols)
        sql = f"COPY {table_name}({col_list}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
        self.cur.copy_expert(sql, buf)
        self.conn.commit()

    def prepareTables(self):
        # load all MAR/MNAR CSVs
        for path, tbl in self.mar_table_for.items():
            schema = self.inferSchema(path)
            cols   = list(schema)
            ddl    = ',\n  '.join(f'"{c}" {t}' for c,t in schema.items())
            self.cur.execute(f"""
                DROP TABLE IF EXISTS {tbl};
                CREATE TABLE {tbl} ({ddl});
                TRUNCATE {tbl};
            """)
            self.conn.commit()
            self.loadCSV(path, tbl, cols)

        # load any ground-truth complete tables
        if self.groundTR:
            for path, tbl in self.full_table_for.items():
                schema = self.inferSchema(path)
                cols   = list(schema)
                ddl    = ',\n  '.join(f'"{c}" {t}' for c,t in schema.items())
                self.cur.execute(f"""
                    DROP TABLE IF EXISTS {tbl};
                    CREATE TABLE {tbl} ({ddl});
                    TRUNCATE {tbl};
                """)
                self.conn.commit()
                self.loadCSV(path, tbl, cols)


    def _table_columns(self, table_name: str):
        """Return set of column names for the given table."""
        self.cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
        """, (table_name.strip('"'),))
        return {r[0] for r in self.cur.fetchall()}

    def _extract_sql_identifiers(self, sql: str) -> set:
        """
        Extract likely column identifiers from SQL text (WHERE/GROUP/JOIN/SELECT names).
        Conservative: returns tokens that look like identifiers; caller intersects with real table columns.
        """
        toks = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sql))
        stop = {
            "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "JOIN", "USING", "ON", "AS",
            "AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE", "CASE", "WHEN", "THEN", "ELSE", "END",
            "SUM", "COUNT", "AVG", "MIN", "MAX", "ARRAY_AGG", "DISTINCT", "LIKE", "BETWEEN", "WITH",
            "LEFT", "RIGHT", "FULL", "OUTER", "INNER", "CROSS", "HAVING"
        }
        return {t.lower() for t in toks if t.upper() not in stop}

    def _ensure_support_indexes(self, sql: str, table_names: List[str]) -> None:
        """
        Create lightweight per-column btree indexes for query-referenced columns.
        Works for any dataset because candidates come from SQL text and are intersected
        with actual table columns.
        """
        idents = self._extract_sql_identifiers(sql)
        for t in table_names:
            t_clean = t.strip('"')
            cols = self._table_columns(t_clean)
            if t_clean not in self._indexed_cols:
                self._indexed_cols[t_clean] = set()
            for c in sorted(idents & cols):
                if c in self._indexed_cols[t_clean]:
                    continue
                idx_name = f"idx_auto_{t_clean}_{c}"
                self.cur.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{t_clean}" ("{c}")')
                self._indexed_cols[t_clean].add(c)
        self.conn.commit()
    def _normalize_query(self, base_query: str) -> str:
        """
        Replace any occurrence of a CSV filename in FROM with its real table name.
        """
        sql = base_query
        for csv_path, tbl in self.mar_table_for.items():
            sql = sql.replace(csv_path, tbl)
        for csv_path, tbl in self.full_table_for.items():
            sql = sql.replace(csv_path, tbl)
        return sql

    def run(self,
            base_query: str,
            ordering_T: Dict[str, List[str]],
            missing_T: List[str],
            ordering_S: Optional[Dict[str, List[str]]] = None,
            missing_S: Optional[List[str]] = None,
            join_key: Optional[str] = None,
            score_threshold: float = 0.0) -> List[Any]:
        """
        1) normalize FROM <file.csv> → FROM <table>
        2) rewrite (single vs join decided by SQL, not just by presence of ordering_S)
        3) execute
        """
        # 1) Normalize
        sql1 = self._normalize_query(base_query.strip().rstrip(';'))

        # Use JOIN path whenever SQL has JOIN ... USING (...); config can have empty ordering_S/missing_S.
        wants_join = re.search(r"\bJOIN\b.*\bUSING\s*\(", sql1, re.IGNORECASE) is not None

        # 2) Rewrite
        _iv_mode = getattr(self, 'interval_mode', None)
        _iv_alpha = getattr(self, 'interval_alpha', None)

        if wants_join:
            # (optional) look up real columns to route WHERE clauses correctly
            m_ts = re.search(r"FROM\s+(?P<T>\S+)\s+JOIN\s+(?P<S>\S+)\s+USING\s*\(", sql1, re.IGNORECASE)
            t_cols = s_cols = None
            if m_ts:
                Tname = m_ts.group('T').strip('"')
                Sname = m_ts.group('S').strip('"')
                # requires _table_columns helper (see note below)
                t_cols = self._table_columns(Tname)
                s_cols = self._table_columns(Sname)
                self._ensure_support_indexes(sql1, [Tname, Sname])

            qr = QueryRewriter(
                sql1, ordering_T, missing_T,
                ordering_S, missing_S, join_key,
                score_threshold,
                t_columns=t_cols, s_columns=s_cols
            )
            if _iv_mode:
                qr.interval_mode = _iv_mode
                qr.interval_alpha = _iv_alpha
            sql2 = qr.JoinQueryRewriter()
        else:
            m_t = re.search(r"FROM\s+(\S+)", sql1, re.IGNORECASE)
            if m_t:
                self._ensure_support_indexes(sql1, [m_t.group(1)])
            qr = QueryRewriter(
                sql1, ordering_T, missing_T,
                None, None, None,
                score_threshold
            )
            if _iv_mode:
                qr.interval_mode = _iv_mode
                qr.interval_alpha = _iv_alpha
            sql2 = qr.groupLevelQueryRewriter()

        # 3) Execute
        import time as _t
        _t0 = _t.perf_counter()
        self.cur.execute(sql2)
        rows = self.cur.fetchall()
        self._sql_elapsed = _t.perf_counter() - _t0
        return rows


class QueryRewriter:
    def __init__(self, base_query: str, ordering_T: Dict[str, List[str]], missing_T: List[str], ordering_S: Optional[Dict[str, List[str]]] = None,
        missing_S: Optional[List[str]] = None, join_key: Optional[str] = None, score_threshold: float = 0.0,
        t_columns: Optional[set] = None, s_columns: Optional[set] = None):
        self.base_query = base_query.strip().rstrip(';')
        self.base_alias = 'base'
        self.score_threshold = score_threshold
        self.t_columns = set(t_columns or [])
        self.s_columns = set(s_columns or [])
        self.join_key = []


        m_join = re.search(
            r"SELECT\s+(?P<cols>.*?)\s+FROM\s+"
            r"(?P<T>\S+)\s+JOIN\s+(?P<S>\S+)\s+USING\s+(?P<using>\(?.*?\)?)"
            r"(?:\s+WHERE\s+(?P<where>.*?))?"
            r"(?:\s+GROUP\s+BY\s+(?P<groupby>.*?))?"
            r"(?=\s+ORDER\s+BY|$)",
            self.base_query,
            re.IGNORECASE | re.DOTALL,
        )

        # If query has JOIN ... USING, use join mode (default empty ordering_S/missing_S if not provided).
        if m_join:
            ordering_S = ordering_S if ordering_S is not None else {}
            missing_S = missing_S if missing_S is not None else []
            self.join_mode    = True
            self.ordering_T   = ordering_T
            self.missing_T    = missing_T
            self.ordering_S   = ordering_S
            self.missing_S    = missing_S
            self.join_key     = join_key


            # the join’s SELECT list (e.g. ["ID"])
            self.select_cols = [c.strip() for c in m_join.group('cols').split(',')]
            self.T           = m_join.group('T')
            self.S           = m_join.group('S')
            self.using_clause   = (m_join.group('using' or '')).strip()
            if self.using_clause.startswith('(') and self.using_clause.endswith(')'):
                self.using_clause = self.using_clause[1:-1]

            self.join_key = [k.strip().strip('"') for k in self.using_clause.split(',') if k.strip()]
            if not self.join_key:
                raise ValueError("USING(...) must include at least one join key.")

            where_str        = m_join.group('where') or ''
            all_conds        = [c.strip() for c in where_str.split('AND')] \
                                   if where_str else []

            # # split the WHERE conditions by which table’s attrs they mention
            # self.where_T = [
            #     c for c in all_conds
            #     if any(attr in c for attr in self.ordering_T.keys())
            # ]
            # self.where_S = [
            #     c for c in all_conds
            #     if any(attr in c for attr in self.ordering_S.keys())
            # ]


            def _columns_mentioned(expr: str):
                # crude identifier grabber; ignore SQL keywords
                toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr)
                stop = {"AND","OR","NOT","IN","IS","NULL","LIKE","BETWEEN","CASE","WHEN","THEN","END",
                        "TRUE","FALSE","ON","USING","WHERE","GROUP","BY","AS","SELECT","FROM","JOIN"}
                return [t for t in toks if t.upper() not in stop]

            self.where_T, self.where_S = [], []
            for c in all_conds:
                cols = _columns_mentioned(c)
                in_T = any(col in self.t_columns for col in cols) if self.t_columns else False
                in_S = any(col in self.s_columns for col in cols) if self.s_columns else False

                if in_T and not in_S:
                    self.where_T.append(c)
                elif in_S and not in_T:
                    self.where_S.append(c)
                else:
                    # fallback heuristics if ambiguous
                    if any(attr in c for attr in self.ordering_T.keys()):
                        self.where_T.append(c)
                    elif any(attr in c for attr in self.ordering_S.keys()):
                        self.where_S.append(c)
                    else:
                        # final default: T
                        self.where_T.append(c)
            #remaining predicates to T-side
            unclaimed = [c for c in all_conds if c not in self.where_T and c not in self.where_S]
            for c in unclaimed:
                if any(k in c for k in self.join_key):
                    self.where_T.append(c)
                else:
                    # If it mentions an S-only attribute name, send to S; else T
                    if any(attr in c for attr in self.ordering_S.keys()):
                        self.where_S.append(c)
                    else:
                        self.where_T.append(c)

            gb_str = (m_join.group('groupby') or '').strip()
            self.group_by_cols = [c.strip() for c in gb_str.split(',')] if gb_str else []

        else:
            # Single‐relation mode
            self.join_mode     = False
            self.ordering      = ordering_T
            self.missing_attrs = missing_T

            m = re.search(
                r"SELECT\s+(?P<cols>.*?)\s+FROM\s+(?P<table>\S+)"
                r"(?:\s+WHERE\s+(?P<where>.*?))?"
                r"(?:\s+GROUP\s+BY\s+(?P<groupby>.*?))?"
                r"(?=\s+ORDER\s+BY|$)",
                self.base_query, re.IGNORECASE | re.DOTALL
            )

            if not m:
                raise ValueError(
                    "Query must be: SELECT <cols> FROM <table> [WHERE ...]"
                )
            cols_str  = m.group('cols')
            table     = m.group('table')
            where_str = m.group('where') or ''
            gb_str    = (m.group('groupby') or '').strip()
            self.table       = table
            self.select_cols = [c.strip() for c in cols_str.split(',')]
            self.where_conds = [c.strip() for c in re.split(r"\bAND\b", where_str, flags=re.IGNORECASE)] if where_str else []
            self.group_by_cols = [c.strip() for c in gb_str.split(',')] if gb_str else []

            # Build factors only for query-relevant missing attrs.
            # Missing attrs that do not appear in SELECT/WHERE/GROUP BY do not affect
            # set membership for this query and should not gate tuple inclusion.
            query_text = " ".join(self.select_cols + self.where_conds + self.group_by_cols)
            relevant_missing = [
                a for a in self.missing_attrs
                if re.search(rf"\b{re.escape(a)}\b", query_text, re.IGNORECASE)
            ]
            factor_attrs = [a for a in self.ordering.keys() if a in relevant_missing]

            self.factors = []
            for attr in factor_attrs:
                cond_cols = self.ordering.get(attr, [])
                cond = next(
                    (c for c in self.where_conds
                    if re.search(rf"\b{attr}\b", c, re.IGNORECASE)),
                    None
                )
                if cond is None:
                    cond = "TRUE"
                self.factors.append({
                    'name': attr,
                    'condition': cond,
                    'conditioning': cond_cols
                })

    def _case_expr(self, idx: int, fact: dict) -> str:
        alias = f"CP{idx+1}"
        name = fact['name']
        cond = fact['condition']
        return (
            f"CASE WHEN {self.base_alias}.{name} IS NOT NULL AND {cond} THEN 1.0 "
            f"WHEN {self.base_alias}.{name} IS NULL THEN {alias}.p_{idx+1} "
            f"ELSE 0.0 END"
        )



    def groupLevelQueryRewriter(self) -> str:
        """
        Cell-level probability per Lubna_VLDB_2026_cursor.pdf (Sec 3.1 Cell-Based
        Model, Sec 3.2.1 Selection). Uncertainty is at the cell (tuple-attribute)
        level; for each tuple we estimate P(φ_i | X_Ai = x) via the stratum where
        X_Ai = x (ordered separating set). No product per tuple: we keep one
        probability per cell (per factor). Product is done only at group level:
        global_probability = ∏_f MIN(cell_prob_f) over factors f.
        Implementation: correlated scalar subqueries per factor; inner query
        outputs cell_prob_f per factor; outer GROUP BY computes product of MINs.
        """
        if not getattr(self, "table", None):
            return self.base_query  # join mode: single-table path not used
        alias, table = self.base_alias, (self.table[:-4] if self.table.lower().endswith('.csv') else self.table)

        phi_miss = [f for f in self.factors if f['name'] in self.missing_attrs]
        phi_obs = [c for c in self.where_conds if not any(attr in c for attr in self.missing_attrs)]

        cond_cols = {col for f in phi_miss for col in f['conditioning']}
        if not cond_cols:
            return self.base_query

        inputGrpBY = [c.strip() for c in getattr(self, 'group_by_cols', []) if c.strip()]
        can_override = inputGrpBY and set(inputGrpBY).issubset(cond_cols)
        activeGrpBY = inputGrpBY if can_override else sorted(cond_cols)

        forced = list(dict.fromkeys(getattr(self, 'force_group_cols', []) or []))
        if forced:
            already = set(activeGrpBY)
            activeGrpBY = activeGrpBY + [c for c in forced if c not in already]

        # --- Cell-level probability via pre-aggregated stats joins (dataset-agnostic fast path) ---
        # For each factor, compute stratum stats once, then join by conditioning columns:
        #   p_i = num_i / den_i where
        #   num_i = SUM(1[attr observed AND condition]) per stratum
        #   den_i = COUNT(attr observed) per stratum
        def _qualify_cond_for_subquery(cond_str: str, inner_alias: str, cols: set) -> str:
            """Prefix bare column names in cond_str with inner_alias. for use inside scalar subquery."""
            out = cond_str
            for c in sorted(cols, key=len, reverse=True):  # longer first to avoid substring matches
                out = re.sub(r'\b' + re.escape(c) + r'\b', f'{inner_alias}.{c}', out, flags=re.IGNORECASE)
            return out

        # ── Merged stats: single subquery scan for all factors ──
        _ms = "merged_stats"
        cell_prob_exprs = []   # list of (name, expr) for inner SELECT
        stats_joins: List[str] = []
        inner_cond_cols = []   # for HAVING: MAX(cond_i)=1 OR MAX(null_i)=1 per factor

        union_cond_cols = sorted({col for f in phi_miss for col in f['conditioning']})

        agg_parts = []
        for f in phi_miss:
            name = f['name']
            cond_x = _qualify_cond_for_subquery(f['condition'], 'x', set(f['conditioning']) | {name})
            agg_parts.append(
                f"AVG(CASE WHEN x.{name} IS NOT NULL THEN "
                f"(CASE WHEN ({cond_x}) THEN 1.0 ELSE 0.0 END) "
                f"ELSE NULL END) AS p_{name}"
            )
            agg_parts.append(f"NULLIF(COUNT(x.{name}), 0) AS den_{name}")
        agg_sql = ', '.join(agg_parts)

        if union_cond_cols:
            grp_cols = ', '.join(f"x.{c}" for c in union_cond_cols)
            stats_sub = (
                f"(SELECT {grp_cols}, {agg_sql} "
                f"FROM {table} x "
                f"WHERE {' AND '.join(f'x.{c} IS NOT NULL' for c in union_cond_cols)} "
                f"GROUP BY {grp_cols}) AS {_ms}"
            )
            join_cond = ' AND '.join(f"{_ms}.{c}={alias}.{c}" for c in union_cond_cols)
            stats_joins.append(f"LEFT JOIN {stats_sub} ON {join_cond}")
        else:
            stats_sub = f"(SELECT {agg_sql} FROM {table} x) AS {_ms}"
            stats_joins.append(f"CROSS JOIN {stats_sub}")

        for f in phi_miss:
            name = f['name']
            cell_prob_exprs.append((name, f"COALESCE({_ms}.p_{name}, 1.0)"))
            inner_cond_cols.append((name, f['condition']))

        # --- Interval bounds (optional) ---
        _iv_mode = getattr(self, 'interval_mode', None)
        _iv_alpha = getattr(self, 'interval_alpha', None)
        cell_bound_exprs = []  # list of (name, lower_expr, upper_expr)
        delta_relvar_exprs = []  # list of (name, relvar_expr)  -- delta method only
        if _iv_mode and _iv_alpha and phi_miss:
            if _iv_mode == "delta":
                for name, cp_expr in cell_prob_exprs:
                    den_expr = f"{_ms}.den_{name}"
                    delta_relvar_exprs.append((name, _delta_relvar_sql(cp_expr, den_expr, _iv_alpha)))
            else:
                alpha_f = 1.0 - (1.0 - _iv_alpha) ** (1.0 / len(phi_miss))
                for name, cp_expr in cell_prob_exprs:
                    den_expr = f"{_ms}.den_{name}"
                    lower, upper = _interval_bounds_sql(_iv_mode, cp_expr, den_expr, alpha_f)
                    cell_bound_exprs.append((name, lower, upper))

        conditioning_not_null_expr = ' AND '.join(f"{alias}.{col} IS NOT NULL" for col in cond_cols)
        group_clause = ', '.join(f"{alias}.{col}" for col in activeGrpBY)
        # Avoid duplicate column names in inner SELECT (would make cell_inner.col ambiguous in outer).
        select_cols_extra = [c for c in self.select_cols if c not in activeGrpBY]
        select_cols_expr_extra = ', '.join(f"{alias}.{c}" for c in select_cols_extra) if select_cols_extra else ''
        select_cols_expr = (group_clause + (', ' + select_cols_expr_extra if select_cols_expr_extra else ''))

        known_cols = set(cond_cols) | set(self.missing_attrs) | set(activeGrpBY) | set(self.select_cols)
        phi_obs_q = [_qualify_cond_for_subquery(c, alias, known_cols) for c in phi_obs]

        complete_terms = []
        for f in phi_miss:
            cond_q = _qualify_cond_for_subquery(f['condition'], alias, set(f['conditioning']) | {f['name']})
            complete_terms.append(f"{alias}.{f['name']} IS NOT NULL AND {cond_q}")
        complete_filter = ' AND '.join(complete_terms + phi_obs_q) if (complete_terms or phi_obs_q) else "TRUE"

        if phi_miss:
            miss_core = '(' + ' OR '.join(f"{alias}.{f['name']} IS NULL" for f in phi_miss) + ')'
        else:
            miss_core = "FALSE"

        miss_guard = ' AND '.join(
            f"({alias}.{f['name']} IS NULL OR {_qualify_cond_for_subquery(f['condition'], alias, set(f['conditioning']) | {f['name']})})"
            for f in phi_miss
        )
        # Qualify observed predicates token-wise (safe), avoiding invalid "base.(...)" forms.
        obs_guard = ' AND '.join(f"({c})" for c in phi_obs_q) if phi_obs else ''

        missing_filter_parts = [miss_core]
        if miss_guard:
            missing_filter_parts.append(miss_guard)
        if obs_guard:
            missing_filter_parts.append(obs_guard)
        missing_filter = ' AND '.join(missing_filter_parts)

        # Inner select: one row per tuple with cell-level probability and HAVING helper columns.
        # Qualify column refs in cond with alias to avoid ambiguity (e.g. with scalar subqueries).
        inner_extra = ', '.join(
            f"(CASE WHEN {_qualify_cond_for_subquery(cond, alias, set(cond_cols_f) | {name})} THEN 1 ELSE 0 END) AS cond_f_{name}, "
            f"(CASE WHEN {alias}.{name} IS NULL THEN 1 ELSE 0 END) AS null_f_{name}"
            for name, cond, cond_cols_f in [(f['name'], f['condition'], f['conditioning']) for f in phi_miss]
        )
        # Fast path: inner query only on rows that need estimation (have NULLs). Complete rows are added via a separate cheap query and merged.
        # Keep missing-case rows even when separator columns are NULL.
        # Otherwise candidate tuples can be dropped before estimation.
        inner_where = f"({missing_filter})"

        # Inner: one column per factor (cell-level prob), no product per tuple
        cell_prob_inner = ', '.join(
            f"{expr} AS cell_prob_{name}" for name, expr in cell_prob_exprs
        )
        bounds_inner = ''
        if cell_bound_exprs:
            bounds_inner = ',\n  ' + ', '.join(
                f"{lo} AS cell_prob_lower_{n}, {hi} AS cell_prob_upper_{n}"
                for n, lo, hi in cell_bound_exprs
            )
        elif delta_relvar_exprs:
            bounds_inner = ',\n  ' + ', '.join(
                f"{rv} AS cell_relvar_{n}" for n, rv in delta_relvar_exprs
            )
        stats_join_sql = (" ".join(stats_joins) + "\n") if stats_joins else ""
        inner_sql = (
            f"SELECT {select_cols_expr},\n"
            f"  {cell_prob_inner}{bounds_inner},\n"
            f"  ({missing_filter}) AS is_missing,\n"
            f"  ({complete_filter}) AS is_complete,\n"
            f"  {inner_extra}\n"
            f"FROM {table} {alias}\n"
            f"{stats_join_sql}"
            f"WHERE {inner_where}"
        )

        group_cols_outer = ', '.join(f"cell_inner.{c}" for c in activeGrpBY)
        select_cols_outer = ', '.join(f"cell_inner.{c}" for c in self.select_cols)
        product_of_mins = ' * '.join(
            f"MIN(cell_inner.cell_prob_{name})" for name, _ in cell_prob_exprs
        )
        bounds_outer = ''
        if cell_bound_exprs:
            prod_lower = ' * '.join(
                f"MIN(cell_inner.cell_prob_lower_{n})" for n, _, _ in cell_bound_exprs
            )
            prod_upper = ' * '.join(
                f"MIN(cell_inner.cell_prob_upper_{n})" for n, _, _ in cell_bound_exprs
            )
            bounds_outer = (
                f",\n  ({prod_lower}) AS global_probability_lower"
                f",\n  ({prod_upper}) AS global_probability_upper"
            )
        elif delta_relvar_exprs:
            z = _z_critical(_iv_alpha)
            sum_rv = ' + '.join(
                f"MAX(cell_inner.cell_relvar_{n})" for n, _ in delta_relvar_exprs
            )
            gp = product_of_mins
            bounds_outer = (
                f",\n  GREATEST(0.0, ({gp}) - {z} * ({gp}) * SQRT(GREATEST(0, {sum_rv}))) AS global_probability_lower"
                f",\n  LEAST(1.0, ({gp}) + {z} * ({gp}) * SQRT(GREATEST(0, {sum_rv}))) AS global_probability_upper"
            )
        hav_cols = ' AND '.join(
            f"(MAX(cell_inner.cond_f_{name})=1 OR MAX(cell_inner.null_f_{name})=1)"
            for name, _ in inner_cond_cols
        )
        # Optional stratified sampling (engineering speed-up): cap rows per group
        # before aggregation in part1. Can be restricted to only very large groups.
        # Disabled by default.
        inner_source_sql = inner_sql
        sample_per_group_limit = getattr(self, "sample_per_group_limit", None)
        if sample_per_group_limit is not None and int(sample_per_group_limit) > 0 and activeGrpBY:
            part_cols = ", ".join(f'"{c.replace(chr(34), chr(34)+chr(34))}"' for c in activeGrpBY)
            order_expr = getattr(self, "sample_order_expr", "random()")
            sample_large_only_threshold = getattr(self, "sample_large_only_threshold", None)
            if sample_large_only_threshold is not None and int(sample_large_only_threshold) > 0:
                inner_source_sql = (
                    f"SELECT * FROM ("
                    f"SELECT src.*, "
                    f"COUNT(*) OVER (PARTITION BY {part_cols}) AS __grp_n, "
                    f"ROW_NUMBER() OVER (PARTITION BY {part_cols} ORDER BY {order_expr}) AS __rn "
                    f"FROM ({inner_sql}) AS src"
                    f") AS sampled "
                    f"WHERE sampled.__grp_n <= {int(sample_large_only_threshold)} "
                    f"OR sampled.__rn <= {int(sample_per_group_limit)}"
                )
            else:
                inner_source_sql = (
                    f"SELECT * FROM ("
                    f"SELECT src.*, ROW_NUMBER() OVER (PARTITION BY {part_cols} ORDER BY {order_expr}) AS __rn "
                    f"FROM ({inner_sql}) AS src"
                    f") AS sampled WHERE sampled.__rn <= {int(sample_per_group_limit)}"
                )

        # Part1: estimation only on missing-case rows (small set → fast)
        part1_sql = (
            f"SELECT {group_cols_outer},\n"
            f"  ({product_of_mins}) AS global_probability{bounds_outer},\n"
            f"  ARRAY_AGG(ROW({select_cols_outer})) FILTER (WHERE cell_inner.is_missing) AS missing_tuples,\n"
            f"  ARRAY_AGG(ROW({select_cols_outer})) FILTER (WHERE cell_inner.is_complete) AS complete_tuples\n"
            f"FROM (\n{inner_source_sql}\n) AS cell_inner\n"
            f"GROUP BY {group_cols_outer}\n"
            f"HAVING {hav_cols}"
        )
        # Part2: complete-case rows only (simple filter + group, no estimation).
        # Do not require separator columns to be non-null here: complete tuples that
        # already satisfy deterministic predicates should not be dropped just because
        # an estimation separator value is null.
        part2_where = f"({complete_filter})"
        part2_group = ', '.join(f"{alias}.{c}" for c in activeGrpBY)
        part2_select = ', '.join(f"{alias}.{c}" for c in self.select_cols)
        part2_sql = (
            f"SELECT {part2_group},\n"
            f"  ARRAY_AGG(ROW({part2_select})) AS complete_tuples\n"
            f"FROM {table} {alias}\n"
            f"WHERE {part2_where}\n"
            f"GROUP BY {part2_group}"
        )
        # Merge: FULL OUTER JOIN so groups that are only in part2 (all-complete) appear; combine complete_tuples
        join_on = ' AND '.join(f"part1.{c} = part2.{c}" for c in activeGrpBY)
        grp_sel = ', '.join(f"COALESCE(part1.{c}, part2.{c}) AS {c}" for c in activeGrpBY)
        bounds_merge = ''
        if cell_bound_exprs or delta_relvar_exprs:
            bounds_merge = (
                f"  COALESCE(part1.global_probability_lower, 1.0) AS global_probability_lower,\n"
                f"  COALESCE(part1.global_probability_upper, 1.0) AS global_probability_upper,\n"
            )
        sql = (
            f"WITH part1 AS (\n{part1_sql}\n),\n"
            f"part2 AS (\n{part2_sql}\n)\n"
            f"SELECT {grp_sel},\n"
            f"  COALESCE(part1.global_probability, 1.0) AS global_probability,\n"
            f"{bounds_merge}"
            f"  COALESCE(part1.missing_tuples, ARRAY[]::record[]) AS missing_tuples,\n"
            f"  array_cat(COALESCE(part1.complete_tuples, ARRAY[]::record[]), COALESCE(part2.complete_tuples, ARRAY[]::record[])) AS complete_tuples\n"
            f"FROM part1 FULL OUTER JOIN part2 ON {join_on};"
        )
        return sql

    def JoinQueryRewriter(self) -> str:
        """
        Partitioned Join (paper §Partitioned Join): Let 𝒳 = 𝐗_T ∪ 𝐗_S.
        Execute join within blocks defined by 𝒳; conditioned on 𝒳, missing cells
        are conditionally independent. Cell-level on each side: T̄ and S̄ are each
        produced by groupLevelQueryRewriter (one prob per factor, product ∏_f MIN(cell_prob_f)
        at group level only). Then p_j := P̃_T · P̃_S (join multiplies the two group-level
        probabilities). We push σ_{K≠⊥} below the join (WHERE K IS NOT NULL on both inputs).
        Output: key_proj (𝒳/join key), global_probability = P̃_T * P̃_S, merge complete/missing
        tuples. Extensional merge (γ_{𝒳,Y; p_out := 1 - ∏(1-p_j)}) when grouping by (𝒳, Y).
        """
        if not self.join_mode or not self.join_key:
            raise ValueError("JoinQueryRewriter called without join context/keys.")
        T = self.T[:-4] if self.T.lower().endswith('.csv') else self.T
        S = self.S[:-4] if self.S.lower().endswith('.csv') else self.S
        t_select_cols = list(dict.fromkeys(self.join_key + self.select_cols))
        t_base = f"SELECT {', '.join(t_select_cols)} FROM {T}"
        # σ_{K≠⊥}: push filter below join so we do not enumerate repairs for missing join keys
        t_where = [f"{k} IS NOT NULL" for k in self.join_key]
        if self.where_T:
            t_where.extend(self.where_T)
        t_base += " WHERE " + " AND ".join(t_where)

        _iv_mode = getattr(self, 'interval_mode', None)
        _iv_alpha = getattr(self, 'interval_alpha', None)

        qrT = QueryRewriter(t_base, self.ordering_T, self.missing_T)
        qrT.force_group_cols = self.join_key[:]
        qrT.select_cols = self.select_cols[:]
        if _iv_mode:
            qrT.interval_mode = _iv_mode
            qrT.interval_alpha = _iv_alpha
        t_sql = qrT.groupLevelQueryRewriter().rstrip(';')

        s_base = f"SELECT {', '.join(self.join_key)} FROM {S}"
        s_where = [f"{k} IS NOT NULL" for k in self.join_key]
        if self.where_S:
            s_where.extend(self.where_S)
        s_base += " WHERE " + " AND ".join(s_where)

        qrS = QueryRewriter(s_base, self.ordering_S, self.missing_S)
        qrS.force_group_cols = self.join_key[:]
        if _iv_mode:
            qrS.interval_mode = _iv_mode
            qrS.interval_alpha = _iv_alpha
        s_sql = qrS.groupLevelQueryRewriter().rstrip(';')
        if "global_probability" not in s_sql and (not self.missing_S or not self.ordering_S):
            gb = ", ".join(self.join_key)
            sel_qual = ", ".join(f"inner_s.{k}" for k in self.join_key)
            bounds_wrap = ""
            if _iv_mode:
                bounds_wrap = "1.0::float AS global_probability_lower, 1.0::float AS global_probability_upper, "
            s_sql = (
                f"SELECT {gb}, 1.0::float AS global_probability, {bounds_wrap}"
                f"ARRAY[]::record[] AS missing_tuples, "
                f"ARRAY_AGG(ROW({sel_qual})) AS complete_tuples "
                f"FROM ({s_sql}) AS inner_s GROUP BY {gb}"
            )

        on_clause = ' AND '.join([f"subT.{k} = subS.{k}" for k in self.join_key])

        outer_group_by = ''
        if getattr(self, 'group_by_cols', None):
            gb_cols = ', '.join([f"subT.{c}" for c in self.group_by_cols])
            outer_group_by = f"\nGROUP BY {gb_cols}"

        key_proj = ', '.join([f"subT.{k} AS {k}" for k in self.join_key])

        bounds_join = ""
        if _iv_mode:
            bounds_join = (
                f"  (subT.global_probability_lower * subS.global_probability_lower) AS global_probability_lower,\n"
                f"  (subT.global_probability_upper * subS.global_probability_upper) AS global_probability_upper,\n"
            )

        return (
            f"SELECT {key_proj},\n"
            f"  (subT.global_probability * subS.global_probability) AS global_probability,\n"
            f"{bounds_join}"
            f"  CASE WHEN cardinality(subT.missing_tuples)>0 OR cardinality(subS.missing_tuples)>0\n"
            f"       THEN (subT.complete_tuples || subT.missing_tuples)\n"
            f"       ELSE subT.missing_tuples[0:0] END AS missing_tuples,\n"
            f"  CASE WHEN cardinality(subT.complete_tuples)>0 AND cardinality(subS.complete_tuples)>0\n"
            f"       THEN subT.complete_tuples ELSE subT.complete_tuples[0:0] END AS complete_tuples\n"
            f"FROM (\n{t_sql}\n) AS subT\n"
            f"JOIN (\n{s_sql}\n) AS subS\n"
            f"ON {on_clause}"
            f"{outer_group_by};"
        )








# # ##-------------------- with executer --------------------------###
# if __name__=='__main__':
#     conn = psycopg2.connect(
#         host="localhost", port=5433,
#         dbname="mydb",   user="alzamill", password=os.environ.get("PGPASSWORD", "")
#     )

#     csv_queries = {
#       'jobs':    {'csv': 'set_mnar_jobs.csv',    'table': 'set_mnar_jobs'},
#       'ratings': {'csv': 'set_mnar_ratings.csv', 'table': 'set_mnar_ratings'}
#     }

#     executor = QueryExecutor(conn, csv_queries)

#     base_q = 'SELECT ID FROM set_mnar_jobs.csv WHERE income > 100000 AND bonus > 5000'
#     ordering_T     = {'income': ['job'], 'bonus': ['job']}
#     missing_attrs_T= ['income', 'bonus']

#     results = executor.run(
#         base_q, ordering_T=ordering_T, missing_T=missing_attrs_T)
#     for grp, prob, certain, possible in results:
#         print(f"group={grp}, prob={prob:.2f}, certain={certain}, possible={possible}")

#     base_q = 'SELECT ID FROM set_mnar_jobs.csv JOIN set_mnar_ratings.csv USING (job) WHERE income > 100000 AND bonus > 5000'
#     ordering_S     = {'rating': ['job']}
#     missing_attrs_S= ['rating']
#     results = executor.run(
#         base_q, ordering_T=ordering_T, missing_T=missing_attrs_T, ordering_S=ordering_S, missing_S=missing_attrs_S,join_key='job')
#     for grp, prob, certain, possible in results:
#         print(f"group={grp}, prob={prob:.2f}, certain={certain}, possible={possible}")




#     conn.close()
