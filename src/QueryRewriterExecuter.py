
import psycopg2
import pandas as pd
from io import StringIO
import csv, re, os, time
import myDataAnalyzer
from collections import OrderedDict

class QueryRewriter:
    def __init__(self):
        pass
    def mcarsingle(self, attr, tbl, where):
            return f"""
                WITH stats AS (
                SELECT
                    AVG("{attr}"::NUMERIC)      AS est,
                    VAR_SAMP("{attr}"::NUMERIC) AS v_obs,
                    COUNT("{attr}")             AS n_obs
                FROM {tbl} {where}
                )
                SELECT est, SQRT(v_obs/n_obs) AS stderr FROM stats;
            """

    def marsingle(self, selC, attr, tbl, where, grpBy):
        return f"""
            WITH grp AS (
              SELECT {selC},
                     COUNT(*)       AS n_all,
                     COUNT("{attr}")AS n_obs,
                     AVG("{attr}"::NUMERIC)     AS avg_obs,
                     VAR_SAMP("{attr}"::NUMERIC) AS var_obs
              FROM {tbl} {where}
              GROUP BY {grpBy}
            )
            SELECT
              SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
              SQRT(
                SUM(
                  (var_obs/n_obs)
                  * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                )
              ) AS stderr
            FROM grp;
        """

    def mcarjoin(self, attr, tblL, tblR, key, where):
        return f"""
            SET enable_hashjoin=ON;
            WITH j AS (
              SELECT l."{attr}"::NUMERIC AS bal
              FROM {tblL} AS l JOIN {tblR} AS r USING("{key}")
              {where}
            ), stats AS (
              SELECT
                AVG(bal)    AS est,
                VAR_SAMP(bal) AS v_obs,
                COUNT(bal)  AS n_obs
              FROM j
            )
            SELECT est, SQRT(v_obs/n_obs) AS stderr FROM stats;
        """

    def marjoin(self, select_clause, tblL, tblR, where, key, grpBy):
        return f"""
            SET enable_hashjoin=ON;
            WITH j AS (
              SELECT {select_clause}
              FROM {tblL} AS l JOIN {tblR} AS r USING("{key}")
              {where}
            ), grp AS (
              SELECT {grpBy + ',' if grpBy else ''}
                     COUNT(*)      AS n_all,
                     COUNT(bal)    AS n_obs,
                     AVG(bal)      AS avg_obs,
                     VAR_SAMP(bal) AS var_obs
              FROM j
              {('GROUP BY ' + grpBy) if grpBy else ''}
            )
            SELECT
              SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
              SQRT(
                SUM(
                  (var_obs/n_obs)
                  * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                )
              ) AS stderr
            FROM grp;
        """
    def mnarsingle(self, strata_select, attr, tbl, where, strata_groupby):
        """
        AVG(attr) under MNAR, stratified by 'G' (the identifiability keys).
        'strata_select' is a SELECT-list of the G columns (with aliases if needed).
        'strata_groupby' is the comma-joined GROUP BY list for those columns.
        """
        if not strata_groupby:
            raise ValueError("MNAR requires a non-empty strata (G) satisfying Y ⟂ R_Y | G.")
        return f"""
            WITH grp AS (
              SELECT {strata_select},
                     COUNT(*)                         AS n_all,
                     COUNT("{attr}")                 AS n_obs,
                     AVG("{attr}"::NUMERIC)          AS avg_obs,
                     VAR_SAMP("{attr}"::NUMERIC)     AS var_obs
              FROM {tbl} {where}
              GROUP BY {strata_groupby}
            )
            SELECT
              SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
              SQRT(
                SUM(
                  (var_obs/NULLIF(n_obs,0))
                  * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                )
              ) AS stderr
            FROM grp;
        """

    # mnar join
    def mnarjoin(self, select_clause, tblL, tblR, where, key, strata_groupby):
        """
        AVG(l.attr) under MNAR on a join. 'select_clause' must include:
          - l."<attr>" AS bal
          - the MNAR strata keys 'G' from left/right as distinct columns
        'strata_groupby' is the comma-joined GROUP BY over those keys.
        """
        if not strata_groupby:
            raise ValueError("MNAR JOIN requires a non-empty strata (G) across the join.")
        return f"""
            SET enable_hashjoin=ON;
            WITH j AS (
              SELECT {select_clause}
              FROM {tblL} AS l JOIN {tblR} AS r USING({key})
              {where}
            ), grp AS (
              SELECT {strata_groupby},
                     COUNT(*)      AS n_all,
                     COUNT(bal)    AS n_obs,
                     AVG(bal)      AS avg_obs,
                     VAR_SAMP(bal) AS var_obs
              FROM j
              GROUP BY {strata_groupby}
            )
            SELECT
              SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
              SQRT(
                SUM(
                  (var_obs/NULLIF(n_obs,0))
                  * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                )
              ) AS stderr
            FROM grp;
        """

    # query with group by in input over single relation
    def mnarSingleGroupBy(self, inner_sel, tbl, where, inner_grp, outer_group_cols):
        """
        Input query already has GROUP BY (outer groups).
        We MNAR-correct inside each outer group using the MNAR strata G.
        """
        if not inner_grp:
            raise ValueError("MNAR SingleGroupBy requires inner MNAR strata (G).")
        return f"""
            WITH filtered AS (
              SELECT {inner_sel}
              FROM {tbl} {where}
            ), grp AS (
              SELECT {inner_grp},
                     COUNT(*)      AS n_all,
                     COUNT(bal)    AS n_obs,
                     AVG(bal)      AS avg_obs,
                     VAR_SAMP(bal) AS var_obs
              FROM filtered
              GROUP BY {inner_grp}
            )
            SELECT
              {", ".join(f'"{c}"' for c in outer_group_cols)},
              SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
              SQRT(
                SUM(
                  (var_obs/NULLIF(n_obs,0))
                  * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                )
              ) AS stderr
            FROM grp
            GROUP BY {", ".join(f'"{c}"' for c in outer_group_cols)};
        """

    # input query has join with group by
    def mnarJoinGroupBy(self, select_clause, tblL, tblR, where, key, strata_grp_fields, gb_select):
        """
        Input query has GROUP BY cols (outer). We MNAR-correct within each outer group.
        'strata_grp_fields' = G plus the OUTER group-by fields (all in one GROUP BY).
        """
        if not strata_grp_fields:
            raise ValueError("MNAR JoinGroupBy requires MNAR strata (G) + outer GROUP BY fields.")
        return f"""
            SET enable_hashjoin=ON;
            WITH joined AS (
              SELECT {select_clause}, {gb_select}
              FROM {tblL} AS l
              JOIN {tblR} AS r USING("{key}")
              {where}
            ), grp AS (
              SELECT {strata_grp_fields},
                     COUNT(*)      AS n_all,
                     COUNT(bal)    AS n_obs,
                     AVG(bal)      AS avg_obs,
                     VAR_SAMP(bal) AS var_obs
              FROM joined
              GROUP BY {strata_grp_fields}
            )
            SELECT {gb_select},
                   SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
                   SQRT(
                     SUM(
                       (var_obs/NULLIF(n_obs,0))
                       * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                     )
                   ) AS stderr
            FROM grp
            GROUP BY {gb_select};
        """

    ############### rewrite queries that already have group by in the input query ###############
    def mcarsingleGroupBy(self,sel, attr,tbl,where):
         return f"""
                WITH s AS (
                SELECT {sel}, "{attr}"::NUMERIC AS bal
                FROM {tbl} {where}
                )
                SELECT
                {sel},
                AVG(bal)                      AS est,
                SQRT(VAR_SAMP(bal)/COUNT(bal)) AS stderr
                FROM s
                GROUP BY {sel};
            """
    def marSingleGroupBy(self, inner_sel,tbl,where,inner_grp,group_cols):
         return f"""
                WITH filtered AS (
                SELECT {inner_sel}
                FROM {tbl} {where}
                ), grp AS (
                SELECT {inner_grp},
                        COUNT(*)      AS n_all,
                        COUNT(bal)    AS n_obs,
                        AVG(bal)      AS avg_obs,
                        VAR_SAMP(bal) AS var_obs
                FROM filtered
                GROUP BY {inner_grp}
                )
                SELECT
                {", ".join(f'"{c}"' for c in group_cols)},
                SUM(avg_obs*n_all)
                    / NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
                SQRT(
                    SUM(
                    (var_obs/n_obs)
                    * (n_all::NUMERIC
                        /NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                    )
                ) AS stderr
                FROM grp
                GROUP BY {", ".join(f'"{c}"' for c in group_cols)};
            """
    def mcarJoinGroupBy(self,sel,tblL,tblR,where,key,group_cols):
         return f"""
                SET enable_hashjoin=ON;
                WITH j AS (
                SELECT {sel}
                    FROM {tblL} AS l
                    JOIN {tblR} AS r USING("{key}")
                    {where}
                )
                SELECT
                {", ".join(f'"{c}"' for c in group_cols)},
                AVG(bal)                      AS est,
                SQRT(VAR_SAMP(bal)/COUNT(bal)) AS stderr
                FROM j
                GROUP BY {", ".join(f'"{c}"' for c in group_cols)};
            """


class QueryExecutor:
    TYPE_MAP = {'int64':'BIGINT','float64':'NUMERIC'}

    def __init__(self, conn, csv_queries):
        self.conn        = conn
        self.cur         = conn.cursor()
        self.csv_queries = csv_queries # my dict of queries and the cuases // see main section

        # maps each MAR CSV path to its SQL table name
        self.mar_table_for = {}
        for meta in csv_queries.values():
            paths  = meta['csv'] if isinstance(meta['csv'], list) else [meta['csv']]
            tables = meta['table'] if isinstance(meta['table'], list) else [meta['table']]
            for p,t in zip(paths, tables):
                self.mar_table_for[p] = t

        # in case of injected datasets, map each complete CSV to its SQL table name the ground‐truth)
        self.full_table_for = {}
        self.groundTR = any(meta.get('complete_csv') for meta in csv_queries.values())
        for meta in csv_queries.values():
            if meta.get('complete_csv'):
                paths  = meta['complete_csv']    if isinstance(meta['complete_csv'], list)    else [meta['complete_csv']]
                tables = meta['complete_table']  if isinstance(meta['complete_table'], list)  else [meta['complete_table']]
                for p,t in zip(paths, tables):
                    self.full_table_for[p] = t

        self.prepareTables()


    def inferSchema(self, path):
            df = pd.read_csv(path, nrows=1000, low_memory=False)
            df.columns = df.columns.str.lower()
            df = df.loc[:, ~df.columns.str.startswith('unnamed:')]
            return {col:self.TYPE_MAP.get(str(dt),'TEXT') for col,dt in df.dtypes.items()}

    def loadCSV(self, path, table_name, cols):
        df = pd.read_csv(path, keep_default_na=True, na_values=['',' ','\\N'])
        df.columns = df.columns.str.lower() # convert to lowercase columsn names I ahd errors with sql if I dont convert
        df = df.loc[:, ~df.columns.str.startswith('unnamed:')] ## some of my datastest had such cols so I am dropping them
        df = df[cols]
        buf = StringIO()
        df.to_csv(buf, index=False, header=False, na_rep='\\N', quoting=csv.QUOTE_MINIMAL)
        buf.seek(0)
        col_list = ', '.join(f'"{c}"' for c in cols)
        sql = f"COPY {table_name}({col_list}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
        self.cur.copy_expert(sql, buf)
        self.conn.commit()

    def prepareTables(self):
        # Create - load MCAR/MAR tables
        for path,tbl in self.mar_table_for.items():
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

        # Create - load complete tables in case of injected data
        if self.groundTR:
            for path,tbl in self.full_table_for.items():
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


    def _normalize_mar_causes(self,meta_val, n_lists): ## for manr keys
        """
        Accepts:
        - None / {} : -> [ {} ] * n_lists
        - dict      : -> [ dict ] * n_lists
        - list[dict] length==n_lists : -> as-is
        """
        if not meta_val:
            return [dict() for _ in range(n_lists)]
        if isinstance(meta_val, dict):
            return [meta_val] * n_lists
        if isinstance(meta_val, list) and len(meta_val) == n_lists and all(isinstance(d, dict) for d in meta_val):
            return meta_val
        raise ValueError("mar_causes must be a dict or a list[dict] matching MNAR_Strata length")

    def _expand_with_causes(self,keys, cause_map):
        """
        Replace any key present in cause_map by its causes, recursively.
        Keep order stable; deduplicate; stop on cycles.
        """
        out, seen = [], set()

        def expand_one(k, stack):
            if k in stack:   # cycle guard
                return
            if k in cause_map and cause_map[k]:
                stack.add(k)
                for c in cause_map[k]:
                    expand_one(c, stack)
                stack.remove(k)
            else:
                if k not in seen:
                    seen.add(k)
                    out.append(k)

        for k in keys:
            expand_one(k, set())
        return out  # stable unique list

    def _qualify_group_cols(self,group_cols, schL, schR):
        sel_cols = []
        for c in group_cols:
            if c in schL:
                sel_cols.append(f'l."{c}"')
            elif c in schR:
                sel_cols.append(f'r."{c}"')
            else:
                sel_cols.append(f'"{c}"')  # fallback
        return ", ".join(sel_cols), sel_cols

    def _basename_noext(self,p: str) -> str:
      b = os.path.basename(p)
      b, _ = os.path.splitext(b)
      return b.lower()

    def _resolve_csv_token(self,token: str, csv_list: list[str]) -> str:
        base = self._basename_noext(token)
        for p in csv_list:
            if self._basename_noext(p) == base:
                return p
        raise ValueError(f"Query refers to {token!r}, but it's not in this dataset's csv list.")


    def compute2(self):
        z = 1.96
        results = {}
        rewrite = QueryRewriter()
        QT =-1
        JQT =-1
        for ds_name, meta in self.csv_queries.items():
            has_truth  = bool(meta.get('complete_csv'))
            csv_list   = meta['csv'] if isinstance(meta['csv'], list) else [meta['csv']]
            raw_causes = meta.get('Cause', [])
            full_list  = meta.get('complete_csv', [])

            # normalize causes  list-of-lists of lowercase names
            if not raw_causes:
                cause_list = [[] for _ in csv_list]
            elif all(isinstance(c, str) for c in raw_causes):
                cause_list = [[c.lower()] for c in raw_causes]
            else:
                cause_list = [[c.lower() for c in sub] for sub in raw_causes]


            ## mnar
            raw_mnar = meta.get('MNAR_Strata', [])
            if not raw_mnar:
                mnar_list = [[] for _ in csv_list]
            elif all(isinstance(c, str) for c in raw_mnar):
                mnar_list = [[c.lower()] for c in raw_mnar]
            else:
                mnar_list = [[c.lower() for c in sub] for sub in raw_mnar]
            mar_causes_meta = self._normalize_mar_causes(meta.get("mar_causes"), len(mnar_list))



            results[ds_name] = {}

            for qry in meta['queries']:
                QT =-1
                JQT =-1
                my_join_timer_flag=False
                # ─── 1) pull off GROUP BY and WHERE ───────────────────────
                # gb = re.search(r'GROUP\s+BY\s+(.+?)(?:;|$)', qry, re.IGNORECASE)
                # group_cols = [c.strip().strip('"').lower() for c in gb.group(1).split(',')] if gb else []

                # wm = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|$)', qry, re.IGNORECASE)
                # cond = wm.group(1).strip() if wm else None
                # where = f"WHERE {cond}" if cond else ""

                # jm = re.search(
                #     r'FROM\s+(\S+\.csv)\s+JOIN\s+(\S+\.csv)\s+USING\s*\((\w+)\)'
                #     r'(?:\s+WHERE\s+(.+))?', qry, re.IGNORECASE
                # )

                gb = re.search(r'\bGROUP\s+BY\s+(.+?)(?:;|$)', qry, re.IGNORECASE)
                group_cols = [c.strip().strip('"').lower() for c in gb.group(1).split(',')] if gb else []

                wm = re.search(r'\bWHERE\s+(.+?)(?:\s+GROUP\s+BY|;|$)', qry, re.IGNORECASE)
                cond  = wm.group(1).strip() if wm else None
                where = f"WHERE {cond}" if cond else ""

                # 2) Detect JOIN (csv token may or may not end with .csv)
                jm = re.search(
                    r'\bFROM\s+([^\s;]+?)\s+JOIN\s+([^\s;]+?)\s+USING\s*\(\s*([A-Za-z_]\w*)\s*\)',
                    qry, re.IGNORECASE
                )

                rows = None
                if jm:
                    # — JOIN case —
                    my_join_timer_flag = True
                    # attr, left_csv, right_csv, key = (
                    #     re.search(r'AVG\((\w+)\)', qry, re.IGNORECASE).group(1).lower(),
                    #     jm.group(1), jm.group(2), jm.group(3)
                    # )
                    # key = key.strip('"').strip().lower()
                    attr = re.search(r'AVG\s*\(\s*([A-Za-z_]\w*)\s*\)', qry, re.IGNORECASE).group(1).lower()
                    left_tok, right_tok, key = jm.group(1), jm.group(2), jm.group(3)
                    key = key.strip('"').strip().lower()

                    # map tokens>> actual configured paths by basename (case-insensitive, no ext)
                    left_csv  = self._resolve_csv_token(left_tok, csv_list)
                    right_csv = self._resolve_csv_token(right_tok, csv_list)

                    # map CSVs>> paths>> indices>> tables
                    l_base, r_base = os.path.basename(left_csv), os.path.basename(right_csv)
                    l_path = next(p for p in csv_list if os.path.basename(p) == l_base)
                    r_path = next(p for p in csv_list if os.path.basename(p) == r_base)
                    iL, iR = csv_list.index(l_path), csv_list.index(r_path)
                    tblL, tblR = self.mar_table_for[l_path], self.mar_table_for[r_path]

                    # ensure Y is on left
                    if attr not in self.inferSchema(l_path):
                        l_path, r_path, tblL, tblR = r_path, l_path, tblR, tblL
                        iL, iR = iR, iL

                    # schemas AFTER swap
                    schL = set(self.inferSchema(l_path))
                    schR = set(self.inferSchema(r_path))

                    # causes AFTER swap
                    causes_L = [c for c in cause_list[iL] if c in schL]
                    causes_R = [c for c in cause_list[iR] if c in schR]

                    # MNAR strata AFTER swap (expand by mar_causes)
                    mnar_L_eff = self._expand_with_causes(mnar_list[iL], mar_causes_meta[iL])
                    mnar_R_eff = self._expand_with_causes(mnar_list[iR], mar_causes_meta[iR])

                    # ground-truth tables AFTER swap
                    if has_truth:
                        if len(full_list) == 1:
                            ftL = ftR = self.full_table_for[full_list[0]]
                        else:
                            ftL = self.full_table_for[full_list[iL]]
                            ftR = self.full_table_for[full_list[iR]]



                    # if mnar_L or mnar_R:
                    #     # MNAR JOIN
                    #     select_fields = [f'l."{attr}"::NUMERIC AS bal'] \
                    #                     + [f'l."{c}" AS "{c}_L"' for c in mnar_L] \
                    #                     + [f'r."{c}" AS "{c}_R"' for c in mnar_R]
                    #     select_clause = ", ".join(select_fields)
                    #     grpBy = ", ".join([f'"{c}_L"' for c in mnar_L] + [f'"{c}_R"' for c in mnar_R])
                    #     global_sql = rewrite.mnarjoin(select_clause, tblL, tblR, where, key, grpBy)


                    if mnar_L_eff or mnar_R_eff:
                        select_fields = [f'l."{attr}"::NUMERIC AS bal'] \
                                        + [f'l."{c}" AS "{c}_L"' for c in mnar_L_eff] \
                                        + [f'r."{c}" AS "{c}_R"' for c in mnar_R_eff]
                        select_clause = ", ".join(select_fields)
                        grpBy = ", ".join([f'"{c}_L"' for c in mnar_L_eff] + [f'"{c}_R"' for c in mnar_R_eff])
                        global_sql = rewrite.mnarjoin(select_clause, tblL, tblR, where, key, grpBy)
                    if not causes_L and not causes_R:
                        # MCAR‐JOIN
                        global_sql = rewrite.mcarjoin(attr, tblL, tblR, key, where)
                    else:
                        # MAR‐JOIN
                        select_fields = [f'l."{attr}" AS bal'] + \
                                        [f'l."{c}" AS "{c}_L"' for c in causes_L] + \
                                        [f'r."{c}" AS "{c}_R"' for c in causes_R]
                        select_clause = ", ".join(select_fields)
                        left_grp  = [f'"{c}_L"' for c in causes_L]
                        right_grp = [f'"{c}_R"' for c in causes_R]
                        grpBy     = ", ".join(left_grp + right_grp)
                        # grpBy         = ", ".join(f'"{c}_L"' for c in causes_L + causes_R)## old --
                        # group_by_fields = ( ## if have inputgroupby ?
                        #           [f'"{c}_L"' for c in causes_L] +
                        #           [f'"{c}_R"' for c in causes_R] +
                        #           [f'"{c}"'     for c in group_cols]
                        #       )
                        # grpBy = ", ".join(group_by_fields)

                        global_sql    = rewrite.marjoin(select_clause, tblL, tblR, where, key, grpBy)

                    # if GROUP BY vendor_id (or any group_cols), we’ll build group_sql below…
                    group_sql = None
                    if group_cols:
                        # gb_select = ", ".join(f'l."{c}"' for c in group_cols) ###
                        # if mnar_L or mnar_R:
                        #     select_fields = [f'l."{attr}"::NUMERIC AS bal', gb_select] \
                        #                     + [f'l."{c}" AS "{c}_L"' for c in mnar_L] \
                        #                     + [f'r."{c}" AS "{c}_R"' for c in mnar_R]
                        #     select_clause = ", ".join(select_fields)
                        #     strata_grp_fields = ", ".join(
                        #         [f'"{c}_L"' for c in mnar_L] + [f'"{c}_R"' for c in mnar_R] + [gb_select]
                        #     )
                        #     group_sql = rewrite.mnarJoinGroupBy(
                        #         select_clause, tblL, tblR, where, key, strata_grp_fields, gb_select
                        #     )
                        # Use left alias for cols in left schema, right alias for cols in right schema
                        sel_cols = []
                        for c in group_cols:
                            if c in schL:
                                sel_cols.append(f'l."{c}"')
                            elif c in schR:
                                sel_cols.append(f'r."{c}"')
                            else:
                                sel_cols.append(f'"{c}"')
                        gb_select = ", ".join(sel_cols)

                        if mnar_L_eff or mnar_R_eff:
                            select_fields = [f'l."{attr}"::NUMERIC AS bal', gb_select] \
                                            + [f'l."{c}" AS "{c}_L"' for c in mnar_L_eff] \
                                            + [f'r."{c}" AS "{c}_R"' for c in mnar_R_eff]
                            select_clause = ", ".join(select_fields)
                            strata_grp_fields = ", ".join(
                                [f'"{c}_L"' for c in mnar_L_eff] + [f'"{c}_R"' for c in mnar_R_eff] + [gb_select]
                            )
                            group_sql = rewrite.mnarJoinGroupBy(
                                select_clause, tblL, tblR, where, key, strata_grp_fields, gb_select
                            )
                        if not causes_L and not causes_R:
                            # MCAR + GROUP BY
                            group_sql = f"""
                              SET enable_hashjoin=ON;
                              WITH j AS (
                                SELECT {gb_select}, l."{attr}"::NUMERIC AS bal
                                FROM {tblL} AS l
                                JOIN {tblR} AS r USING("{key}")
                                {where}
                              )
                              SELECT {gb_select},
                                    AVG(bal)                      AS est,
                                    SQRT(VAR_SAMP(bal)/COUNT(bal)) AS stderr
                              FROM j
                              GROUP BY {gb_select};
                            """
                        else:
                            # MAR + GROUP BY
                            causes_grp = [f'"{c}_L"' for c in causes_L] + [f'"{c}_R"' for c in causes_R]
                            gb_fields  = ", ".join(causes_grp + [f'"{c}"' for c in group_cols])
                            group_sql = f"""
                              SET enable_hashjoin=ON;
                              WITH joined AS (
                                SELECT {select_clause}, {gb_select}
                                FROM {tblL} AS l
                                JOIN {tblR} AS r USING("{key}")
                                {where}
                              ), grp AS (
                                SELECT {gb_fields},
                                      COUNT(*)      AS n_all,
                                      COUNT(bal)    AS n_obs,
                                      AVG(bal)      AS avg_obs,
                                      VAR_SAMP(bal) AS var_obs
                                FROM joined
                                GROUP BY {gb_fields}
                              )
                              SELECT {gb_select},
                                    SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
                                    SQRT(
                                      SUM(
                                        (var_obs/n_obs)
                                        * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                                      )
                                    ) AS stderr
                              FROM grp
                              GROUP BY {gb_select};
                            """
                else:
                    # — SINGLE‐TABLE case —
                    # m = re.match(r'SELECT\s+AVG\((\w+)\)\s+FROM\s+(\S+\.csv)', qry, re.IGNORECASE)
                    # attr, csvf = m.group(1).lower(), m.group(2)
                    m = re.search(
                        r'\bSELECT\s+AVG\s*\(\s*([A-Za-z_]\w*)\s*\)\s+FROM\s+([^\s;]+?)(?=\s+(WHERE|GROUP\s+BY)\b|;|$)',
                        qry, re.IGNORECASE
                    )
                    if not m:
                        raise ValueError(f"Cannot parse single-table AVG ... FROM ... in query:\n{qry}")

                    attr  = m.group(1).lower()
                    token = m.group(2)
                    csvf  = self._resolve_csv_token(token, csv_list)

                    base = os.path.basename(csvf)
                    mar_path = next(p for p in csv_list if os.path.basename(p)==base)
                    idx      = csv_list.index(mar_path)
                    tbl      = self.mar_table_for[mar_path]
                    causes   = cause_list[idx]
                    mnar_keys = mnar_list[idx]

                    if has_truth:
                          full_path = full_list[idx] if idx < len(full_list) else full_list[0]
                          ft = self.full_table_for[ full_path ]

                    mnar_keys_orig = mnar_list[idx]
                    mnar_keys_eff  = self._expand_with_causes(mnar_keys_orig, mar_causes_meta[idx])

                    if mnar_keys_eff:
                        strata_select  = ", ".join(f'"{c}" AS "{c}"' for c in mnar_keys_eff)
                        strata_groupby = ", ".join(f'"{c}"' for c in mnar_keys_eff)
                        global_sql = rewrite.mnarsingle(strata_select, attr, tbl, where, strata_groupby)
                    if not causes:
                        global_sql = rewrite.mcarsingle(attr, tbl, where)
                    else:
                        selC    = ", ".join(f'"{c}" AS "{c}"' for c in causes)
                        grpBy   = ", ".join(f'"{c}"' for c in causes)
                        #new removes dups in the selection if group by in input query already ahs the cause
                        # lhs_of_selC = { expr.split(" AS ")[0] for expr in selC.split(", ") }

                        # grpBy = ", ".join(
                        #     part.split(" AS ")[0]            # strip alias if present
                        #     for part in grpBy.split(", ")
                        #     if part.split(" AS ")[0] not in lhs_of_selC
                        # )
                        global_sql = rewrite.marsingle(selC, attr, tbl, where, grpBy)
                        # print("MAR single query without group by is: ", global_sql)

                    group_sql = None
                    if group_cols:
                        sel = ", ".join(f'"{c}"' for c in group_cols)
                        if mnar_keys_eff:
                            inner_sel = ", ".join(
                                [f'"{c}"' for c in group_cols] +
                                [f'"{attr}"::NUMERIC AS bal'] +
                                [f'"{c}" AS "{c}"' for c in mnar_keys_eff]
                            )
                            inner_grp = ", ".join(f'"{c}"' for c in (group_cols + mnar_keys_eff))
                            group_sql = rewrite.mnarSingleGroupBy(inner_sel, tbl, where, inner_grp, group_cols)
                        if not causes:
                            # MCAR + GROUP BY
                            group_sql = f"""
                              WITH s AS (
                                SELECT {sel}, "{attr}"::NUMERIC AS bal
                                FROM {tbl} {where}
                              )
                              SELECT {sel},
                                    AVG(bal)                      AS est,
                                    SQRT(VAR_SAMP(bal)/COUNT(bal)) AS stderr
                              FROM s
                              GROUP BY {sel};
                            """
                        else:
                            # MAR + GROUP BY
                            inner_sel = ", ".join(
                              [f'"{c}"' for c in group_cols] +
                              [f'"{attr}"::NUMERIC AS bal'] +
                              [f'"{c}" AS "{c}"' for c in causes]
                            )
                            inner_grp = ", ".join(f'"{c}"' for c in (group_cols + causes))
                            group_sql = f"""
                              WITH filtered AS (
                                SELECT {inner_sel}
                                FROM {tbl} {where}
                              ), grp AS (
                                SELECT {inner_grp},
                                      COUNT(*)      AS n_all,
                                      COUNT(bal)    AS n_obs,
                                      AVG(bal)      AS avg_obs,
                                      VAR_SAMP(bal) AS var_obs
                                FROM filtered
                                GROUP BY {inner_grp}
                              )
                              SELECT {", ".join(f'"{c}"' for c in group_cols)},
                                    SUM(avg_obs*n_all)/NULLIF((SELECT SUM(n_all) FROM grp),0) AS est,
                                    SQRT(
                                      SUM(
                                        (var_obs/n_obs)
                                        * (n_all::NUMERIC/NULLIF((SELECT SUM(n_all) FROM grp),0))^2
                                      )
                                    ) AS stderr
                              FROM grp
                              GROUP BY {", ".join(f'"{c}"' for c in group_cols)};
                            """

                # ── run q hat
                start = time.time()
                self.cur.execute(global_sql)
                mytimer = time.time() - start
                # print("executed global MAR single query without group by: ", global_sql)
                # print(" the contents of self.cur.fetchone() are:  ", self.cur.fetchone()) ## bug shoudlnt call it here, it will flush out the contents
                if my_join_timer_flag:
                    JQT=mytimer
                else:
                    QT=mytimer
                # est, stderr = map(float, self.cur.fetchone())
                row = self.cur.fetchone() or (None, None)
                est_raw, se_raw = row
                est    = float(est_raw) if est_raw is not None else 0.0
                stderr = float(se_raw)  if se_raw  is not None else 0.0
                lo, hi = est - z*stderr, est + z*stderr



                # 1) Ground‐truth lookup
                if has_truth:
                    if not group_cols:
                        # single/global ground truth
                        if jm:
                            # join ground truth
                            gt_sql = f"""
                              SELECT AVG(l."{attr}"::NUMERIC)
                              FROM {ftL} AS l
                              JOIN {ftR} AS r USING("{key}")
                              {where};
                            """
                        else:
                            # single‐table ground truth
                            gt_sql = f"""
                              SELECT AVG("{attr}"::NUMERIC)
                              FROM {ft}
                              {where};
                            """
                        self.cur.execute(gt_sql)
                        # gt = float(self.cur.fetchone()[0])
                        row_gt = self.cur.fetchone()[0] or None
                        gt = float(row_gt) if row_gt is not None else 0.0
                    else:
                        # per‐group ground truth
                        if jm:
                            gt_sel_parts = []
                            for i, c in enumerate(group_cols, start=1):
                                if c in schL:
                                    gt_sel_parts.append(f'l."{c}" AS g{i}')
                                elif c in schR:
                                    gt_sel_parts.append(f'r."{c}" AS g{i}')
                                else:
                                    gt_sel_parts.append(f'"{c}" AS g{i}')  # fallback
                            gt_sel_str = ", ".join(gt_sel_parts)

                            # GROUP BY positions 1..k
                            grp_positions = ", ".join(str(i) for i in range(1, len(gt_sel_parts)+1))

                            gt_grp_sql = f"""
                              WITH base AS (
                                SELECT {gt_sel_str},
                                      l."{attr}"::NUMERIC AS y
                                FROM {ftL} AS l
                                JOIN {ftR} AS r USING({key})
                                {where}
                              )
                              SELECT {", ".join(f"g{i}" for i in range(1, len(gt_sel_parts)+1))},
                                    AVG(y) AS gt
                              FROM base
                              GROUP BY {grp_positions};
                            """
                        else:
                            gt_sel_str = ", ".join(f'"{c}"' for c in group_cols)
                            grp_positions = ", ".join(str(i+1) for i in range(len(group_cols)))
                            gt_grp_sql = f"""
                              SELECT {gt_sel_str},
                                    AVG("{attr}"::NUMERIC) AS gt
                              FROM {ft}
                              {where}
                              GROUP BY {grp_positions};
                            """
                        start_time = time.time()
                        self.cur.execute(gt_grp_sql)
                       # QT= time.time() - start_time

                        gt_rows = { tuple(r[:-1]): r[-1] for r in self.cur.fetchall() }


                #  No GROUP BY
                if not group_cols:
                    entry = {
                        'estimate':         est,
                        'stderr':           stderr,
                        'QT':               QT,
                        'JQT':              JQT,
                        'CI95':             (lo, hi),
                        'normalized_width': (hi - lo)/est if est else None,
                    }
                    if has_truth:
                        entry['ground_truth'] = gt
                        if gt !=0:
                          entry['accuracy']     = (1 - abs(est - gt)/gt)*100
                        else:
                            entry['accuracy']     = -1
                    results[ds_name][qry] = entry

                #  Otherwise, run the per‐group query
                else:
                    start = time.time()
                    self.cur.execute(group_sql)
                    mytimer = time.time() - start
                    if my_join_timer_flag: JQT=mytimer
                    else: QT=mytimer

                    rows = self.cur.fetchall()

                    # assemble per‐group metrics
                    group_metrics = []
                    for rec in rows:
                        # *key, rec_est, rec_stderr = rec
                        *gvals, rec_est, rec_stderr = rec
                        gkey = tuple(gvals)
                        if rec_stderr == None : continue ## std bight be too low and sometime none cuz small group

                        rec_est    = float(rec_est)
                        rec_stderr = float(rec_stderr)
                        # key = tuple(key)
                        # if any(v is None for v in key):
                        #     continue
                        if any(v is None for v in gkey):
                            continue

                        if jm:

                            _, cond_sel_list = self._qualify_group_cols(group_cols, schL, schR)
                            kc = " AND ".join(f"{col} = %s" for col in cond_sel_list)
                            count_sql = (
                                f"SELECT COUNT(*) FROM {tblL} AS l JOIN {tblR} AS r USING({key}) "
                                f"{where} {'AND' if where else 'WHERE'} {kc};"
                            )
                            self.cur.execute(count_sql, gkey)
                        else:
                            kc = " AND ".join(f'"{c}" = %s' for c in group_cols)
                            count_sql = (
                              f"SELECT COUNT(*) FROM {tbl} "
                              f"{where} {'AND' if where else 'WHERE'} {kc};"
                            )
                            self.cur.execute(count_sql, gkey)
                        # self.cur.execute(count_sql, key)  # `key` here is the tuple of group values from the row
                        n = self.cur.fetchone()[0]

                        if has_truth:
                            gt_val = float(gt_rows[gkey])
                            acc_g  = (1 - abs(rec_est - gt_val)/gt_val)*100
                        else:
                            gt_val = None
                            acc_g  = None

                        group_metrics.append({
                            'group':        gkey,
                            'n':            n,
                            'estimate':     rec_est,
                            'stderr':       rec_stderr,
                            'ground_truth': gt_val,
                            'accuracy':     acc_g
                        })

                    # macro / micro accuracy
                    analyzer = myDataAnalyzer.myDataAnalyzer()
                    macro = analyzer.unweighted_accuracy(group_metrics)
                    micro = analyzer.weighted_accuracy(group_metrics)

                    results[ds_name][qry] = {
                        'estimate':         est,
                        'stderr':           stderr,
                        'QT':               QT,
                        'JQT':              JQT,
                        'CI95':             (lo, hi),
                        'normalized_width': (hi - lo)/est if est else None,
                        'per_group':        group_metrics,
                        'accuracy_macro':   macro,
                        'accuracy_micro':   micro
                    }
                    print("=========== END =================")

        return results




### --- start mcar mar runners ----
# import json
# with open("configs/all_queries.json") as f:
# # with open("configs/all_queries_dist.json") as f:
#     allData = json.load(f)
#     csv_queries_bank_mar = allData["bank_mar"]
#     csv_queries_nyc_mar  = allData["nyc_mar"]
#     csv_queries_real_MAR = allData["real_mar"]
#     csv_queries_real_MCAR = allData["real_mcar"]

#     csv_queries_bank_mcar = allData["bank_mcar"]
#     csv_queries_nyc_mcar = allData["nyc_mcar"]
#     csv_queries_bitcoin_mcar = allData["bit_macr"]
#     csv_queries_bitcoin_mar = allData["bit_mar"]

# query_config=[csv_queries_bank_mar,csv_queries_nyc_mar,
#               csv_queries_real_MAR, csv_queries_real_MCAR,
#               csv_queries_bank_mcar, csv_queries_nyc_mcar,
#               csv_queries_bitcoin_mcar, csv_queries_bitcoin_mar
#               ]

# query_config=[
#               csv_queries_bitcoin_mcar, csv_queries_bitcoin_mar
#               ] ###
# acc_list=[]
# delta_w=[]
# QT_s =[]
# JQT_s =[]
# element_to_remove = -1
# # connect & run
# conn = psycopg2.connect(
#     host="localhost", port=5433, dbname="mydb",
#     user="alzamill", password=os.environ.get("PGPASSWORD", "")
# )
# for csv_queries in query_config:
#     executor = QueryExecutor(conn, csv_queries)
#     results = executor.compute2()
#     for datasetName in csv_queries:
#         acc_list=[]
#         delta_w=[]
#         QT_s =[]
#         JQT_s =[]
#         for q, r in results[datasetName].items():
#             #lo, hi = r["CI95"]
#             print(f"\nQuery: {q}")
#             print(f" Estimate           = {r['estimate']:.2f} ± {r['stderr']:.4f}")
#         # print(f" Ground Truth       = {r['ground_truth']:.2f}")
#             if 'accuracy' in r:
#                 print(f" Accuracy           = {r['accuracy']:.2f}%")
#                 acc_list.append(r['accuracy']) ## just to compare with dist based
#             if 'accuracy_micro' in r:
#                 acc_list.append(r['accuracy_micro'])
#                 print(f"GBY Accuracy           = {r['accuracy_micro']:.2f}%")
#            # print(f" 95% CI             = [{lo:.2f}, {hi:.2f}]")
#             print(f" Normalized width   = {r['normalized_width']:.4f}")
#             delta_w.append(r['normalized_width'])
#             print(f" QT   = {r['QT']:.5f}")
#             if r['QT'] != -1:
#                 QT_s.append(r['QT'])
#                 # if 'accuracy' in r:
#                   # acc_list.append(r['accuracy']) ## just to compare with dist based
#             print(f" JQT   = {r['JQT']:.5f}")
#             if r['JQT'] != -1:
#                 JQT_s.append(r['JQT'])

#         # if element_to_remove in QT_s:
#         #     QT_s.remove(element_to_remove)

#         # if element_to_remove in JQT_s:
#         #     JQT_s.remove(element_to_remove)

#         from statistics import mean

#         #print("average accs for bank mar 5%:",  mean(acc_list))
#         if not acc_list:
#                 accMean= -1
#         else:
#             accMean= mean(acc_list)
#         analyzer = myDataAnalyzer.myDataAnalyzer(datasetName=datasetName, output_dir="psql_results",out_file="mar_psql_acc_3_wjoin.txt")
#         analyzer.add_stats(accMean,mean(QT_s),mean(JQT_s),mean(delta_w))
#         # analyzer.add_stats(accMean,mean(QT_s),-1,mean(delta_w))
#         analyzer.addNewLine()
#     executor.cur.close()
# conn.close()


### --- end mcar mar runners ----






# # ### --- start mnar runners ----
# import json
# with open("data/mnar1_agg_inj_query.json") as f:
# # with open("configs/all_queries_dist.json") as f:
#     allData = json.load(f)
#     csv_queries_bank_mnar = allData["bank_manr1_agg"]
#     csv_queries_nyc_mnar  = allData["nyc_manr1_agg"]
#     csv_queries_bitcoin_mnar = allData["bit_manr1_agg"]

# query_config=[csv_queries_bank_mnar,csv_queries_nyc_mnar, csv_queries_bitcoin_mnar]

# ##
# acc_list=[]
# delta_w=[]
# QT_s =[]
# JQT_s =[]
# element_to_remove = -1
# # connect & run
# conn = psycopg2.connect(
#     host="localhost", port=5433, dbname="mydb",
#     user="alzamill", password=os.environ.get("PGPASSWORD", "")
# )
# for csv_queries in query_config:
#     executor = QueryExecutor(conn, csv_queries)
#     results = executor.compute2()
#     for datasetName in csv_queries:
#         acc_list=[]
#         delta_w=[]
#         QT_s =[]
#         JQT_s =[]
#         for q, r in results[datasetName].items():
#             #lo, hi = r["CI95"]
#             print(f"\nQuery: {q}")
#             print(f" Estimate           = {r['estimate']:.2f} ± {r['stderr']:.4f}")
#         # print(f" Ground Truth       = {r['ground_truth']:.2f}")
#             if 'accuracy' in r:
#                 print(f" Accuracy           = {r['accuracy']:.2f}%")
#                 acc_list.append(r['accuracy']) ## just to compare with dist based
#             if 'accuracy_micro' in r:
#                 acc_list.append(r['accuracy_micro'])
#                 print(f"GBY Accuracy           = {r['accuracy_micro']:.2f}%")
#            # print(f" 95% CI             = [{lo:.2f}, {hi:.2f}]")
#             print(f" Normalized width   = {r['normalized_width']:.4f}")
#             delta_w.append(r['normalized_width'])
#             print(f" QT   = {r['QT']:.5f}")
#             if r['QT'] != -1:
#                 QT_s.append(r['QT'])
#                 # if 'accuracy' in r:
#                   # acc_list.append(r['accuracy']) ## just to compare with dist based
#             print(f" JQT   = {r['JQT']:.5f}")
#             if r['JQT'] != -1 or r['JQT']==0 :
#                 r['JQT']+=1e-9
#                 JQT_s.append(r['JQT'])

#         # if element_to_remove in QT_s:
#         #     QT_s.remove(element_to_remove)

#         # if element_to_remove in JQT_s:
#         #     JQT_s.remove(element_to_remove)

#         from statistics import mean

#         #print("average accs for bank mar 5%:",  mean(acc_list))
#         if not acc_list:
#                 accMean= -1
#         else:
#             accMean= mean(acc_list)
#         analyzer = myDataAnalyzer.myDataAnalyzer(datasetName=datasetName, output_dir="psql_results",out_file="mnar1_agg_psql_injected.txt")
#         analyzer.add_stats(accMean,mean(QT_s),mean(JQT_s),mean(delta_w))
#         # analyzer.add_stats(accMean,mean(QT_s),-1,mean(delta_w))
#         analyzer.addNewLine()
#     executor.cur.close()
# conn.close()


# ### --- end mnar runners ----



# ### --- start real mnar runners ----
if __name__ == "__main__":
    import json
    with open("configs/real_mnar_agg.json") as f:
        allData = json.load(f)
        csv_queries_mi_mnar = allData["mi_mnar_avg"]

    query_config=[csv_queries_mi_mnar]

    acc_list=[]
    delta_w=[]
    QT_s =[]
    JQT_s =[]
    element_to_remove = -1
    conn = psycopg2.connect(
        host="localhost", port=5433, dbname="mydb",
        user="alzamill", password=os.environ.get("PGPASSWORD", "")
    )
    for csv_queries in query_config:
        executor = QueryExecutor(conn, csv_queries)
        results = executor.compute2()
        for datasetName in csv_queries:
            acc_list=[]
            delta_w=[]
            QT_s =[]
            JQT_s =[]
            for q, r in results[datasetName].items():
                print(f"\nQuery: {q}")
                print(f" Estimate           = {r['estimate']:.2f} ± {r['stderr']:.4f}")
                if 'accuracy' in r:
                    print(f" Accuracy           = {r['accuracy']:.2f}%")
                    acc_list.append(r['accuracy'])
                    if 'accuracy_micro' in r:
                        acc_list.append(r['accuracy_micro'])
                        print(f"GBY Accuracy           = {r['accuracy_micro']:.2f}%")
                if r['normalized_width'] is not None and pd.notna(r['normalized_width']):
                    print(f"Normalized width = {r['normalized_width']:.4f}")
                else:
                    r['normalized_width']=0
                    print("Normalized width = None")
                delta_w.append(r['normalized_width'])
                print(f" QT   = {r['QT']:.5f}")
                if r['QT'] != -1:
                    QT_s.append(r['QT'])
                print(f" JQT   = {r['JQT']:.5f}")
                if r['JQT'] != -1 or r['JQT']==0 :
                    r['JQT']+=1e-9
                    JQT_s.append(r['JQT'])

            from statistics import mean

            if not acc_list:
                accMean= -1
            else:
                accMean= mean(acc_list)
            analyzer = myDataAnalyzer.myDataAnalyzer(datasetName=datasetName, output_dir="psql_results",out_file="real_mnar_agg_psql.txt")
            analyzer.add_stats(accMean,mean(QT_s),mean(JQT_s),mean(delta_w))
            analyzer.addNewLine()
        executor.cur.close()
    conn.close()


### --- end mnar runners ----


### for indidiual test

# csv_queries = {
#   ## manr test  works---------------------------------

#   "bitcoin_mnar_5": {
# "csv": ["data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_5.csv",
# "data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_5_join2.csv",
# "data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_5_join1.csv"],
# "table": [
# "mnar5_bitc0",
# "mnar5_bitc2j",
# "mnar5_bitc1j"
# ],
# "complete_csv": [
#     "rwDatasets/BitcoinHeistData_complete.csv",
#     "Injected_JoinsData/bit_complete1.csv",
#     "Injected_JoinsData/bit_complete2.csv"
# ],
# "complete_table": [
# "full_bitc0",
# "full_bitc1",
# "full_bitc2"
# ],

# "MNAR_Strata": [
#     ['day', 'neighbors', 'weight', 'year'],
#     ['neighbors', 'day', 'year', 'weight'],
#     ['day', 'neighbors']

# ],
# "mar_causes":[
#     {"year": ['day', 'neighbors', 'weight']},
#      {"year": ['day', 'neighbors', 'weight']},
#      {}

# ],
# "queries": [
#     # "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_5.csv GROUP BY looped",
#     # "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_5.csv",
#     # "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_5.csv where length > 2 and weight < 3",
#     # "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_5.csv where neighbors > 0",
#     "Select AVG(income) from data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_5_join2.csv join data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_5_join1.csv USING (id) where day > 4"
#      ],
#     },
#   ## manr test  works---------------------------------
# #   "bitcoin_mnar_5": {
# # "csv": ["data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_10.csv",
# # "data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_10_join1.csv",
# # "data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_10_join2.csv"],
# # "table": [
# # "mnar10_bitc0",
# # "mnar10_bitc1",
# # "mnar10_bitc2"
# # ],
# # "complete_csv": [
# #     "rwDatasets/BitcoinHeistData_complete.csv",
# #     "Injected_JoinsData/bit_complete1.csv",
# #     "Injected_JoinsData/bit_complete2.csv"
# # ],
# # "complete_table": [
# # "full_bitc0",
# # "full_bitc1",
# # "full_bitc2"
# # ],

# # "MNAR_Strata": [
# #     ['day', 'neighbors', 'weight', 'year'],
# #     ['neighbors', 'day', 'year', 'weight'],
# #     ['day', 'neighbors']

# # ],
# # "mar_causes":[
# #     {"year": ['day', 'neighbors', 'weight']},
# #      {"year": ['day', 'neighbors', 'weight']},
# #      {}

# # ],
# # "queries": [
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_10.csv GROUP BY looped",
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_10.csv",
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_10.csv where length > 2 and weight < 3",
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_10.csv where neighbors > 0",
# #     "Select AVG(income) from data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_10_join1.csv join data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_10_join2.csv USING (id) where day > 4"
# #      ],
# #     },
# #   ## manr test  works---------------------------------
# #   "bitcoin_mnar_5": {
# # "csv": ["data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_20.csv",
# # "data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_20_join1.csv",
# # "data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_20_join2.csv"],
# # "table": [
# # "mnar20_bitc0",
# # "mnar20_bitc1",
# # "mnar20_bitc2"
# # ],
# # "complete_csv": [
# #     "rwDatasets/BitcoinHeistData_complete.csv",
# #     "Injected_JoinsData/bit_complete1.csv",
# #     "Injected_JoinsData/bit_complete2.csv"
# # ],
# # "complete_table": [
# # "full_bitc0",
# # "full_bitc1",
# # "full_bitc2"
# # ],

# # "MNAR_Strata": [
# #     ['day', 'neighbors', 'weight', 'year'],
# #     ['neighbors', 'day', 'year', 'weight'],
# #     ['day', 'neighbors']

# # ],
# # "mar_causes":[
# #     {"year": ['day', 'neighbors', 'weight']},
# #      {"year": ['day', 'neighbors', 'weight']},
# #      {}

# # ],
# # "queries": [
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_20.csv GROUP BY looped",
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_20.csv",
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_20.csv where length > 2 and weight < 3",
# #     "Select AVG(income) From data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_20.csv where neighbors > 0",
# #     "Select AVG(income) from data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_20_join1.csv join data/Mnar1JoinsData/BitcoinHeistData_agg_mnar1_20_join2.csv USING (id) where day > 4"
# #      ],
# #     },

# ###-------------------------------------


# }

# acc_list=[]
# delta_w=[]
# QT_s =[]
# JQT_s =[]
# element_to_remove = -1
# # connect & run
# conn = psycopg2.connect(
#     host="localhost", port=5433, dbname="mydb",
#     user="alzamill", password=os.environ.get("PGPASSWORD", "")
# )

# executor = QueryExecutor(conn, csv_queries)
# results = executor.compute2()
# for datasetName in csv_queries:
#     for q, r in results[datasetName].items():
#         lo, hi = r["CI95"]
#         print(f"\nQuery: {q}")
#         print(f" Estimate           = {r['estimate']:.2f} ± {r['stderr']:.4f}")
#     # print(f" Ground Truth       = {r['ground_truth']:.2f}")
#         if 'accuracy' in r:
#             print(f" Accuracy           = {r['accuracy']:.2f}%")
#             acc_list.append(r['accuracy'])
#         if 'accuracy_micro' in r:
#             acc_list.append(r['accuracy_micro'])
#             print(f" Group by query Accuracy           = {r['accuracy_micro']:.2f}%")
#         print(f" 95% CI             = [{lo:.2f}, {hi:.2f}]")
#         print(f" Normalized width   = {r['normalized_width']:.4f}")
#         delta_w.append(r['normalized_width'])
#         print(f" QT   = {r['QT']:.5f}")
#         if r['QT'] != -1:
#             QT_s.append(r['QT'])
#         # print(f" JQT   = {r['JQT']:.5f}")
#         # if r['JQT'] != -1:
#         #     JQT_s.append(r['JQT'])

#     if element_to_remove in QT_s:
#          QT_s.remove(element_to_remove)

#     # if element_to_remove in JQT_s:
#     #     JQT_s.remove(element_to_remove)

#     from statistics import mean

#     #print("average accs for bank mar 5%:",  mean(acc_list))
#     if not acc_list:
#             accMean= -1
#     else:
#         accMean= mean(acc_list)
#     # analyzer = myDataAnalyzer.myDataAnalyzer(datasetName=datasetName, output_dir="psql_results",out_file="mar_psql.txt")
#     # analyzer.add_stats(accMean,mean(QT_s),mean(JQT_s),mean(delta_w))
#     # analyzer.addNewLine()
# executor.cur.close()
# conn.close()
