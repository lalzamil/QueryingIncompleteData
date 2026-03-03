import re
import time
import csv
import pandas as pd
import numpy as np
import psycopg2
from io import StringIO
import myDataAnalyzer 
from statistics import mean
import signal

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()


signal.signal(signal.SIGALRM, _timeout_handler)

class IntervalAnswers:
    """
    Implements Algorithm 2 (“Query Evaluation of AVG”) from Zhang et al. (2019)  
    """

    def __init__(self, cursor):
        self.cur = cursor

    def getIntervalAnswer_singleRelation(self,
                                         mar_table_name: str,
                                         complete_table: str,
                                         attr: str,
                                         where_sql: str = ""
                                        ) :
        """
         implements Algorithm 2 for AVG on a single incomplete table:
        """

       
        ws = where_sql.strip().rstrip(';')
        if ws:
            wc = ws.lstrip()
            if not wc.upper().startswith("WHERE "):
                wc = "WHERE " + wc
            where_clause = wc
        else:
            where_clause = ""

        #    We look for any "col <=/>=/</>=" pattern to discover which columns appear in the predicate.
        predicate_cols = re.findall(r'([A-Za-z_]\w*)\s*(?:=|<|>|<=|>=)', where_clause)
        predicate_cols = list(set(col.lower() for col in predicate_cols))


        #    We fetch *every* attr‐cell (which might be NULL) from rows that certainly satisfy the predicate.
        if where_clause:
            ta_all_sql = f"""
                SELECT {attr}
                FROM {mar_table_name}
                {where_clause};
            """
        else:
            ta_all_sql = f"""
                SELECT {attr}
                FROM {mar_table_name};
            """
        self.cur.execute(ta_all_sql)
        TA_all = [r[0] for r in self.cur.fetchall()]  # list possibly containing None

        # ──────
        if predicate_cols:
            null_conds = " OR ".join(f'"{c}" IS NULL' for c in predicate_cols)
            ma_all_sql = f"""
                SELECT {attr}
                FROM {mar_table_name}
                WHERE ({null_conds});
            """
            self.cur.execute(ma_all_sql)
            MA_all = [r[0] for r in self.cur.fetchall()]
        else:
            # If no predicate, then there are no “maybe rows,” because a row is either in TA or fails predicate.
            MA_all = []

      
        N = len(TA_all) + len(MA_all)
        if N == 0:
            # No relevant rows at all
            return (0.0, 0.0)

       
        if complete_table:
            if where_clause:
                minmax_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {complete_table}
                    {where_clause}
                      AND {attr} IS NOT NULL;
                """
            else:
                minmax_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {complete_table}
                    WHERE {attr} IS NOT NULL;
                """
        else:
            # No complete table:
            if where_clause:
                minmax_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {mar_table_name}
                    {where_clause}
                      AND {attr} IS NOT NULL;
                """
            else:
                minmax_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {mar_table_name}
                    WHERE {attr} IS NOT NULL;
                """
        self.cur.execute(minmax_sql)
        row = self.cur.fetchone()
        if row is None or row[0] is None or row[1] is None:
            a, b = 0.0, 0.0
        else:
            a, b = float(row[0]), float(row[1])

        # UPPER BOUND (substitute every None by b)
     
        TA_sub_b = [ (b if v is None else float(v)) for v in TA_all ]
        MA_sub_b = [ (b if v is None else float(v)) for v in MA_all ]

      
        total_sum_u = sum(TA_sub_b) + sum(MA_sub_b)
        MA_desc_b = sorted(MA_sub_b, reverse=True)

  
        for v in MA_desc_b:
            current_avg_u = total_sum_u / N
            if v > current_avg_u:
                total_sum_u = total_sum_u - b + v
            else:
                break

        ub = total_sum_u / N


        TA_sub_a = [ (a if v is None else float(v)) for v in TA_all ]
        MA_sub_a = [ (a if v is None else float(v)) for v in MA_all ]

     
        total_sum_l = sum(TA_sub_a) + sum(MA_sub_a)


        MA_asc_a = sorted(MA_sub_a)


        for w in MA_asc_a:
            current_avg_l = total_sum_l / N
            if w < current_avg_l:
                total_sum_l = total_sum_l - a + w
            else:
                break

        lb = total_sum_l / N

        return (lb, ub)

    def getIntervalAnswer_Join(self,
                               mar_table_names:     list,
                               complete_table_names:list,
                               attr:                str,
                               where_sql:           str = ""
                              ):
        """
        extends Algorithm 2 for AVG on  join of multiple incomplete tables. 

        """

     
        ws = where_sql.strip().rstrip(';')
        if ws:
            wc = ws.lstrip()
            if not wc.upper().startswith("WHERE "):
                wc = "WHERE " + wc
            where_clause = wc
        else:
            where_clause = ""

      
        colsets = []
        for tbl in mar_table_names:
            self.cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (tbl,))
            cols = {r[0].lower() for r in self.cur.fetchall()}
            colsets.append(cols)
        join_keys = sorted(set.intersection(*colsets))

    
        if len(mar_table_names) == 1:
            return self.getIntervalAnswer_singleRelation(
                mar_table_name = mar_table_names[0],
                complete_table = (complete_table_names[0] if complete_table_names else None),
                attr           = attr,
                where_sql      = where_clause
            )

     
        from_mar_clause = mar_table_names[0]
        for t in mar_table_names[1:]:
            keys_comma = ", ".join(f'"{k}"' for k in join_keys)
            from_mar_clause = f"{from_mar_clause} JOIN {t} USING ({keys_comma})"

    
        count_join_sql = f"""
            SELECT COUNT(*)
            FROM {from_mar_clause}
            {where_clause};
        """
        self.cur.execute(count_join_sql)
        N_join = int(self.cur.fetchone()[0] or 0)
        if N_join == 0:
            return (0.0, 0.0)

        #  TA_join_all 
        if where_clause:
            ta_join_sql = f"""
                SELECT {attr}
                FROM {from_mar_clause}
                {where_clause};
            """
        else:
            ta_join_sql = f"""
                SELECT {attr}
                FROM {from_mar_clause};
            """
        self.cur.execute(ta_join_sql)
        TA_join_all = [r[0] for r in self.cur.fetchall()]

        #   MA_join_all 
        predicate_cols = re.findall(r'([A-Za-z_]\w*)\s*(?:=|<|>|<=|>=)', where_clause)
        predicate_cols = list(set(col.lower() for col in predicate_cols))
        if predicate_cols:
            null_conds = " OR ".join(f'"{c}" IS NULL' for c in predicate_cols)
            ma_join_sql = f"""
                SELECT {attr}
                FROM {from_mar_clause}
                WHERE ({null_conds});
            """
            self.cur.execute(ma_join_sql)
            MA_join_all = [r[0] for r in self.cur.fetchall()]
        else:
            MA_join_all = []

       
        if where_clause:
            sum_obs_join_sql = f"""
                SELECT COALESCE(SUM({attr}::NUMERIC), 0)
                FROM {from_mar_clause}
                {where_clause}
                  AND {attr} IS NOT NULL;
            """
            n_obs_join_sql = f"""
                SELECT COUNT({attr})
                FROM {from_mar_clause}
                {where_clause}
                  AND {attr} IS NOT NULL;
            """
        else:
            sum_obs_join_sql = f"""
                SELECT COALESCE(SUM({attr}::NUMERIC), 0)
                FROM {from_mar_clause}
                WHERE {attr} IS NOT NULL;
            """
            n_obs_join_sql = f"""
                SELECT COUNT({attr})
                FROM {from_mar_clause}
                WHERE {attr} IS NOT NULL;
            """
        self.cur.execute(sum_obs_join_sql)
        sum_obs_join = float(self.cur.fetchone()[0] or 0.0)
        self.cur.execute(n_obs_join_sql)
        n_obs_join = int(self.cur.fetchone()[0] or 0)
        n_miss_join = N_join - n_obs_join

       
        if complete_table_names and all(complete_table_names):
            from_full_clause = complete_table_names[0]
            for t in complete_table_names[1:]:
                keys_comma = ", ".join(f'"{k}"' for k in join_keys)
                from_full_clause = f"{from_full_clause} JOIN {t} USING ({keys_comma})"

            if where_clause:
                minmax_full_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {from_full_clause}
                    {where_clause}
                      AND {attr} IS NOT NULL;
                """
            else:
                minmax_full_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {from_full_clause}
                    WHERE {attr} IS NOT NULL;
                """
            self.cur.execute(minmax_full_sql)
            row = self.cur.fetchone()
            if row is None or row[0] is None or row[1] is None:
                a_join, b_join = 0.0, 0.0
            else:
                a_join, b_join = float(row[0]), float(row[1])
        else:
            
            if where_clause:
                minmax_mar_join_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {from_mar_clause}
                    {where_clause}
                      AND {attr} IS NOT NULL;
                """
            else:
                minmax_mar_join_sql = f"""
                    SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                    FROM {from_mar_clause}
                    WHERE {attr} IS NOT NULL;
                """
            self.cur.execute(minmax_mar_join_sql)
            row = self.cur.fetchone()
            if row is None or row[0] is None or row[1] is None:
                a_join, b_join = 0.0, 0.0
            else:
                a_join, b_join = float(row[0]), float(row[1])

     
        TA_sub_b = [(b_join if v is None else float(v)) for v in TA_join_all]
        MA_sub_b = [(b_join if v is None else float(v)) for v in MA_join_all]

        total_sum_u = sum(TA_sub_b) + sum(MA_sub_b)
        MA_desc_b = sorted(MA_sub_b, reverse=True)

        for v in MA_desc_b:
            current_avg_u = total_sum_u / N_join
            if v > current_avg_u:
                total_sum_u = total_sum_u - b_join + v
            else:
                break

        ub_join = total_sum_u / N_join

        
        TA_sub_a = [(a_join if v is None else float(v)) for v in TA_join_all]
        MA_sub_a = [(a_join if v is None else float(v)) for v in MA_join_all]

        total_sum_l = sum(TA_sub_a) + sum(MA_sub_a)
        MA_asc_a = sorted(MA_sub_a)

        for w in MA_asc_a:
            current_avg_l = total_sum_l / N_join
            if w < current_avg_l:
                total_sum_l = total_sum_l - a_join + w
            else:
                break

        lb_join = total_sum_l / N_join

        return (lb_join, ub_join)




class IntervalQueryExecutor:
    TYPE_MAP = {'int64': 'BIGINT', 'float64': 'NUMERIC'}

    def __init__(self, conn, csv_queries):
        """
        conn: a psycopg2 connection
        csv_queries: a dict of the form
          {
            "datasetName": {
              "csv":           [ path_to_mar_csv_1, path_to_mar_csv_2, … ],
              "table":         [ mar_table_name_1,   mar_table_name_2,  … ],
              "complete_csv":  [ path_to_full_csv_1, path_to_full_csv_2, … ],
              "complete_table":[ full_table_name_1, full_table_name_2, … ],
              "queries":       [ "SELECT AVG(attr) FROM path [WHERE …] [GROUP BY …];", … ]
            },
            …
          }
        """
        self.conn        = conn
        self.cur         = conn.cursor()
        self.csv_queries = csv_queries

        # Build mappings:
        self.mar_table_for  = {} 
        self.full_table_for = {}  

        self.groundTruth = any(meta.get('complete_csv') for meta in csv_queries.values())

        for meta in csv_queries.values():
            for path, tbl in zip(meta['csv'], meta['table']):
                self.mar_table_for[path] = tbl
            if meta.get('complete_csv'):
                for path, tbl in zip(meta['complete_csv'], meta['complete_table']):
                    self.full_table_for[path] = tbl

        # Load all tables into psql
        self.prepareTables()

       
        self.recovery = IntervalAnswers(self.cur)

    def inferSchema(self, path):
        """
        Inspect the first 100 rows of a CSV to guess each column's dtype.
        Returns a dict { column_name  - > SQL type } for CREATE TABLE.
        """
        df = pd.read_csv(path, nrows=100)
        df.columns = df.columns.str.lower()
        df = df.loc[:, ~df.columns.str.startswith('unnamed:')]
        schema = {}
        for col, dt in df.dtypes.items():
            dtstr = str(dt)
            schema[col] = self.TYPE_MAP.get(dtstr, 'TEXT')
        return schema

    def loadCSV(self, path, table_name, cols):
        """
        Load a CSV file into a Postgres table via COPY.
        - `path` is the filesystem path to the CSV
        - `table_name` is the target table in Postgres
        - `cols` is the list of column names (lower-cased)
        """
        df = pd.read_csv(path, keep_default_na=True, na_values=['', ' ', '\\N'])
        df.columns = df.columns.str.lower()
        df = df.loc[:, ~df.columns.str.startswith('unnamed:')]
        df = df[cols]

        buf = StringIO()
        df.to_csv(buf, index=False, header=False, na_rep='\\N', quoting=csv.QUOTE_MINIMAL)
        buf.seek(0)

        col_list = ', '.join(f'"{c}"' for c in cols)
        sql = f"""
            COPY {table_name}({col_list})
            FROM STDIN WITH (FORMAT CSV, NULL '\\N')
        """
        self.cur.copy_expert(sql, buf)
        self.conn.commit()

    def prepareTables(self):
        """
        For each MAR-CSV path  - > mar_table_name:
        1) DROP TABLE IF EXISTS
        2) CREATE TABLE with inferred schema
        3) COPY the CSV contents into it
        """
        for path, tbl in self.mar_table_for.items():
            schema = self.inferSchema(path)
            cols   = list(schema.keys())
            ddl    = ',\n  '.join(f'"{c}" {t}' for c, t in schema.items())

            self.cur.execute(f"""
                DROP TABLE IF EXISTS {tbl};
                CREATE TABLE {tbl} ({ddl});
                TRUNCATE {tbl};
            """)
            self.conn.commit()

            self.loadCSV(path, tbl, cols)

    def estimate_interval(self):
        """
        For each dataset & each query in self.csv_queries:
        1) Strip trailing semicolon, then parse
           “SELECT AVG(attr) FROM <from_clause> [WHERE …] [GROUP BY …];”
        2) Extract all “.csv” paths from FROM clause (one or a JOIN).
        3) Map each CSV path  - > its MAR table name, and collect
           the corresponding complete-CSV path(s) as well.
        4) If no GROUP BY: call getIntervalAnswer_singleRelation or getIntervalAnswer_Join.
           If GROUP BY (single table only): run per-group SQL + pandas logic.
        5) Return a nested dict: results[dataset][query] = {...}.
        """
        results = {}

        for ds_name, meta in self.csv_queries.items():
            results[ds_name] = {}

            csv_list  = meta['csv']                  # list of MAR CSV paths
            full_list = meta.get('complete_csv', []) # parallel list of COMPLETE CSV paths

            for qry in meta['queries']:
                QTime = -1
                JQTime =-1
                # mytimer = time.time()
                # time_out_limit=300
                # elabsed = time.time()
                try:
                    # Schedule an alarm in 300 seconds
                    signal.alarm(120)
                    # 
                    qry_clean = qry.strip().rstrip(';')

                    #  Parse “SELECT AVG(attr) FROM … [WHERE …] [GROUP BY …]”
                    m = re.match(
                        r'''SELECT\s+AVG\(\s*(\w+)\s*\)\s+FROM\s+(.+?)'''
                        r'''(?:\s+WHERE\s+(.+?))?'''
                        r'''(?:\s+GROUP\s+BY\s+(.+))?$''',
                        qry_clean, re.IGNORECASE
                    )
                    if not m:
                        raise ValueError(f"Cannot parse query: {qry!r}")
                    attr, from_clause, where_clause, gb_clause = m.groups()
                    attr = attr.lower()
                    from_clause = from_clause.strip()

                 
                    if where_clause:
                        wc = where_clause.strip().rstrip(';')
                        where_sql = f"WHERE {wc}"
                    else:
                        where_sql = ""

                  
                    if gb_clause:
                        gc = gb_clause.strip().rstrip(';')
                        group_cols = [c.strip().strip('"').lower() for c in gc.split(',')]
                    else:
                        group_cols = []

                  
                    csv_paths = re.findall(r'(\S+?\.csv)', from_clause)
                    if not csv_paths:
                        raise ValueError(f"No CSV path found in FROM clause: {from_clause!r}")
                    for p in csv_paths:
                        if p not in csv_list:
                            raise ValueError(f"FROM clause CSV {p!r} not in csv_list for dataset {ds_name}")

                  
                    mar_table_names = [self.mar_table_for[p] for p in csv_paths]
                    complete_table_names = []
                    for p in csv_paths:
                        if p in self.full_table_for:
                            complete_table_names.append(self.full_table_for[p])

                  
                    if len(group_cols) == 0:
                        

                        if len(mar_table_names) == 1:
                            mar_tbl = mar_table_names[0]
                            full_tbl = complete_table_names[0] if len(complete_table_names) == 1 else None
                            start = time.time()
                            lb, ub = self.recovery.getIntervalAnswer_singleRelation(
                                mar_table_name = mar_tbl,
                                complete_table = full_tbl,
                                attr           = attr,
                                where_sql      = where_sql
                            )
                            QTime = time.time() - start
                        else:
                            mar_tbls = mar_table_names
                            if len(complete_table_names) == len(mar_tbls):
                                full_tbls = complete_table_names
                            else:
                                full_tbls = [None] * len(mar_tbls)
                            start = time.time()
                            lb, ub = self.recovery.getIntervalAnswer_Join(
                                mar_table_names      = mar_tbls,
                                complete_table_names = full_tbls,
                                attr                 = attr,
                                where_sql            = where_sql
                            )
                            JQTime = time.time() - start

                        
                        results[ds_name][qry] = {
                            'lb': lb,
                            'ub': ub,
                            'QT': QTime,
                            'JQT': JQTime
                        }

                   
                    else:
                        if len(mar_table_names) > 1:
                            raise ValueError("GROUP BY over a JOIN is not implemented in this snippet.")

                        mar_tbl = mar_table_names[0]
                        full_tbl = complete_table_names[0] if complete_table_names else None

                        sel_gb = ", ".join(f'"{c}"' for c in group_cols)
                        group_sql = f"""
                            SELECT
                            {sel_gb}                   AS group_key,
                            COUNT(*)                   AS N_g,
                            COALESCE(SUM({attr}::NUMERIC), 0)  AS sum_obs_g,
                            COUNT({attr})              AS n_obs_g
                            FROM {mar_tbl}
                            {where_sql}
                            GROUP BY {sel_gb};
                        """
                        t0 = time.time()
                        self.cur.execute(group_sql)
                        rows = self.cur.fetchall()
                       

                        
                        a_dict = {}
                        b_dict = {}
                        for rec in rows:
                            *group_vals, N_g, sum_obs_g, n_obs_g = rec
                            conds = []
                            for c_name, c_val in zip(group_cols, group_vals):
                                if c_val is None:
                                # Use 'IS NULL' when the group key is actually NULL
                                 conds.append(f'"{c_name}" IS NULL')
                                elif isinstance(c_val, str):
                                    conds.append(f'"{c_name}" = \'{c_val}\'')
                                else:
                                    conds.append(f'"{c_name}" = {c_val}')
                            group_where = " AND ".join(conds)

                            if full_tbl:
                                if where_sql:
                                    minmax_g_sql = f"""
                                        SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                                        FROM {full_tbl}
                                        {where_sql}
                                        AND {attr} IS NOT NULL
                                        AND {group_where};
                                    """
                                else:
                                    minmax_g_sql = f"""
                                        SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                                        FROM {full_tbl}
                                        WHERE {attr} IS NOT NULL
                                        AND {group_where};
                                    """
                            else:
                                if where_sql:
                                    minmax_g_sql = f"""
                                        SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                                        FROM {mar_tbl}
                                        {where_sql}
                                        AND {attr} IS NOT NULL
                                        AND {group_where};
                                    """
                                else:
                                    minmax_g_sql = f"""
                                        SELECT MIN({attr}::NUMERIC), MAX({attr}::NUMERIC)
                                        FROM {mar_tbl}
                                        WHERE {attr} IS NOT NULL
                                        AND {group_where};
                                    """
                            self.cur.execute(minmax_g_sql)
                            row2 = self.cur.fetchone()
                            key = tuple(group_vals) if len(group_vals) > 1 else (group_vals[0],)
                            if row2 is None or row2[0] is None or row2[1] is None:
                                a_dict[key] = 0.0
                                b_dict[key] = 0.0
                            else:
                                a_dict[key] = float(row2[0])
                                b_dict[key] = float(row2[1])

                        
                        per_group = []
                        for rec in rows:
                            *group_vals, N_g, sum_obs_g, n_obs_g = rec

                            N_g = float(N_g)
                            sum_obs_g = float(sum_obs_g)
                            n_obs_g = float(n_obs_g)
                            key = tuple(group_vals) if len(group_vals) > 1 else (group_vals[0],)
                            n_miss_g = N_g - n_obs_g
                            a_g = a_dict[key]
                            b_g = b_dict[key]

                            lb_g = (sum_obs_g + n_miss_g * a_g) / N_g
                            ub_g = (sum_obs_g + n_miss_g * b_g) / N_g

                            per_group.append({
                                'group':    key,
                                'n':        N_g,
                                'interval': (lb_g, ub_g)
                            })

                        elapsed = time.time() - t0
                        results[ds_name][qry] = {
                            'per_group': per_group,
                            'QT':   elapsed
                        }
                
                except TimeoutException:
                    print(f"query {qry} took > 5 minutes; skipping to next.")
                    results[ds_name][qry] = {
                            'lb': -111,
                            'ub': -111,
                            'QT': -111,
                            'JQT': -111
                        }

                finally:
                  
                    signal.alarm(0)


        return results



# ## Example usage (to verify that semicolons no longer break the SQL):

# csv_queries = {
#     # "nyc_mcar_5%": {
#     #     "csv": [
#     #         "rwDatasets/nyc_MCAR_5.0.csv"
#     #     ],
#     #     "table": [
#     #         "mcar5_nyc0"
#     #     ],
#     #     "complete_csv": [
#     #       
#     #         # "rwDatasets/nyc_complete.csv"
#     #     ],
#     #     "complete_table": [
#     #         # Unused, since we read complete CSV via pandas
#     #     ],
#     #     "queries": [
#     #         # Note: both forms (with or without trailing “;”) now work
#     #         "SELECT AVG(passenger_count) FROM rwDatasets/nyc_MCAR_5.0.csv WHERE vendor_id = 2",
#     #         "SELECT AVG(passenger_count) FROM rwDatasets/nyc_MCAR_5.0.csv WHERE vendor_id = 2 GROUP BY vendor_id"
#     #     ]
#     # }

#            "bit_MCAR_10%": {
#       "csv": [
#         "rwDatasets/BitcoinHeistData_MCAR_1.0.csv",
#         "Injected_JoinsData/bitcoin_mcar10_1.csv",
#         "Injected_JoinsData/bitcoin_mcar10_2.csv"
#       ],
#       "table": [
#         "mcar10_bit0",
#         "mcar10_bit1",
#         "mcar10_bit2"
#       ],
#       "complete_csv": [
#         "rwDatasets/BitcoinHeistData_complete.csv",
#         "Injected_JoinsData/bit_complete1.csv",
#         "Injected_JoinsData/bit_complete2.csv"
#       ],
#       "complete_table": [
#         "full_bit0",
#         "full_bit1",
#         "full_bit2"
#       ],
#       "Cause": [
#         [],
#         [],
#         []
#       ],
#       "queries": [
#         # "Select AVG(income) From rwDatasets/BitcoinHeistData_MCAR_1.0.csv GROUP BY looped",
#         # "Select AVG(income) From rwDatasets/BitcoinHeistData_MCAR_1.0.csv",
#         # "Select AVG(income) From rwDatasets/BitcoinHeistData_MCAR_1.0.csv where length > 2 and weight < 3",
#         # "Select AVG(income) From rwDatasets/BitcoinHeistData_MCAR_1.0.csv where neighbors > 0",
#          "Select AVG(income) from Injected_JoinsData/bitcoin_mcar10_2.csv join Injected_JoinsData/bitcoin_mcar10_1.csv USING (id) where day > 4"
#       ]
#     },
# }

# # 1) Connect to Postgres:
# conn = psycopg2.connect(
#     host="localhost", port=****, dbname="db",
#     user="user", password="****"
# )
# executor = IntervalQueryExecutor(conn, csv_queries)

# # 2) Run the estimation:
# results = executor.estimate_interval()

# # 3) Print lb/ub for each query:
# for ds_name, queries_dict in results.items():
#     print(f"\n=== Dataset: {ds_name} ===")
#     for qry, out in queries_dict.items():
#         print("\nQuery:", qry)
#         if 'lb' in out and 'ub' in out:
#             print(f"   - > lb       = {out['lb']:.6f}")
#             print(f"   - > ub       = {out['ub']:.6f}")
#             print(f"   - > SQL time = {out['JQT']:.4f} sec")
#         elif 'per_group' in out:
#             print(f"   - > SQL time (grouped) = {out['QT']:.4f} sec")
#             for rec in out['per_group']:
#                 grp_key = rec['group']
#                 n       = rec['n']
#                 lb_g, ub_g = rec['interval']
#                 if isinstance(grp_key, tuple) and len(grp_key) == 1:
#                     grp_key = grp_key[0]
#                 print(f"    • group={grp_key!r}, n={n}, interval=({lb_g:.6f}, {ub_g:.6f})")

# executor.cur.close()
# conn.close()








    




import json
with open("all_queries.json") as f:
    allData = json.load(f)
    csv_queries_bank_mar = allData["bank_mar"]
    csv_queries_nyc_mar  = allData["nyc_mar"]
    csv_queries_real_MAR = allData["real_mar"]
    csv_queries_real_MCAR = allData["real_mcar"]

    csv_queries_bank_mcar = allData["bank_mcar"]
    csv_queries_nyc_mcar = allData["nyc_mcar"]
    csv_queries_bitcoin_mcar = allData["bit_macr"]
    csv_queries_bitcoin_mar = allData["bit_mar"]

query_config=[csv_queries_bank_mar,csv_queries_nyc_mar,
              csv_queries_real_MAR, csv_queries_real_MCAR,
              csv_queries_bank_mcar, csv_queries_nyc_mcar,
              csv_queries_bitcoin_mcar, csv_queries_bitcoin_mar    
              ]


# query_config=[csv_queries_bitcoin_mcar, csv_queries_bitcoin_mar   
# ]

# 1) Connect to Postgres (adjust host/port/dbname/user/password):
conn = psycopg2.connect(host="localhost", port=123, dbname="db",
                        user="user", password="****")


for csv_queries in query_config:
    executor = IntervalQueryExecutor(conn, csv_queries)
    results = executor.estimate_interval()
    
    for ds_name, queries_dict in results.items():
     #   print(f"\n=== Dataset: {ds_name} ===\n")
        delta_w=[]
        QT_s =[]
        JQT_s =[]
        for qry, out in queries_dict.items():
            print("Query:")
            print("  ", qry)

            # CASE 1: no GROUP BY ⇒ there is an 'lb' key
            if 'lb' in out and 'ub' in out:
                lb = out['lb']
                ub = out['ub']
                # print(f" lb= {lb:.6f}")
                # print(f" ub= {ub:.6f}")
                delta_w.append((lb,ub))
                if out['QT'] != -1:
                     QT_s.append(out['QT'])
                     qt = out['QT']
                     print(f" OT  = {qt:.4f} sec")
                if out['JQT'] != -1:
                     JQT_s.append(out['JQT'])
                     jqt = out['JQT']
                     print(f" JOT  = {jqt:.4f} sec")


                

            # CASE 2: GROUP BY 
            if 'per_group' in out:
                if out['QT'] != -1:
                    QT_s.append(out['QT'])
                    qt = out['QT']
                    print(f" OT  = {qt:.4f} sec")
                # print("per‐group intervals:")
                for rec in out['per_group']:
                    grp_key  = rec['group']    # a tuple of group‐column values
                    count_n  = rec['n']        # group size
                    (lb_g, ub_g) = rec['interval']
                    delta_w.append((lb_g, ub_g))
                    # # pretty‐print a tuple of length 1 without the trailing comma:
                    # if isinstance(grp_key, tuple) and len(grp_key) == 1:
                    #     grp_key = grp_key[0]
                    # print(f" group = {grp_key!r}, n = {count_n}, interval = ({lb_g:.6f}, {ub_g:.6f})")
        
        accMean= -1

        analyzer = myDataAnalyzer.myDataAnalyzer(datasetName=ds_name, output_dir="psql_results",out_file="mar_psql_interval_2.txt")
        delatw_mean=analyzer.average_normalized_width(delta_w)
        analyzer.add_stats(accMean,mean(QT_s),mean(JQT_s),delatw_mean)
        analyzer.addNewLine()
    executor.cur.close()
conn.close()


# # ## test

# ## for test :::


# csv_queries = {
#     "nyc_mcar_5%": {
#     "csv": [
#         "rwDatasets/nyc_MCAR_5.0.csv"
#     ],
#     "table": [
#         "mcar5_nyc0"
#     ],
#     "complete_csv": [
#         "rwDatasets/nyc_complete.csv"
#     ],
#     "complete_table": [
#         "full_nyc0"
#     ],
#     "queries": [
#         # Note the semicolon at the end so our regex matches cleanly:
#         "SELECT AVG(passenger_count) FROM rwDatasets/nyc_MCAR_5.0.csv where vendor_id = 2"
#     ]
#     }
# }


# # ## for test :::


# csv_queries = {



#     "bank_mar_5%": {
#     "csv": [
#       "rwDatasets/bank_MAR_5.0.csv",
#       "Injected_JoinsData/bank_mar5_1.csv",
#       "Injected_JoinsData/bank_mar5_2.csv"
#     ],
#     "table": ["mar_b0", "mar_b1", "mar_b2"],
#     "complete_csv": [
#       "rwDatasets/bank_complete.csv",
#       "Injected_JoinsData/bank_complete1.csv",
#       "Injected_JoinsData/bank_complete2.csv"
#     ],
#     "complete_table": ["full_b0", "full_b1", "full_b2"],
#     # "Cause": ["campaign","campaign","campaign"],
#         "Cause": [
#        ["campaign"],    # for bank_MAR_5.0.csv
#        ["campaign"],        # for bank_mar5_1.csv
#        ["campaign"]      # for bank_mar5_2.csv
#     ],
#     "queries": [
#       "SELECT AVG(balance) FROM rwDatasets/bank_MAR_5.0.csv",
#       # "Select AVG(balance) From rwDatasets/bank_MAR_5.0.csv where education = 'primary'",
#       # #"Select AVG(balance) From rwDatasets/bank_MAR_5.0.csv where loan = 'yes'",
#       # "Select AVG(balance) From rwDatasets/bank_MAR_5.0.csv where duration < 100",
#       #  "SELECT AVG(balance) FROM rwDatasets/bank_MAR_5.0.csv where loan = 'yes' GROUP BY housing",
#      # "SELECT AVG(balance) FROM Injected_JoinsData/bank_mar5_1.csv JOIN Injected_JoinsData/bank_mar5_2.csv USING (customer_id)"
#     #   "SELECT AVG(balance) FROM Injected_JoinsData/bank_mar5_1.csv JOIN Injected_JoinsData/bank_mar5_2.csv USING (customer_id) WHERE housing = 'no'"
#     ]
#   },

#   }






# # # 2) Run the estimation:
# # results = executor.estimate_interval()

# # # 3) Print out lb/ub for each query:
# # for ds_name in results:
# #     for query, out in results[ds_name].items():
# #         print(f"Dataset   = {ds_name}")
# #         print(f"Query     = {query}")
# #         print(f"   - > lb    = {out['lb']:.6f}")
# #         print(f"   - > ub    = {out['ub']:.6f}")
# #         print(f"   - > SQL‐time = {out['QT']:.4f} sec\n")

# # executor.cur.close()
# # conn.close()

# results = executor.estimate_interval()

# for ds_name, queries_dict in results.items():
#     print(f"\n=== Dataset: {ds_name} ===\n")

#     for qry, out in queries_dict.items():
#         print("Query:")
#         print("  ", qry)

#         # CASE 1: no GROUP BY ⇒ there is an 'lb' key
#         if 'lb' in out and 'ub' in out:
#             lb = out['lb']
#             ub = out['ub']
#             qt = out['QT']
#             print(f"   - > lb          = {lb:.6f}")
#             print(f"   - > ub          = {ub:.6f}")
#             print(f"   - > SQL time    = {qt:.4f} sec")

#         # CASE 2: GROUP BY ⇒ there's a 'per_group' list
#         elif 'per_group' in out:
#             qt = out['QT']
#             print(f"   - > SQL time (grouped) = {qt:.4f} sec")
#             print("   - > per‐group intervals:")
#             for rec in out['per_group']:
#                 grp_key  = rec['group']    # a tuple of group‐column values
#                 count_n  = rec['n']        # group size
#                 (lb_g, ub_g) = rec['interval']
#                 # pretty‐print a tuple of length 1 without the trailing comma:
#                 if isinstance(grp_key, tuple) and len(grp_key) == 1:
#                     grp_key = grp_key[0]
#                 print(f"    • group = {grp_key!r}, n = {count_n}, interval = ({lb_g:.6f}, {ub_g:.6f})")

#         else:
#             # Something unexpected came back
#             print("  [Error: result has no 'lb' or 'per_group']")
