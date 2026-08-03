
import psycopg2
import pandas as pd
from io import StringIO
import csv, re, os, time
import myDataAnalyzer

class DsiributionRecovery:
    def __init__(self, cur):
        self.cur = cur

    def ComputeJointProbablityTable(self,
            table_name: str,
            xo_cols: list[str],
            xm_col: str,
            joint_table: str = "joint_prob"):
        """
        Builds TEMP TABLE joint_prob(
            [xo_cols...],
            xm_col,
            r       -- 1 if xm_col IS NULL, else 0
            cnt     -- integer count
            prob    -- NUMERIC fraction = cnt / N_total
        )
        """
        # 1) total row count
        self.cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        N = self.cur.fetchone()[0]

        # 2) prepare SELECT list and GROUP BY list
        if xo_cols:
            xo_list    = ", ".join(f'"{c}"' for c in xo_cols)
            select_cols = f"{xo_list},\n  \"{xm_col}\""
            group_by    = f"{xo_list}, \"{xm_col}\", r"
        else:
            select_cols = f"\"{xm_col}\""
            group_by    = f"\"{xm_col}\", r"

        r_case = f'CASE WHEN "{xm_col}" IS NULL THEN 1 ELSE 0 END AS r'

        ddl = f"""
DROP TABLE IF EXISTS {joint_table};
CREATE TEMP TABLE {joint_table} AS
SELECT
  {select_cols},
  {r_case},
  COUNT(*)                 AS cnt,
  COUNT(*)::NUMERIC/{N}    AS prob
FROM {table_name}
GROUP BY {group_by};
""".strip()

        # 3) execute
        # print("DEBUG joint DDL:\n", ddl)  # uncomment to inspect generated SQL
        self.cur.execute(ddl)
        self.cur.connection.commit()
        return joint_table

    def mcar_distribution_recovery(self,
            joint_table: str,
            xo_cols: list[str],
            xm_col: str,
            dist_table: str = "mcar_dist"):
        """
        MCAR direct deletion WITH optional GROUP support:
        P(Y=y | xo) = SUM_{xo} P(xo,y,r=0)
        """
        #  drop any old dist table
        self.cur.execute(f"DROP TABLE IF EXISTS {dist_table};")

        #  build SELECT and GROUP BY lists
        if xo_cols:
            xo_list   = ", ".join(f'"{c}"' for c in xo_cols)
            select_cols = f"{xo_list},\n  \"{xm_col}\" AS y"
            group_by    = f"{xo_list}, \"{xm_col}\""
        else:
            select_cols = f"\"{xm_col}\" AS y"
            group_by    = f"\"{xm_col}\""

        #  combine and run SQL
        sql = f"""
    CREATE TEMP TABLE {dist_table} AS
    SELECT
    {select_cols},
    SUM(prob) AS prob
    FROM {joint_table}
    WHERE r = 0
    GROUP BY {group_by};
    """.strip()

        # print(sql)
        self.cur.execute(sql)
        self.cur.connection.commit()
        return dist_table

    def mar_distribution_recovery(self,
            joint_table: str,
            xo_cols: list[str],
            xm_col: str,
            dist_table: str = "mar_dist"):
        """
        MAR direct deletion WITH GROUP support:
        P(Y=y | xo) = SUM_xo [ P(xo,y,r=0)/P(xo,r=0) ] * P(xo)
        """
        # drop old
        self.cur.execute(f"DROP TABLE IF EXISTS {dist_table};")

        xo_list    = ", ".join(f'"{c}"' for c in xo_cols)
        alias_list = ", ".join(f'j."{c}" AS "{c}"' for c in xo_cols)
        join1      = " AND ".join(f'j."{c}" = x."{c}"' for c in xo_cols)
        join2      = " AND ".join(f'py."{c}" = t."{c}"' for c in xo_cols)

        sql = f"""
                CREATE TEMP TABLE {dist_table} AS
                WITH
                px_total AS (
                    SELECT {xo_list}, SUM(prob) AS p_x
                    FROM {joint_table}
                    GROUP BY {xo_list}
                ),
                px_obs AS (
                    SELECT {xo_list}, SUM(prob) AS p_x_obs
                    FROM {joint_table}
                    WHERE r = 0
                    GROUP BY {xo_list}
                ),
                py_given_x AS (
                    SELECT
                    {alias_list},
                    j."{xm_col}"       AS y,
                    j.prob / x.p_x_obs AS p_y_given_x
                    FROM {joint_table} j
                    JOIN px_obs x ON {join1}
                    WHERE j.r = 0
                )
                SELECT
                {xo_list},
                y,
                SUM(p_y_given_x * t.p_x) AS prob
                FROM py_given_x py
                JOIN px_total t ON {join2}
                GROUP BY {xo_list}, y;
                """
        self.cur.execute(sql)
        self.cur.connection.commit()
        return dist_table

    def mar_distribution_recovery(self,
            joint_table: str,
            xo_cols: list[str],
            xm_col: str,
            dist_table: str = "mar_dist"):
        """
        MAR direct deletion WITH GROUP support:
        P(Y=y | xo) = SUM_xo [ P(xo,y,r=0)/P(xo,r=0) ] * P(xo)
        """
        #  drop any existing dist table
        self.cur.execute(f"DROP TABLE IF EXISTS {dist_table};")

        #  prepare column lists
        xo_list     = ", ".join(f'"{c}"' for c in xo_cols)
        # for final SELECT: py."col" AS "col"
        alias_py    = ", ".join(f'py."{c}" AS "{c}"' for c in xo_cols)
        # for final GROUP BY: py."col"
        group_by_py = ", ".join(f'py."{c}"' for c in xo_cols)

        # join conditions for the CTEs
        join1 = " AND ".join(f'j."{c}" = x."{c}"' for c in xo_cols)
        join2 = " AND ".join(f'py."{c}" = t."{c}"' for c in xo_cols)

        #  build and execute the single CREATE AS WITH ... SELECT
        sql = f"""
CREATE TEMP TABLE {dist_table} AS
WITH
  px_total AS (
    SELECT {xo_list}, SUM(prob) AS p_x
    FROM {joint_table}
    GROUP BY {xo_list}
  ),
  px_obs AS (
    SELECT {xo_list}, SUM(prob) AS p_x_obs
    FROM {joint_table}
    WHERE r = 0
    GROUP BY {xo_list}
  ),
  py_given_x AS (
    SELECT
      {", ".join(f'j."{c}" AS "{c}"' for c in xo_cols)},
      j."{xm_col}"       AS y,
      j.prob / x.p_x_obs AS p_y_given_x
    FROM {joint_table} j
    JOIN px_obs x
      ON {join1}
    WHERE j.r = 0
  )
SELECT
  {alias_py},
  py.y             AS y,
  SUM(py.p_y_given_x * t.p_x) AS prob
FROM py_given_x py
JOIN px_total t
  ON {join2}
GROUP BY {group_by_py}, py.y;
"""
        self.cur.execute(sql)
        self.cur.connection.commit()
        return dist_table






# class QueryExecutor:
#     TYPE_MAP = {'int64':'BIGINT','float64':'NUMERIC'}

#     def __init__(self, conn, csv_queries):
#         self.conn        = conn
#         self.cur         = conn.cursor()
#         self.csv_queries = csv_queries
#         self.mar_table_for  = {}
#         self.full_table_for = {}
#         self.groundTR = any(meta.get('complete_csv') for meta in csv_queries.values())

#         # map CSV to table name
#         for meta in csv_queries.values():
#             for p,t in zip(meta['csv'], meta['table']):
#                 self.mar_table_for[os.path.basename(p)] = t
#             if meta.get('complete_csv'):
#                 for p,t in zip(meta['complete_csv'], meta['complete_table']):
#                     self.full_table_for[os.path.basename(p)] = t

#         self.prepareTables()
#         # reuse one recovery object
#         self.recovery = DsiributionRecovery(self.cur)


class DistributionQueryExecutor:
    TYPE_MAP = {'int64':'BIGINT', 'float64':'NUMERIC'}

    def __init__(self, conn, csv_queries):
        self.conn         = conn
        self.cur          = conn.cursor()
        self.csv_queries  = csv_queries

        # Map full CSV path to SQL table name
        self.mar_table_for  = {}  # e.g. "rwDatasets/bank_MAR_5.0.csv" -> "mar_b0"
        self.full_table_for = {}  # e.g. "rwDatasets/bank_complete.csv" -> "full_b0"

        # Detect whether we have ground‐truth tables
        self.groundTR = any(meta.get('complete_csv') for meta in csv_queries.values())

        # Populate both mappings using the full paths
        for meta in csv_queries.values():
            for path, tbl in zip(meta['csv'], meta['table']):
                self.mar_table_for[path] = tbl
            if meta.get('complete_csv'):
                for path, tbl in zip(meta['complete_csv'], meta['complete_table']):
                    self.full_table_for[path] = tbl

        # Create and load all tables
        self.prepareTables()

        # Reuse distribution‐recovery helper
        self.recovery = DsiributionRecovery(self.cur)



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
        # Create in psql - load MCAR/MAR tables
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
        # Create inpasl - load complete tables in case of injected data
        if self.groundTR:
            for path, tbl in self.full_table_for.items():
                schema = self.inferSchema(path)
                cols   = list(schema)
                # Drop/Create the “full” (ground‐truth) table
                col_defs = ',\n  '.join(f'"{c}" {t}' for c,t in schema.items())
                self.cur.execute(f"""
                    DROP TABLE IF EXISTS {tbl};
                    CREATE TABLE {tbl} (
                      {col_defs}
                    );
                    TRUNCATE {tbl};
                """)
                self.conn.commit()
                # Actually load the CSV rows into {tbl}
                self.loadCSV(path, tbl, cols)

    def run_recovery(self):
        results = {}

        for ds_name, meta in self.csv_queries.items():
            results[ds_name] = {}

            csv_list    = meta['csv']               # MAR CSV full paths
            full_list   = meta.get('complete_csv')  # Complete CSV full paths
            cause_meta  = meta.get('Cause', [[]] * len(csv_list))
            if all(isinstance(c, str) for c in cause_meta):
                cause_meta = [[c] for c in cause_meta]

            for qry in meta['queries']:
                QT =0
                modeling_time =0
                # ── parse SELECT AVG(attr) FROM path [WHERE …] [GROUP BY …]
                m = re.match(
                    r'SELECT\s+AVG\((\w+)\)\s+FROM\s+(\S+)'
                    r'(?:\s+WHERE\s+(.+?))?'
                    r'(?:\s+GROUP\s+BY\s+(.+))?$',
                    qry, re.IGNORECASE
                )
                if not m:
                    raise ValueError(f"Cannot parse query: {qry!r}")
                attr, path, where_clause, gb_clause = m.groups()
                attr = attr.lower()

                #  MAR table name
                tbl = self.mar_table_for[path]

                #  WHERE and GROUP BY handling
                where_sql  = f"WHERE {where_clause}" if where_clause else ""
                group_cols = [c.strip().strip('"').lower()
                              for c in gb_clause.split(',')] if gb_clause else []

                #  Find this CSV’s index and its causes
                idx    = csv_list.index(path)
                causes = [c.lower() for c in (cause_meta[idx] or [])]

                xo_cols = group_cols + causes

                #  Joint probability recovery
                start = time.time()
                joint_tbl = self.recovery.ComputeJointProbablityTable(
                    table_name=tbl,
                    xo_cols=xo_cols,
                    xm_col=attr,
                    joint_table="joint_prob"
                )
                modeling_time = (time.time() - start)

                # Marginal recovery (MCAR vs MAR)
                if not causes:
                        start = time.time()
                        dist_tbl = self.recovery.mcar_distribution_recovery(
                        joint_table=joint_tbl,
                        xo_cols=xo_cols,
                        xm_col=attr,
                        dist_table="mcar_dist"
                    )
                        QT+= time.time() - start
                else:
                    start = time.time()
                    dist_tbl = self.recovery.mar_distribution_recovery(
                        joint_table=joint_tbl,
                        xo_cols=xo_cols,
                        xm_col=attr,
                        dist_table="mar_dist"
                    )
                    QT+= time.time() - start

                # Compute estimate(s)
                if group_cols:
                    sel_gb = ", ".join(f'"{c}"' for c in group_cols)
                    avg_sql = f"""
                      SELECT
                        {sel_gb},
                        SUM(y::NUMERIC * prob) AS estimate
                      FROM {dist_tbl}
                      GROUP BY {sel_gb};
                    """
                else:
                        if not causes:

                    # avg_sql = f"""
                    #   SELECT
                    #     SUM(y::NUMERIC * prob) AS estimate
                    #   FROM {dist_tbl};
                    # """
                        # MCAR: normalize by p_obs
                          avg_sql = f"""
                            WITH obs AS (
                            -- overall probability of observing Y
                            SELECT SUM(prob) AS p_obs
                            FROM joint_prob
                            WHERE r = 0
                            )
                            SELECT
                            SUM(y::NUMERIC * prob) / (SELECT p_obs FROM obs) AS estimate
                            FROM {dist_tbl};
                            """.strip()
                        else:
                            # MAR + no GROUP: distribution already normalized
                            avg_sql = f"""
                                    SELECT
                                    SUM(y::NUMERIC * prob) AS estimate
                                    FROM {dist_tbl};
                                    """.strip()


                start = time.time()
                self.cur.execute(avg_sql)
                QT+= time.time() - start
                rows = self.cur.fetchall()

                #  Ground‐truth lookup using the same index
                has_truth = bool(full_list)
                if has_truth:
                    full_path = full_list[idx]           # **use MAR index**
                    full_tbl  = self.full_table_for[full_path]
                    if group_cols:
                        sel_gb = ", ".join(f'"{c}"' for c in group_cols)
                        gt_sql = f"""
                          SELECT
                            {sel_gb},
                            AVG("{attr}"::NUMERIC) AS gt
                          FROM {full_tbl}
                          {where_sql}
                          GROUP BY {sel_gb};
                        """
                        self.cur.execute(gt_sql)
                        gt_rows = {tuple(r[:-1]): r[-1]
                                   for r in self.cur.fetchall()}
                    else:
                        gt_sql = f"""
                          SELECT AVG("{attr}"::NUMERIC)
                          FROM {full_tbl}
                          {where_sql};
                        """
                        self.cur.execute(gt_sql)
                        gt = float(self.cur.fetchone()[0])
                        # gt = (self.cur.fetchone()[0])

                #   results
                if not group_cols:
                    est   = float(rows[0][0]) if rows else None
                    entry = {'estimate': est, 'QT': QT, 'modeling_time':modeling_time }
                    if has_truth:
                        entry['ground_truth'] = gt
                        entry['accuracy']     = (1 - abs(est-gt)/gt) * 100
                    results[ds_name][qry] = entry
                else:
                    # h group sizes from the original MAR table
                    sel_gb     = ", ".join(f'"{c}"' for c in group_cols)
                    count_sql  = f"""
                        SELECT
                        {sel_gb},
                        COUNT(*) AS n
                        FROM {tbl}
                        {where_sql}
                        GROUP BY {sel_gb};
                    """
                    self.cur.execute(count_sql)
                    raw_counts = self.cur.fetchall()
                    # build a lookup: key‐tuple to n
                    counts = {
                        tuple(r[:-1]) if len(group_cols)>1 else (r[0],): r[-1]
                        for r in raw_counts
                    }

                    #  per‐group metrics,
                    per_group = []
                    for grp_vals, est in rows:
                        # normalize grp_vals to a tuple
                        key = (grp_vals,) if not isinstance(grp_vals, (list,tuple)) else tuple(grp_vals)
                        key = tuple(key)
                        if any(v is None for v in key):
                            continue
                        est = float(est)

                        # ground‐truth and accuracy
                        # gt_val = float(gt_rows.get(key)) if has_truth else None ## old
                        raw_gt = gt_rows.get(key) ## new
                        gt_val = float(raw_gt) if (has_truth and raw_gt is not None) else None ## new
                        acc    = (1 - abs(est-gt_val)/gt_val)*100 if has_truth and gt_val else None

                        # lookup group size
                        n = counts.get(key, 0)

                        per_group.append({
                            'group':        key,
                            'n':            n,
                            'estimate':     est,
                            'ground_truth': gt_val,
                            'accuracy':     acc
                        })

                    #  compute macro and micro
                    analyzer = myDataAnalyzer.myDataAnalyzer()
                    macro = analyzer.unweighted_accuracy(per_group)
                    micro = analyzer.weighted_accuracy(per_group)

                    #  store results
                    results[ds_name][qry] = {
                        'per_group':      per_group,
                        'QT':             QT,
                        'modeling_time': modeling_time ,
                        'accuracy_macro': macro,
                        'accuracy_micro': micro
                    }


        return results









if __name__ == "__main__":
    import json
    with open("configs/all_queries_dist.json") as f:
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

    element_to_remove = -1
    conn = psycopg2.connect(
        host="localhost", port=5433, dbname="mydb",
        user="alzamill", password=os.environ.get("PGPASSWORD", "")
    )
    for csv_queries in query_config:
        executor = DistributionQueryExecutor(conn, csv_queries)
        results = executor.run_recovery()
        for datasetName in csv_queries:
            acc_list=[]
            QT_s =[]
            JQT_s =[]
            for q, r in results[datasetName].items():
                print(f"\nQuery: {q}")
                if 'accuracy' in r:
                    print(f" Accuracy           = {r['accuracy']:.2f}%")
                    print(f" Estimate           = {r['estimate']:.2f}")
                    acc_list.append(r['accuracy'])
                if 'accuracy_micro' in r:
                    acc_list.append(r['accuracy_micro'])
                    print(f"GBY Accuracy           = {r['accuracy_micro']:.2f}%")
                print(f" exe   = {r['QT']:.5f}")
                if r['QT'] != -1:
                    QT_s.append(r['QT'])
                print(f" mdl   = {r['modeling_time']:.5f}")
                if r['modeling_time'] != -1:
                    JQT_s.append(r['modeling_time'])

            from statistics import mean

            if not acc_list:
                accMean= -1
            else:
                if element_to_remove in acc_list:
                    acc_list.remove(element_to_remove)
                accMean= mean(acc_list)
            analyzer = myDataAnalyzer.myDataAnalyzer(datasetName=datasetName, output_dir="psql_results",out_file="mar_psql_dist_4.txt")
            analyzer.add_stats(accMean,mean(QT_s),mean(JQT_s))
            analyzer.addNewLine()
        executor.cur.close()
    conn.close()







# ## for test :::


# csv_queries = {

#        "nyc_mcar_5%": {
#     "csv": [
#       "rwDatasets/nyc_MCAR_5.0.csv",
#       "Injected_JoinsData/nyc_mcar5_1.csv",
#       "Injected_JoinsData/nyc_mcar5_2.csv"
#     ],
#     "table": ["mcar5_n0", "mcar5_n1", "mcar5_n2"],
#     "complete_csv": [
#       "rwDatasets/nyc_complete.csv",
#       "Injected_JoinsData/nyc_complete1.csv",
#       "Injected_JoinsData/nyc_complete2.csv"
#     ],
#     "complete_table": ["full_n0", "full_n1", "full_n2"],
#         "Cause": [
#      [],    # for bank_MAR_5.0.csv
#        [],        # for bank_mar5_1.csv
#        []
#     ],
#     "queries": [
#                 #  "Select AVG(passenger_count) From rwDatasets/nyc_MCAR_5.0.csv GROUP BY vendor_id",
#                  "Select AVG(passenger_count) From rwDatasets/nyc_MCAR_5.0.csv",

#     ]
#   },
# # }


# #     "bank_mar_5%": {
# #     "csv": [
# #       "rwDatasets/bank_MAR_5.0.csv",
# #       "Injected_JoinsData/bank_mar5_1.csv",
# #       "Injected_JoinsData/bank_mar5_2.csv"
# #     ],
# #     "table": ["mar_b0", "mar_b1", "mar_b2"],
# #     "complete_csv": [
# #       "rwDatasets/bank_complete.csv",
# #       "Injected_JoinsData/bank_complete1.csv",
# #       "Injected_JoinsData/bank_complete2.csv"
# #     ],
# #     "complete_table": ["full_b0", "full_b1", "full_b2"],
# #     # "Cause": ["campaign","campaign","campaign"],
# #         "Cause": [
# #        ["campaign"],    # for bank_MAR_5.0.csv
# #        ["campaign"],        # for bank_mar5_1.csv
# #        ["campaign"]      # for bank_mar5_2.csv
# #     ],
# #     "queries": [
# #       "SELECT AVG(balance) FROM rwDatasets/bank_MAR_5.0.csv",
# #       "Select AVG(balance) From rwDatasets/bank_MAR_5.0.csv where education = 'primary'",
# #       #"Select AVG(balance) From rwDatasets/bank_MAR_5.0.csv where loan = 'yes'",
# #       "Select AVG(balance) From rwDatasets/bank_MAR_5.0.csv where duration < 100",
# #        "SELECT AVG(balance) FROM rwDatasets/bank_MAR_5.0.csv where loan = 'yes' GROUP BY housing",
# #     ]
# #   },


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

# executor = DistributionQueryExecutor(conn, csv_queries)
# results = executor.run_recovery()
# for datasetName in csv_queries:
#     for q, r in results[datasetName].items():

#         print(f"\nQuery: {q}")
#         # print(f" Estimate           = {r['estimate']:.2f} ")
#     # print(f" Ground Truth       = {r['ground_truth']:.2f}")
#         if 'accuracy' in r:
#             print(f" Estimate           = {r['estimate']:.2f} ")
#             print(f" Accuracy           = {r['accuracy']:.2f}%")
#             acc_list.append(r['accuracy'])
#         if 'accuracy_micro' in r:
#             acc_list.append(r['accuracy_micro'])
#             print(f" Group by query Accuracy           = {r['accuracy_micro']:.2f}%")


#         print(f" QT   = {r['QT']:.5f}")
#         if r['QT'] != -1:
#             QT_s.append(r['QT'])
# executor.cur.close()
# conn.close()
