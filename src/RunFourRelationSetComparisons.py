#!/usr/bin/env python3
"""Compare CAEX, CAMC, and CADE on a five-query multi-relation workload."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import psycopg2

from MCDBPostgresNative import NativeMCDB, parse_query, qident, relation_mapping
from RunSectionComparisons import (
    CONN,
    clean_name,
    columns,
    direct_coverage,
    ensure_loaded,
    execute_truth_set,
    load_factor_map,
    normalize_key,
    output_tvd,
    ratio_of_mean_interval_width,
    run_camc_set,
    set_timeout,
    source_columns,
    timeout_row,
    token_map,
    write_csv,
)
from RunSectionComparisonsFullData import query_attributes
from RunnerSetQueriy import evaluate_query_with_groups
from SetQueryRewriterExecuter import QueryExecutor as CAEXSetExecutor
from nonAgg_direct import _load_table, run_direct_per_tuple


def relation_chain(query: str) -> Tuple[List[str], str]:
    compact = " ".join(query.strip().rstrip(";").split())
    match = re.search(
        r"\bFROM\s+(?P<from>.+?)(?=\s+WHERE\s+|\s+GROUP\s+BY\s+|\s+HAVING\s+|$)",
        compact,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("The query has no FROM clause")
    from_clause = match.group("from")
    first = re.match(r"(?P<table>\S+)", from_clause)
    joins = re.findall(
        r"\bJOIN\s+(?P<table>\S+)\s+USING\s*\(\s*(?P<key>[A-Za-z_]\w*)\s*\)",
        from_clause,
        re.IGNORECASE,
    )
    if not first or not joins:
        raise ValueError("The query is not a USING join chain")
    keys = {key.lower() for _table, key in joins}
    if len(keys) != 1:
        raise ValueError("Every relation must use the same join key")
    return [first.group("table")] + [table for table, _key in joins], keys.pop()


def flatten_join_query(query: str, view: str) -> str:
    return re.sub(
        r"\bFROM\s+.+?(?=\s+WHERE\s+|\s+GROUP\s+BY\s+|\s+HAVING\s+|$)",
        f"FROM {view}",
        " ".join(query.strip().rstrip(";").split()),
        count=1,
        flags=re.IGNORECASE,
    )


class FourRelationNativeMCDB(NativeMCDB):
    """Compile a complete-key join chain without materializing the join."""

    def __init__(self, connection, logical_columns):
        super().__init__(connection)
        self.logical_columns = [set(column.lower() for column in values) for values in logical_columns]

    def compile_compact_summary(self, query, relations):
        tokens, join_key = relation_chain(query)
        if len(tokens) <= 2:
            return super().compile_compact_summary(query, relations)
        spec = parse_query(query)
        bundles = [self._resolve_bundle(token, relations) for token in tokens]
        h_values = {bundle.h for bundle in bundles}
        if len(h_values) != 1:
            raise ValueError("Joined bundles must use the same H")
        if any(bundle.is_random(join_key) for bundle in bundles):
            raise ValueError("The four-relation workload requires a complete join key")
        aliases = [f"r{position}" for position in range(len(bundles))]
        needed = [column for column in self._needed_columns(spec) if column != join_key]
        selected = []
        for column in needed:
            sources = [position for position, values in enumerate(self.logical_columns) if column in values]
            if len(sources) != 1:
                raise ValueError(f"Query column {column} has {len(sources)} sources")
            position = sources[0]
            selected.append(
                "%s AS %s" % (
                    self._value_expression(bundles[position], aliases[position], column, "g.idx"),
                    qident(column),
                )
            )
        from_sql = "%s %s" % (qident(bundles[0].bundle_table), aliases[0])
        for position in range(1, len(bundles)):
            from_sql += " JOIN %s %s ON %s.%s=%s.%s" % (
                qident(bundles[position].bundle_table), aliases[position],
                aliases[0], qident(join_key), aliases[position], qident(join_key),
            )
        suffix = ", " + ", ".join(selected) if selected else ""
        h = h_values.pop()
        world_rows = (
            "world_rows AS (SELECT g.idx%s FROM %s "
            "CROSS JOIN generate_series(1, %d) AS g(idx))"
        ) % (suffix, from_sql, h)
        answer_cte, output_columns = self._answer_cte(spec, "world_rows")
        summary_sql, summary_columns = self._summary_sql(spec, output_columns, h)
        sql = "WITH %s,\n%s %s" % (world_rows, answer_cte, summary_sql)
        return spec, sql, summary_columns


def create_aligned_subsets(conn, source_tables: Sequence[str], truth_tables: Sequence[str],
                           join_key: str, prefix: str, row_limit: int,
                           seed: int) -> Tuple[List[str], List[str]]:
    cursor = conn.cursor()
    key_table = clean_name(prefix + "_keys")[:55]
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(key_table))
    joins = " ".join(
        "JOIN %s p%d ON p0.%s=p%d.%s" % (
            qident(table), position, qident(join_key), position, qident(join_key)
        ) for position, table in enumerate(source_tables[1:], 1)
    )
    truth_joins = " ".join(
        "JOIN %s t%d ON p0.%s=t%d.%s" % (
            qident(table), position, qident(join_key), position, qident(join_key)
        ) for position, table in enumerate(truth_tables)
    )
    key_query = (
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
        "SELECT p0.%s AS join_key FROM %s p0 %s %s "
        "WHERE p0.%s IS NOT NULL"
        % (
            qident(key_table),
            qident(join_key),
            qident(source_tables[0]),
            joins,
            truth_joins,
            qident(join_key),
        )
    )
    if row_limit > 0:
        key_query += (
            " ORDER BY hashtextextended(p0.%s::text, %d) LIMIT %d"
            % (qident(join_key), int(seed), int(row_limit))
        )
    cursor.execute(key_query)
    cursor.execute("CREATE UNIQUE INDEX ON %s (join_key)" % qident(key_table))

    def make(source: str, name: str) -> str:
        projection = ", ".join(
            "s.%s AS %s" % (qident(actual), qident(normalized))
            for actual, normalized in source_columns(conn, source) if normalized != "_rid"
        )
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(name))
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT row_number() OVER (ORDER BY s.%s)::bigint AS %s, %s "
            "FROM %s s JOIN %s k ON s.%s=k.join_key" % (
                qident(name), qident(join_key), qident("_rid"), projection,
                qident(source), qident(key_table), qident(join_key),
            )
        )
        cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (qident(name), qident("_rid")))
        cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (qident(name), qident(join_key)))
        cursor.execute("ANALYZE %s" % qident(name))
        return name

    predicted = [make(table, clean_name(f"{prefix}_pred_{position}")[:55])
                 for position, table in enumerate(source_tables)]
    truth = [make(table, clean_name(f"{prefix}_truth_{position}")[:55])
             for position, table in enumerate(truth_tables)]
    conn.commit()
    return predicted, truth


def create_join_view(conn, tables: Sequence[str], join_key: str, name: str) -> str:
    cursor = conn.cursor()
    selected = [f"r0.{qident('_rid')} AS {qident('_rid')}"]
    used = {"_rid"}
    for position, table in enumerate(tables):
        alias = f"r{position}"
        for column in columns(conn, table):
            if (
                column == "_rid"
                or column.endswith("_nullsym")
                or column in used
            ):
                continue
            selected.append(f"{alias}.{qident(column)} AS {qident(column)}")
            used.add(column)
    source = "%s r0" % qident(tables[0])
    for position, table in enumerate(tables[1:], 1):
        source += " JOIN %s r%d ON r0.%s=r%d.%s" % (
            qident(table), position, qident(join_key), position, qident(join_key),
        )
    cursor.execute("DROP VIEW IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP VIEW %s AS SELECT %s FROM %s" % (
            qident(name), ", ".join(selected), source,
        )
    )
    conn.commit()
    return name


def project_joint_bundle(conn, bundle, attributes: Sequence[str], join_key: str,
                         name: str):
    """Project one jointly sampled bundle into a physical query relation."""
    join_key = join_key.lower()
    attributes = [attribute.lower() for attribute in attributes]
    logical = [join_key] + list(attributes)
    input_columns = set(bundle.columns)
    selected = ["_rid"] + logical
    for attribute in attributes:
        symbol_column = f"{attribute}_nullsym"
        if symbol_column in input_columns:
            selected.append(symbol_column)
        if attribute in bundle.missing_attributes:
            selected.append(bundle.sample_column(attribute))
    selected.append("__present")
    selected = list(dict.fromkeys(selected))
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS SELECT %s FROM %s" % (
            qident(name), ", ".join(qident(column) for column in selected),
            qident(bundle.bundle_table),
        )
    )
    cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (qident(name), qident("_rid")))
    cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (qident(name), qident(join_key)))
    cursor.execute("ANALYZE %s" % qident(name))
    conn.commit()
    kept_columns = tuple(column for column in bundle.columns if column in set(selected))
    kept_types = {column: value for column, value in bundle.column_types.items()
                  if column in set(selected)}
    kept_missing = tuple(attribute for attribute in bundle.missing_attributes
                         if attribute in attributes)
    return dataclasses.replace(
        bundle, base_table=name, bundle_table=name, columns=kept_columns,
        column_types=kept_types, missing_attributes=kept_missing,
        sampling_s=0.0, encoding_s=0.0,
    )


def required_missing_for_queries(
    queries: Sequence[str],
    selected_queries: Sequence[int],
    actual_missing: Sequence[str],
    ordering: Mapping[str, Sequence[str]],
) -> List[str]:
    """Return query-relevant missing attributes and their missing separators."""
    required = set()
    for query_index, query in enumerate(queries, 11):
        if query_index in selected_queries:
            required.update(query_attributes(query))
    missing_set = set(actual_missing)
    active = {attribute for attribute in required if attribute in missing_set}
    pending = list(active)
    while pending:
        attribute = pending.pop()
        for dependency in ordering.get(attribute, ()):
            if dependency in missing_set and dependency not in active:
                active.add(dependency)
                pending.append(dependency)
    return [attribute for attribute in actual_missing if attribute in active]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bank_four_relation_queries.json")
    parser.add_argument("--set-config", default="configs/mnar_set_queries.json")
    parser.add_argument("--factor-file")
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--only-queries", default="11,12,13")
    parser.add_argument("--methods", default="CAEX,CAMC,CADE")
    parser.add_argument("--factor-check-only", action="store_true")
    parser.add_argument("--db-host", default=CONN["host"])
    parser.add_argument("--db-port", type=int, default=CONN["port"])
    parser.add_argument("--db-name", default=CONN["dbname"])
    parser.add_argument("--db-user", default=CONN["user"])
    parser.add_argument("--db-password", default=CONN["password"])
    parser.add_argument("--output")
    args = parser.parse_args()

    workload = json.load(open(args.config))
    dataset = workload["dataset"]
    original = json.load(open(args.set_config))[workload["set_config_key"]]
    factor_file = args.factor_file or workload["factor_file"]
    output = args.output or f"psql_results/{dataset}_semantic_join/{dataset}_semantic_join_results.csv"
    factor_map = load_factor_map(factor_file)
    rows: List[Dict[str, Any]] = []
    selected_queries = {
        int(value)
        for value in args.only_queries.split(",")
        if value.strip()
    }
    if not selected_queries or not selected_queries.issubset(set(range(11, 16))):
        raise ValueError("--only-queries must select a subset of Q11--Q15")
    methods = {
        value.strip().upper()
        for value in args.methods.split(",")
        if value.strip()
    }
    if not methods or not methods.issubset({"CAEX", "CAMC", "CADE"}):
        raise ValueError("--methods must select CAEX, CAMC, and/or CADE")
    if args.factor_check_only:
        methods = {"CAMC"}
    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    try:
        for rate in [int(value) for value in args.rates.split(",") if value.strip()]:
            set_timeout(conn, 0)
            rate_config = workload["rates"][str(rate)]
            block = next(name for name in original if name.endswith(f"_{rate}"))
            metadata = original[block]
            for path, table in zip(rate_config["csv"], rate_config["table"]):
                _load_table(conn, path, table, force=True)
            _load_table(
                conn,
                rate_config["factor_csv"],
                rate_config["factor_table"],
                force=True,
            )
            for path, table in zip(rate_config["complete_csv"], rate_config["complete_table"]):
                _load_table(conn, path, table, force=True)
            prefix = clean_name(f"{dataset}4_{rate}")
            pred_tables, truth_tables = create_aligned_subsets(
                conn, rate_config["table"], rate_config["complete_table"], workload["join_key"],
                prefix, args.rows, args.seed,
            )
            pred_view = create_join_view(conn, pred_tables, workload["join_key"], f"{dataset}4_pred_view_{rate}")
            truth_view = create_join_view(conn, truth_tables, workload["join_key"], f"{dataset}4_truth_view_{rate}")
            ordering = {str(key).lower(): [str(value).lower() for value in values]
                        for key, values in factor_map.items()}
            inspector = NativeMCDB(conn)
            actual_missing = list(inspector.missing_attributes(pred_view))
            missing = required_missing_for_queries(
                rate_config["set_queries"],
                selected_queries,
                actual_missing,
                ordering,
            )
            print(
                f"{dataset} multi-relation {rate}% query-relevant missing attributes: "
                + ", ".join(missing),
                flush=True,
            )
            executor_meta = {block: {"csv": [pred_view], "table": [pred_view],
                                     "complete_csv": [truth_view], "complete_table": [truth_view]}}
            caex = CAEXSetExecutor(conn, executor_meta, skip_prepare=True)
            caex.interval_mode = "delta"
            caex.interval_alpha = 0.05
            caex._ordering_T = ordering
            caex._missing_T = missing

            partition_values = list(workload["partitions"].values())
            camc = FourRelationNativeMCDB(conn, partition_values)
            mappings: Dict[str, Any] = {}
            sampling_s = 0.0
            encoding_s = 0.0
            fallback_symbols = 0
            if "CAMC" in methods:
                joint_bundle = camc.create_bundle(
                    pred_view, tuple(missing), ordering, args.h,
                    seed=float(args.seed % 1000) / 1000.0,
                    prefix=f"camc_{dataset}_semantic_{rate}",
                    strict=True,
                    n_bins=5,
                    factor_table=rate_config["factor_table"],
                )
                sampling_s = joint_bundle.sampling_s
                encoding_s = joint_bundle.encoding_s
                projection_started = time.perf_counter()
                for position, (table, attributes) in enumerate(zip(pred_tables, partition_values)):
                    bundle = project_joint_bundle(
                        conn, joint_bundle, attributes, workload["join_key"],
                        clean_name(f"camc_{dataset}_{rate}_relation_{position}")[:55],
                    )
                    csv_token = rate_config["csv"][position]
                    mappings.update(relation_mapping(
                        bundle, csv_token, os.path.basename(csv_token),
                        os.path.splitext(os.path.basename(csv_token))[0], rate_config["table"][position],
                    ))
                encoding_s += time.perf_counter() - projection_started
                fallback_symbols = sum(joint_bundle.unresolved_draws.values())
            query_count = len(selected_queries)
            sampling_share = sampling_s / query_count
            encoding_share = encoding_s / query_count
            print(f"{dataset} multi-relation {rate}% setup: factor sampling {sampling_s:.3f}s and encoding {encoding_s:.3f}s amortized", flush=True)
            if args.factor_check_only:
                print(f"{dataset} multi-relation {rate}% factor check complete", flush=True)
                continue

            for query_index, query in enumerate(rate_config["set_queries"], 11):
                if query_index not in selected_queries:
                    continue
                flat_pred = flatten_join_query(query, pred_view)
                flat_truth = flatten_join_query(query, truth_view)
                truth_set = execute_truth_set(conn, flat_truth)
                base = {"workload": "set", "dataset": dataset, "block": f"{dataset}_semantic_{rate}",
                        "rate": rate, "query_index": query_index, "query": query, "h": args.h,
                        "relation_count": len(rate_config["csv"]), "row_limit": args.rows,
                        "total_factor_sampling_s": sampling_s,
                        "total_encoding_s": encoding_s,
                        "factor_fallback_symbols": fallback_symbols}
                if "CAEX" in methods:
                    set_timeout(conn, args.timeout)
                    try:
                        metrics, _groups, _truth = evaluate_query_with_groups(
                            caex, flat_pred, flat_truth, ordering, missing,
                        )
                        rows.append({**base, "method": "CAEX", "time_s": metrics["time_pred_s"],
                                     "metric": metrics.get("tv_cond", metrics.get("tv_prob")),
                                     "coverage": metrics.get("interval_coverage"),
                                     "delta_w": metrics.get("normalized_interval_width"),
                                     "status": "ok"})
                    except Exception as error:
                        conn.rollback()
                        rows.append(timeout_row(base, "CAEX", args.timeout, error))
                if "CADE" in methods:
                    set_timeout(conn, args.timeout)
                    try:
                        result = run_direct_per_tuple(
                            conn, flat_pred, pred_view, truth_view, missing, ordering, return_groups=True,
                        )
                        if result.get("error"):
                            raise RuntimeError(result["error"])
                        groups = result.get("groups", [])
                        rows.append({**base, "method": "CADE", "time_s": result["sql_time_s"],
                                     "metric": result.get("tv_prob"), "coverage": direct_coverage(groups),
                                     "delta_w": result.get("delta_w"), "status": "ok"})
                    except Exception as error:
                        conn.rollback()
                        rows.append(timeout_row(base, "CADE", args.timeout, error))
                if "CAMC" in methods:
                    set_timeout(conn, args.timeout)
                    try:
                        elapsed, metric, coverage, count, delta_w = run_camc_set(
                            camc, query, mappings, truth_set, args.h,
                        )
                        rows.append({**base, "method": "CAMC",
                                     "time_s": elapsed + sampling_share + encoding_share,
                                     "query_time_s": elapsed,
                                     "factor_sampling_share_s": sampling_share,
                                     "encoding_share_s": encoding_share,
                                     "metric": metric, "coverage": coverage, "delta_w": delta_w,
                                     "status": "ok", "result_rows": count})
                    except Exception as error:
                        conn.rollback()
                        failed = timeout_row(base, "CAMC", args.timeout, error)
                        failed.update({
                            "time_s": args.timeout + sampling_share + encoding_share,
                            "query_time_s": float(args.timeout),
                            "factor_sampling_share_s": sampling_share,
                            "encoding_share_s": encoding_share,
                        })
                        rows.append(failed)
                write_csv(output, rows)
                print(f"{dataset}_semantic_{rate} set Q{query_index} complete", flush=True)
    finally:
        conn.close()
    write_csv(output, rows)
    print(output)


if __name__ == "__main__":
    main()
