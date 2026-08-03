#!/usr/bin/env python3
"""Run the certain-answers baseline on the semantic Q1--Q15 workload."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Sequence

import psycopg2

from RunnerCertainAnswersBagTVD import evaluate_certain
from RunSectionComparisonsFullData import (
    CONN,
    block_for_rate,
    clean_name,
    columns,
    ensure_loaded,
    is_timeout,
    parse_query,
    prepare_block,
    qident,
    read_csv,
    replace_tokens,
    set_timeout,
    source_columns,
    token_map,
    write_csv,
)


SET_GROUPS = {
    "bank": "bank_manr1_set",
    "nyc": "nyc_manr1_set",
    "bitcoin": "bit_manr1_set",
}


def relation_maps(csvs: Sequence[str], tables: Sequence[str],
                  complete_tables: Sequence[str]):
    pred = token_map(csvs, tables, tables)
    truth = token_map(csvs, tables, complete_tables)
    return pred, truth


def prepare_relations(conn, csvs: Sequence[str], tables: Sequence[str],
                      complete_csvs: Sequence[str],
                      complete_tables: Sequence[str]):
    for path, table in zip(csvs, tables):
        ensure_loaded(conn, path, table)
    for path, table in zip(complete_csvs, complete_tables):
        ensure_loaded(conn, path, table)


def ensure_loaded_exact(conn, csv_path: str, table: str) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT to_regclass(%s)", (qident(table),))
    exists = cursor.fetchone()[0] is not None
    count = 0
    if exists:
        cursor.execute("SELECT count(*) FROM %s" % qident(table))
        count = int(cursor.fetchone()[0])
    conn.commit()
    if count == 0:
        ensure_loaded(conn, csv_path, table, force=True)


def create_full_alias(conn, source: str, name: str) -> str:
    cursor = conn.cursor()
    cursor.execute("DROP VIEW IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP VIEW %s AS SELECT * FROM %s"
        % (qident(name), qident(source))
    )
    conn.commit()
    return name


def create_full_deduplicated_right(conn, left: str, right: str,
                                   key: str, name: str) -> str:
    left_names = set(columns(conn, left))
    selected = []
    for actual, normalized in source_columns(conn, right):
        if normalized == key.lower() or normalized not in left_names:
            selected.append(
                "r.%s AS %s" % (qident(actual), qident(normalized))
            )
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
        "SELECT row_number() OVER (ORDER BY r.ctid)::bigint AS %s, %s "
        "FROM %s r"
        % (
            qident(name),
            qident("_rid"),
            ", ".join(selected),
            qident(right),
        )
    )
    cursor.execute(
        "CREATE UNIQUE INDEX ON %s (%s)"
        % (qident(name), qident("_rid"))
    )
    cursor.execute(
        "CREATE INDEX ON %s (%s)"
        % (qident(name), qident(key.lower()))
    )
    cursor.execute("ANALYZE %s" % qident(name))
    conn.commit()
    return name


def prepare_full_two_relations(conn, metadata: Mapping[str, Any],
                               dataset: str, block: str):
    csvs = list(metadata["csv"])
    tables = list(metadata["table"])
    complete_csvs = list(metadata["complete_csv"])
    complete_tables = list(metadata["complete_table"])
    for path, table in zip(csvs, tables):
        ensure_loaded_exact(conn, path, table)
    for path, table in zip(complete_csvs, complete_tables):
        ensure_loaded_exact(conn, path, table)
    join_spec = next(
        (
            parse_query(query)
            for query in metadata["queries"]
            if " JOIN " in query.upper()
        ),
        None,
    )
    if join_spec is None or not join_spec.join_column:
        raise ValueError(f"{block} has no join query")
    prefix = clean_name(f"certain_{dataset}_{block}")
    predicted_single = create_full_alias(
        conn,
        tables[0],
        clean_name(prefix + "_pred0")[:55],
    )
    predicted_left = create_full_alias(
        conn,
        tables[1],
        clean_name(prefix + "_pred1")[:55],
    )
    complete_single = create_full_alias(
        conn,
        complete_tables[0],
        clean_name(prefix + "_truth0")[:55],
    )
    complete_left = create_full_alias(
        conn,
        complete_tables[1],
        clean_name(prefix + "_truth1")[:55],
    )
    predicted_right = create_full_deduplicated_right(
        conn,
        tables[1],
        tables[2],
        join_spec.join_column,
        clean_name(prefix + "_pred2_unique")[:55],
    )
    complete_right = create_full_deduplicated_right(
        conn,
        complete_tables[1],
        complete_tables[2],
        join_spec.join_column,
        clean_name(prefix + "_truth2_unique")[:55],
    )
    predicted_tables = [
        predicted_single,
        predicted_left,
        predicted_right,
    ]
    truth_tables = [
        complete_single,
        complete_left,
        complete_right,
    ]
    return {
        "pred_map": token_map(csvs, tables, predicted_tables),
        "truth_map": token_map(csvs, tables, truth_tables),
    }


def workload_for_rate(conn, dataset: str, rate: int,
                      two_config: Mapping[str, Any],
                      multi_config: Mapping[str, Any],
                      row_limit: int,
                      seed: int):
    group = SET_GROUPS[dataset]
    block, two = block_for_rate(two_config[group], rate)
    multi = multi_config["rates"][str(rate)]

    if row_limit <= 0:
        prepared = prepare_full_two_relations(
            conn,
            two,
            dataset,
            block,
        )
    else:
        prepared = prepare_block(
            conn,
            two,
            dataset,
            block,
            "set",
            row_limit,
            seed,
        )
    two_pred_map = prepared["pred_map"]
    two_truth_map = prepared["truth_map"]

    multi_csvs = list(multi["csv"])
    multi_tables = list(multi["table"])
    multi_complete_csvs = list(multi["complete_csv"])
    multi_complete_tables = list(multi["complete_table"])
    prepare_relations(
        conn, multi_csvs, multi_tables,
        multi_complete_csvs, multi_complete_tables,
    )
    multi_pred_map, multi_truth_map = relation_maps(
        multi_csvs, multi_tables, multi_complete_tables
    )

    queries = []
    for index, query in enumerate(two["queries"], 1):
        queries.append((
            index,
            replace_tokens(query, two_pred_map),
            replace_tokens(query, two_truth_map),
        ))
    for index, query in enumerate(multi["set_queries"], 11):
        queries.append((
            index,
            replace_tokens(query, multi_pred_map),
            replace_tokens(query, multi_truth_map),
        ))
    if [index for index, _pred, _truth in queries] != list(range(1, 16)):
        raise ValueError(f"{dataset} at {rate}% does not define Q1--Q15")
    return block, queries


def timeout_row(base: Mapping[str, Any], timeout_s: int,
                error: Exception) -> Dict[str, Any]:
    return {
        **base,
        "method": "Certain Answers",
        "time_s": float(timeout_s),
        "metric": None,
        "coverage": None,
        "delta_w": None,
        "certain_answer_time_s": None,
        "original_sql_time_s": None,
        "ground_truth_time_s": None,
        "status": "timeout" if is_timeout(error) else "error",
        "error": str(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--two-config",
        default="configs/semantic_two_relation_set_queries.json",
    )
    parser.add_argument(
        "--multi-config",
        help="Dataset-specific multi-relation configuration",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only-queries", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--db-host", default=CONN["host"])
    parser.add_argument("--db-port", type=int, default=CONN["port"])
    parser.add_argument("--db-name", default=CONN["dbname"])
    parser.add_argument("--db-user", default=CONN["user"])
    parser.add_argument("--db-password", default=CONN["password"])
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset not in SET_GROUPS:
        raise ValueError(f"Unknown dataset: {dataset}")
    multi_path = args.multi_config or f"configs/{dataset}_semantic_join_queries.json"
    with open(args.two_config) as handle:
        two_config = json.load(handle)
    with open(multi_path) as handle:
        multi_config = json.load(handle)
    if multi_config["dataset"] != dataset:
        raise ValueError(
            f"{multi_path} is for {multi_config['dataset']}, not {dataset}"
        )

    selected_queries = {
        int(value) for value in args.only_queries.split(",") if value.strip()
    }
    rows: List[Dict[str, Any]] = read_csv(args.output) if args.resume else []
    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    conn.autocommit = False
    try:
        for rate in [
            int(value) for value in args.rates.split(",") if value.strip()
        ]:
            block, queries = workload_for_rate(
                conn,
                dataset,
                rate,
                two_config,
                multi_config,
                args.rows,
                args.seed,
            )
            for query_index, pred_query, truth_query in queries:
                if selected_queries and query_index not in selected_queries:
                    continue
                rows = [
                    row for row in rows
                    if not (
                        str(row.get("dataset", "")).lower() == dataset
                        and int(row.get("rate", -1)) == rate
                        and int(row.get("query_index", -1)) == query_index
                    )
                ]
                base = {
                    "workload": "set",
                    "dataset": dataset,
                    "block": block,
                    "rate": rate,
                    "query_index": query_index,
                    "query": pred_query,
                    "h": None,
                }
                set_timeout(conn, args.timeout)
                try:
                    metrics = evaluate_certain(
                        conn.cursor(), pred_query, truth_query
                    )
                    rows.append({
                        **base,
                        "method": "Certain Answers",
                        "time_s": metrics["time_pred_s"],
                        "certain_answer_time_s": metrics["time_pred_s"],
                        "original_sql_time_s": metrics["time_original_sql_s"],
                        "ground_truth_time_s": metrics["time_ground_truth_s"],
                        "metric": metrics["tv_set"],
                        "uniform_set_tvd": metrics["tv_set"],
                        "bag_frequency_tvd": metrics["tv_bag"],
                        "coverage_aware_tvd": metrics["coverage_aware_tvd"],
                        "coverage": metrics["coverage"],
                        "delta_w": None,
                        "status": "ok",
                        "result_rows": metrics["size_pred"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                    })
                    conn.commit()
                except Exception as error:
                    conn.rollback()
                    rows.append(timeout_row(base, args.timeout, error))
                write_csv(args.output, rows)
                print(
                    f"{dataset} {rate}% certain Q{query_index} complete",
                    flush=True,
                )
    finally:
        conn.close()
    write_csv(args.output, rows)
    print(f"Saved {len(rows)} measurements to {args.output}")


if __name__ == "__main__":
    main()
