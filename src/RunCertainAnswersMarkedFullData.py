#!/usr/bin/env python3
"""Run Certain Answers on the marked-null full-data CAMC queries."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping

import psycopg2

from RunnerCertainAnswersBagTVD import evaluate_certain
from RunCertainAnswersSemantic15 import SET_GROUPS
from RunSectionComparisonsFullData import (
    CONN,
    assign_marked_null_symbols,
    block_for_rate,
    is_timeout,
    prepare_block,
    read_csv,
    remove_redundant_join_grouping,
    replace_tokens,
    set_timeout,
    write_csv,
)


def timeout_row(
    base: Mapping[str, Any],
    timeout_s: int,
    error: Exception,
) -> Dict[str, Any]:
    return {
        **base,
        "method": "Certain Answers",
        "time_s": float(timeout_s),
        "metric": None,
        "coverage": None,
        "delta_w": None,
        "status": "timeout" if is_timeout(error) else "error",
        "error": str(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--two-config",
        default="configs/semantic_two_relation_set_queries.json",
    )
    parser.add_argument("--multi-config")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--db-host", default=CONN["host"])
    parser.add_argument("--db-port", type=int, default=CONN["port"])
    parser.add_argument("--db-name", default=CONN["dbname"])
    parser.add_argument("--db-user", default=CONN["user"])
    parser.add_argument("--db-password", default=CONN["password"])
    args = parser.parse_args()
    with open(args.two_config) as stream:
        two_config = json.load(stream)

    estimator_rows = [
        row
        for row in read_csv(args.input)
        if str(row.get("dataset", "")).lower() == args.dataset.lower()
        and str(row.get("workload", "")).lower() == "set"
        and str(row.get("method", "")).upper() == "CAMC"
        and str(row.get("null_semantics", "")).lower() == "marked"
        and int(row.get("marked_null_group_size", 0)) == 3
        and int(row.get("row_limit", -1)) == 0
        and 1 <= int(row.get("query_index", -1)) <= args.query_limit
    ]
    estimator_rows.sort(
        key=lambda row: (int(row["rate"]), int(row["query_index"]))
    )
    expected = 3 * args.query_limit
    if len(estimator_rows) != expected:
        raise ValueError(
            "%s has %d matching full-data queries, not %d"
            % (args.dataset, len(estimator_rows), expected)
        )

    rows: List[Dict[str, Any]] = []
    connection = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    connection.autocommit = False
    try:
        prepared_rate = None
        for estimator_row in estimator_rows:
            rate = int(estimator_row["rate"])
            if rate != prepared_rate:
                group = SET_GROUPS[args.dataset.lower()]
                block, metadata = block_for_rate(
                    two_config[group],
                    rate,
                )
                prepared = prepare_block(
                    connection,
                    metadata,
                    args.dataset.lower(),
                    block,
                    "set",
                    args.rows,
                    args.seed,
                    force_reload=True,
                )
                marked_statistics = assign_marked_null_symbols(
                    connection,
                    prepared["pred_tables"],
                    3,
                    42,
                )
                prepared_rate = rate
            query = remove_redundant_join_grouping(estimator_row["query"])
            predicted_query = replace_tokens(query, prepared["pred_map"])
            complete_query = replace_tokens(query, prepared["truth_map"])
            base = {
                "workload": "set",
                "dataset": args.dataset.lower(),
                "block": estimator_row.get("block"),
                "rate": rate,
                "query_index": int(estimator_row["query_index"]),
                "query": query,
                "h": None,
            }
            set_timeout(connection, args.timeout)
            try:
                metrics = evaluate_certain(
                    connection.cursor(),
                    predicted_query,
                    complete_query,
                )
                rows.append(
                    {
                        **base,
                        "method": "Certain Answers",
                        "time_s": metrics["time_pred_s"],
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
                        "row_limit": 0,
                        "null_semantics": "marked",
                        "marked_null_group_size": 3,
                        **marked_statistics,
                    }
                )
                connection.commit()
            except Exception as error:
                connection.rollback()
                rows.append(timeout_row(base, args.timeout, error))
            write_csv(args.output, rows)
            print(
                "%s %d%% full-data Certain Q%d: %s"
                % (
                    args.dataset,
                    int(estimator_row["rate"]),
                    int(estimator_row["query_index"]),
                    rows[-1]["status"],
                ),
                flush=True,
            )
    finally:
        connection.close()
    write_csv(args.output, rows)
    print("Saved %d measurements to %s" % (len(rows), args.output))


if __name__ == "__main__":
    main()
