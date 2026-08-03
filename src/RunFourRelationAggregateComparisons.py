#!/usr/bin/env python3
"""Evaluate Q11--Q15 aggregation queries over four meaningful relations."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import psycopg2

from MCDBPostgresNative import NativeMCDB, relation_mapping
from RunFourRelationSetComparisons import (
    FourRelationNativeMCDB,
    create_aligned_subsets,
    create_join_view,
    flatten_join_query,
    project_joint_bundle,
    required_missing_for_queries,
)
from RunSectionComparisons import (
    CONN,
    clean_name,
    load_factor_map,
    run_analytic_aggregate,
    run_camc_aggregate,
    set_timeout,
    timeout_row,
    write_csv,
)
from nonAgg_direct import _load_table


FACTOR_FILES = {
    "bank": "data/MNAR1Data/bank/bank_agg_mnar1_minimal_separators_Xi.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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
    factor_file = (
        args.factor_file
        or workload.get("aggregate_factor_file")
        or FACTOR_FILES.get(dataset)
    )
    if not factor_file:
        raise ValueError("No factorization file was specified for %s" % dataset)
    ordering = {
        str(key).lower(): [str(value).lower() for value in values]
        for key, values in load_factor_map(factor_file).items()
    }
    output = args.output or (
        "psql_results/%s_four_relation/%s_four_relation_aggregate_results.csv"
        % (dataset, dataset)
    )
    results: List[Dict[str, Any]] = []
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
    connection = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    try:
        for rate in [int(value) for value in args.rates.split(",") if value.strip()]:
            set_timeout(connection, 0)
            rate_config = workload["rates"][str(rate)]
            queries = rate_config.get("aggregate_queries", ())
            if len(queries) != 5:
                raise ValueError(
                    "%s %d%% must define exactly Q11--Q15" % (dataset, rate)
                )
            for path, table in zip(
                rate_config["aggregate_csv"],
                rate_config["aggregate_table"],
            ):
                _load_table(connection, path, table, force=True)
            _load_table(
                connection,
                rate_config["aggregate_factor_csv"],
                rate_config["aggregate_factor_table"],
                force=True,
            )
            for path, table in zip(
                rate_config["complete_csv"], rate_config["complete_table"]
            ):
                _load_table(connection, path, table, force=True)

            prefix = clean_name("%s4agg_%d" % (dataset, rate))
            pred_tables, truth_tables = create_aligned_subsets(
                connection,
                rate_config["aggregate_table"],
                rate_config["complete_table"],
                workload["join_key"],
                prefix,
                args.rows,
                args.seed,
            )
            pred_view = create_join_view(
                connection,
                pred_tables,
                workload["join_key"],
                "%s4agg_pred_view_%d" % (dataset, rate),
            )
            truth_view = create_join_view(
                connection,
                truth_tables,
                workload["join_key"],
                "%s4agg_truth_view_%d" % (dataset, rate),
            )
            inspector = NativeMCDB(connection)
            actual_missing = list(inspector.missing_attributes(pred_view))
            missing = required_missing_for_queries(
                queries,
                selected_queries,
                actual_missing,
                ordering,
            )
            print(
                "%s %d%% query-relevant missing attributes: %s"
                % (dataset, rate, ", ".join(missing)),
                flush=True,
            )

            partitions = list(workload["partitions"].values())
            camc = FourRelationNativeMCDB(connection, partitions)
            mappings: Dict[str, Any] = {}
            sampling_s = 0.0
            encoding_s = 0.0
            fallback_symbols = 0
            if "CAMC" in methods:
                joint_bundle = camc.create_bundle(
                    pred_view,
                    tuple(missing),
                    ordering,
                    args.h,
                    seed=float(args.seed % 1000) / 1000.0,
                    prefix="camc_%s4agg_%d" % (dataset, rate),
                    strict=True,
                    n_bins=5,
                    factor_table=rate_config["aggregate_factor_table"],
                )
                sampling_s = joint_bundle.sampling_s
                encoding_s = joint_bundle.encoding_s
                projection_started = time.perf_counter()
                for position, (table, attributes) in enumerate(
                    zip(pred_tables, partitions)
                ):
                    bundle = project_joint_bundle(
                        connection,
                        joint_bundle,
                        attributes,
                        workload["join_key"],
                        clean_name(
                            "camc_%s4agg_%d_relation_%d"
                            % (dataset, rate, position)
                        )[:55],
                    )
                    token = rate_config["aggregate_csv"][position]
                    mappings.update(
                        relation_mapping(
                        bundle,
                        token,
                        os.path.basename(token),
                        os.path.splitext(os.path.basename(token))[0],
                        rate_config["aggregate_table"][position],
                        )
                    )
                encoding_s += time.perf_counter() - projection_started
                fallback_symbols = sum(joint_bundle.unresolved_draws.values())
            query_count = len(selected_queries)
            sampling_share = sampling_s / query_count
            encoding_share = encoding_s / query_count
            print(
                "%s %d%% setup: factor sampling %.3fs and encoding %.3fs amortized"
                % (dataset, rate, sampling_s, encoding_s),
                flush=True,
            )
            if args.factor_check_only:
                print(
                    "%s %d%% factor check complete" % (dataset, rate),
                    flush=True,
                )
                continue

            for query_index, query in enumerate(queries, 11):
                if query_index not in selected_queries:
                    continue
                flat_pred = flatten_join_query(query, pred_view)
                flat_truth = flatten_join_query(query, truth_view)
                base = {
                    "workload": "aggregate",
                    "dataset": dataset,
                    "block": "%s_four_%d" % (dataset, rate),
                    "rate": rate,
                    "query_index": query_index,
                    "query": query,
                    "h": args.h,
                    "relation_count": 4,
                    "row_limit": args.rows,
                    "total_factor_sampling_s": sampling_s,
                    "total_encoding_s": encoding_s,
                    "factor_fallback_symbols": fallback_symbols,
                }
                for method, direct in (("CAEX", False), ("CADE", True)):
                    if method not in methods:
                        continue
                    set_timeout(connection, args.timeout)
                    try:
                        elapsed, metric, coverage, count, delta_w = (
                            run_analytic_aggregate(
                                connection,
                                flat_pred,
                                pred_view,
                                flat_truth,
                                ordering,
                                missing,
                                direct,
                            )
                        )
                        results.append(
                            {
                                **base,
                                "method": method,
                                "time_s": elapsed,
                                "metric": metric,
                                "coverage": coverage,
                                "delta_w": delta_w,
                                "result_rows": count,
                                "status": "ok",
                            }
                        )
                    except Exception as error:
                        connection.rollback()
                        results.append(
                            timeout_row(base, method, args.timeout, error)
                        )
                if "CAMC" in methods:
                    set_timeout(connection, args.timeout)
                    try:
                        elapsed, metric, coverage, count, delta_w = run_camc_aggregate(
                            camc,
                            query,
                            mappings,
                            connection,
                            flat_truth,
                            args.h,
                        )
                        results.append(
                            {
                                **base,
                                "method": "CAMC",
                                "time_s": elapsed + sampling_share + encoding_share,
                                "query_time_s": elapsed,
                                "factor_sampling_share_s": sampling_share,
                                "encoding_share_s": encoding_share,
                                "metric": metric,
                                "coverage": coverage,
                                "delta_w": delta_w,
                                "result_rows": count,
                                "status": "ok",
                            }
                        )
                    except Exception as error:
                        connection.rollback()
                        failed = timeout_row(base, "CAMC", args.timeout, error)
                        failed.update(
                            {
                                "time_s": args.timeout + sampling_share + encoding_share,
                                "query_time_s": float(args.timeout),
                                "factor_sampling_share_s": sampling_share,
                                "encoding_share_s": encoding_share,
                            }
                        )
                        results.append(failed)
                write_csv(output, results)
                print(
                    "%s %d%% aggregation Q%d complete"
                    % (dataset, rate, query_index),
                    flush=True,
                )
    finally:
        connection.close()
    write_csv(output, results)
    print(output)


if __name__ == "__main__":
    main()
