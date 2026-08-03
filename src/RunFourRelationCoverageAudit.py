#!/usr/bin/env python3
"""Recompute CAMC coverage for supported four-relation set queries."""

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
from RunNonAggregationCoverageFullData import corrected_camc_coverage
from RunSectionComparisons import (
    CONN,
    clean_name,
    direct_coverage,
    is_timeout,
    load_factor_map,
    set_timeout,
    write_csv,
)
from nonAgg_direct import _load_table, run_direct_per_tuple


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--only-queries", default="13,14")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--db-host", default=CONN["host"])
    parser.add_argument("--db-port", type=int, default=CONN["port"])
    parser.add_argument("--db-name", default=CONN["dbname"])
    parser.add_argument("--db-user", default=CONN["user"])
    parser.add_argument("--db-password", default=CONN["password"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as stream:
        workload = json.load(stream)
    dataset = workload["dataset"]
    selected = {
        int(value)
        for value in args.only_queries.split(",")
        if value.strip()
    }
    if not selected or not selected.issubset(set(range(11, 16))):
        raise ValueError("--only-queries must select a subset of Q11--Q15")
    factor_map = {
        str(key).lower(): [str(value).lower() for value in values]
        for key, values in load_factor_map(workload["factor_file"]).items()
    }
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
        for rate in [
            int(value)
            for value in args.rates.split(",")
            if value.strip()
        ]:
            set_timeout(connection, 0)
            rate_config = workload["rates"][str(rate)]
            for path, table in zip(rate_config["csv"], rate_config["table"]):
                _load_table(connection, path, table)
            _load_table(
                connection,
                rate_config["factor_csv"],
                rate_config["factor_table"],
            )
            for path, table in zip(
                rate_config["complete_csv"],
                rate_config["complete_table"],
            ):
                _load_table(connection, path, table)

            prefix = clean_name("%s4cov_%d" % (dataset, rate))
            pred_tables, truth_tables = create_aligned_subsets(
                connection,
                rate_config["table"],
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
                "%s4cov_pred_view_%d" % (dataset, rate),
            )
            truth_view = create_join_view(
                connection,
                truth_tables,
                workload["join_key"],
                "%s4cov_truth_view_%d" % (dataset, rate),
            )
            inspector = NativeMCDB(connection)
            actual_missing = list(inspector.missing_attributes(pred_view))
            missing = required_missing_for_queries(
                rate_config["set_queries"],
                selected,
                actual_missing,
                factor_map,
            )
            partitions = list(workload["partitions"].values())
            camc = FourRelationNativeMCDB(connection, partitions)
            bundle = camc.create_bundle(
                pred_view,
                tuple(missing),
                factor_map,
                args.h,
                seed=float(args.seed % 1000) / 1000.0,
                prefix="camc_%s4cov_%d" % (dataset, rate),
                strict=True,
                n_bins=5,
                factor_table=rate_config["factor_table"],
            )
            if sum(bundle.unresolved_draws.values()):
                raise RuntimeError("The factor sample contains fallback draws")
            mappings: Dict[str, Any] = {}
            projection_started = time.perf_counter()
            for position, attributes in enumerate(partitions):
                projected = project_joint_bundle(
                    connection,
                    bundle,
                    attributes,
                    workload["join_key"],
                    clean_name(
                        "camc_%s4cov_%d_relation_%d"
                        % (dataset, rate, position)
                    )[:55],
                )
                token = rate_config["csv"][position]
                mappings.update(
                    relation_mapping(
                        projected,
                        token,
                        os.path.basename(token),
                        os.path.splitext(os.path.basename(token))[0],
                        rate_config["table"][position],
                    )
                )
            encoding_s = bundle.encoding_s + time.perf_counter() - projection_started

            for query_index, query in enumerate(
                rate_config["set_queries"], 11
            ):
                if query_index not in selected:
                    continue
                flat_pred = flatten_join_query(query, pred_view)
                flat_truth = flatten_join_query(query, truth_view)
                direct = run_direct_per_tuple(
                    connection,
                    flat_pred,
                    pred_view,
                    truth_view,
                    missing,
                    factor_map,
                    return_groups=True,
                )
                if direct.get("error"):
                    raise RuntimeError(direct["error"])
                groups = direct.get("groups", [])
                set_timeout(connection, args.timeout)
                try:
                    summary = camc.evaluate_summary(query, mappings)
                    corrected, predicted_count, oracle_count, overlap_count = (
                        corrected_camc_coverage(
                            summary.summary_rows,
                            groups,
                            args.h,
                        )
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "rate": rate,
                            "query_index": query_index,
                            "method": "CAMC",
                            "corrected_coverage": corrected,
                            "cade_recomputed_coverage": direct_coverage(groups),
                            "predicted_group_count": predicted_count,
                            "oracle_group_count": oracle_count,
                            "overlap_group_count": overlap_count,
                            "factor_sampling_s": bundle.sampling_s,
                            "encoding_s": encoding_s,
                            "factor_fallback_symbols": 0,
                            "h": args.h,
                            "status": "ok",
                        }
                    )
                except Exception as error:
                    connection.rollback()
                    rows.append(
                        {
                            "dataset": dataset,
                            "rate": rate,
                            "query_index": query_index,
                            "method": "CAMC",
                            "corrected_coverage": None,
                            "cade_recomputed_coverage": direct_coverage(groups),
                            "predicted_group_count": None,
                            "oracle_group_count": len(groups),
                            "overlap_group_count": None,
                            "factor_sampling_s": bundle.sampling_s,
                            "encoding_s": encoding_s,
                            "factor_fallback_symbols": 0,
                            "h": args.h,
                            "status": "timeout" if is_timeout(error) else "error",
                            "error": str(error),
                        }
                    )
                write_csv(args.output, rows)
                print(
                    "%s %d%% coverage Q%d complete"
                    % (dataset, rate, query_index),
                    flush=True,
                )
    finally:
        connection.close()
    write_csv(args.output, rows)
    print("Saved %d measurements to %s" % (len(rows), args.output))


if __name__ == "__main__":
    main()
