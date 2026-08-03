#!/usr/bin/env python3
"""Recompute only CAMC coverage against the CAEX/CADE complete-data target."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from decimal import Decimal
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import psycopg2

from RunSectionComparisonsFullData import (
    CONN,
    GROUPS,
    SEPARATOR_FILES,
    block_for_rate,
    build_camc,
    clean_name,
    create_join_view,
    flatten_join_query,
    load_factor_map,
    prepare_block,
    remove_redundant_join_grouping,
    replace_tokens,
    set_timeout,
    wilson_interval,
)
from nonAgg_direct import run_direct_per_tuple


DEFAULT_CONFIG = (
    "data/supported_injected/"
    "configs/mnar_set_queries.json"
)
DEFAULT_OLD_RESULTS = (
    "psql_results/section_comparisons/supported_rerun_20260725/"
    "final/section_comparison_results.csv"
)


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_value(value: Any) -> str:
    if value is None:
        return "__NULL__"
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "__NULL__"
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    return str(value).strip()


def canonical_key(values: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(canonical_value(value) for value in values)


def corrected_camc_coverage(
    summary_rows: Sequence[Sequence[Any]],
    oracle_groups: Sequence[Mapping[str, Any]],
    h: int,
) -> tuple[float, int, int, int]:
    predicted: Dict[Tuple[str, ...], float] = {}
    for row in summary_rows:
        if not row:
            continue
        predicted[canonical_key(row[:-1])] = float(row[-1] or 0.0)
    oracle = {
        canonical_key(group["gk"]): float(group["p_oracle"])
        for group in oracle_groups
    }
    keys = set(predicted) | set(oracle)
    covered = 0
    for key in keys:
        probability = max(0.0, min(1.0, predicted.get(key, 0.0)))
        lower, upper = wilson_interval(round(probability * h), h)
        target = max(0.0, min(1.0, oracle.get(key, 0.0)))
        covered += int(lower <= target <= upper)
    coverage = covered / len(keys) if keys else 1.0
    return coverage, len(predicted), len(oracle), len(set(predicted) & set(oracle))


def stored_lookup(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int, int, str], float]:
    result = {}
    for row in rows:
        if row.get("workload") != "set":
            continue
        result[
            (
                str(row["dataset"]).lower(),
                int(row["rate"]),
                int(row["query_index"]),
                str(row["method"]),
            )
        ] = float(row["coverage"])
    return result


def status_lookup(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int, int], str]:
    result = {}
    for row in rows:
        if row.get("workload") != "set" or row.get("method") != "CAMC":
            continue
        result[
            (
                str(row["dataset"]).lower(),
                int(row["rate"]),
                int(row["query_index"]),
            )
        ] = str(row["status"]).lower()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--query-indices")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--old-results", default=DEFAULT_OLD_RESULTS)
    parser.add_argument("--camc-results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--db-host", default=CONN["host"])
    parser.add_argument("--db-port", type=int, default=CONN["port"])
    parser.add_argument("--db-name", default=CONN["dbname"])
    parser.add_argument("--db-user", default=CONN["user"])
    parser.add_argument("--db-password", default=CONN["password"])
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if dataset not in GROUPS:
        raise ValueError("Unknown dataset %s" % dataset)
    rates = [int(value) for value in args.rates.split(",") if value.strip()]
    query_indices = (
        {
            int(value)
            for value in args.query_indices.split(",")
            if value.strip()
        }
        if args.query_indices
        else None
    )
    with open(args.config) as handle:
        config = json.load(handle)
    old = stored_lookup(read_rows(args.old_results))
    camc_status = (
        status_lookup(read_rows(args.camc_results))
        if args.camc_results
        else {}
    )
    set_group, _aggregate_group = GROUPS[dataset]
    factor_map = load_factor_map(SEPARATOR_FILES[dataset])
    output_rows: list[dict[str, Any]] = []

    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    conn.autocommit = False
    try:
        for rate in rates:
            block, meta = block_for_rate(config[set_group], rate)
            set_timeout(conn, 0)
            prepared = prepare_block(
                conn, meta, dataset, block, "set", args.rows, args.seed
            )
            camc_engine, camc_mappings, _sampling_s, _encoding_s, fallback = build_camc(
                conn, meta, prepared, "set", factor_map, args.h, args.seed
            )
            if fallback:
                raise RuntimeError(
                    "%s at %d%% has %d unresolved factor draws"
                    % (dataset, rate, fallback)
                )

            combined_ordering = {
                str(key).lower(): [str(value).lower() for value in values]
                for key, values in (meta.get("ordering_T") or {}).items()
            }
            combined_ordering.update(
                {
                    str(key).lower(): [str(value).lower() for value in values]
                    for key, values in (meta.get("ordering_S") or {}).items()
                }
            )
            combined_missing = [
                str(value).lower()
                for value in (meta.get("missing_attrs_T") or [])
            ]
            combined_missing.extend(
                str(value).lower()
                for value in (meta.get("missing_attrs_S") or [])
            )
            join_pred_view = create_join_view(
                conn,
                prepared["pred_tables"][1],
                prepared["pred_tables"][2],
                prepared["join_key"],
                clean_name("coverage_set_" + block),
            )
            join_truth_view = create_join_view(
                conn,
                prepared["full_tables"][1],
                prepared["full_tables"][2],
                prepared["join_key"],
                clean_name("coverage_set_truth_" + block),
            )

            for query_index, configured_query in enumerate(
                list(meta["queries"])[: args.query_limit], 1
            ):
                if query_indices is not None and query_index not in query_indices:
                    continue
                original_status = camc_status.get(
                    (dataset, rate, query_index),
                    "ok",
                )
                if original_status != "ok":
                    output_rows.append(
                        {
                            "dataset": dataset,
                            "rate": rate,
                            "query_index": query_index,
                            "method": "CAMC",
                            "old_coverage": None,
                            "corrected_coverage": None,
                            "cade_stored_coverage": old[
                                (dataset, rate, query_index, "CADE")
                            ],
                            "cade_recomputed_coverage": None,
                            "predicted_group_count": None,
                            "oracle_group_count": None,
                            "overlap_group_count": None,
                            "h": args.h,
                            "status": original_status,
                            "error": "Skipped because the recorded CAMC query did not complete",
                        }
                    )
                    write_rows(args.output, output_rows)
                    print(
                        "%s %d%% coverage Q%d: skipped recorded %s"
                        % (dataset, rate, query_index, original_status),
                        flush=True,
                    )
                    continue
                query = remove_redundant_join_grouping(configured_query)
                is_join = " JOIN " in query.upper()
                if is_join:
                    direct_query = flatten_join_query(query, join_pred_view)
                    direct_result = run_direct_per_tuple(
                        conn,
                        direct_query,
                        join_pred_view,
                        join_truth_view,
                        combined_missing,
                        combined_ordering,
                        return_groups=True,
                    )
                else:
                    direct_query = replace_tokens(query, prepared["pred_map"])
                    direct_result = run_direct_per_tuple(
                        conn,
                        direct_query,
                        prepared["pred_tables"][0],
                        prepared["full_tables"][0],
                        [
                            str(value).lower()
                            for value in meta.get("missing_attrs_single", [])
                        ],
                        {
                            str(key).lower(): [
                                str(value).lower() for value in values
                            ]
                            for key, values in (
                                meta.get("ordering_single") or {}
                            ).items()
                        },
                        return_groups=True,
                    )
                if direct_result.get("error"):
                    raise RuntimeError(
                        "%s %d%% Q%d oracle failed: %s"
                        % (
                            dataset,
                            rate,
                            query_index,
                            direct_result["error"],
                        )
                    )
                oracle_groups = direct_result.get("groups", [])
                cade_recomputed = sum(
                    int(
                        max(
                            0.0,
                            float(group["p_hat"])
                            - float(group["ci_half"]),
                        )
                        <= float(group["p_oracle"])
                        <= min(
                            1.0,
                            float(group["p_hat"])
                            + float(group["ci_half"]),
                        )
                    )
                    for group in oracle_groups
                ) / len(oracle_groups)
                cade_stored = old[(dataset, rate, query_index, "CADE")]
                if abs(cade_recomputed - cade_stored) > 1e-10:
                    raise RuntimeError(
                        "%s %d%% Q%d changed the CADE target: %.12f != %.12f"
                        % (
                            dataset,
                            rate,
                            query_index,
                            cade_recomputed,
                            cade_stored,
                        )
                    )

                set_timeout(conn, args.timeout)
                try:
                    summary = camc_engine.evaluate_summary(query, camc_mappings)
                except Exception as error:
                    conn.rollback()
                    output_rows.append(
                        {
                            "dataset": dataset,
                            "rate": rate,
                            "query_index": query_index,
                            "method": "CAMC",
                            "old_coverage": old.get(
                                (dataset, rate, query_index, "CAMC")
                            ),
                            "corrected_coverage": None,
                            "cade_stored_coverage": cade_stored,
                            "cade_recomputed_coverage": cade_recomputed,
                            "predicted_group_count": None,
                            "oracle_group_count": len(oracle_groups),
                            "overlap_group_count": None,
                            "h": args.h,
                            "status": (
                                "timeout"
                                if "timeout" in str(error).lower()
                                or "canceling statement" in str(error).lower()
                                else "error"
                            ),
                            "error": str(error),
                        }
                    )
                    write_rows(args.output, output_rows)
                    print(
                        "%s %d%% coverage Q%d: %s"
                        % (
                            dataset,
                            rate,
                            query_index,
                            output_rows[-1]["status"],
                        ),
                        flush=True,
                    )
                    continue
                corrected, predicted_count, oracle_count, overlap_count = (
                    corrected_camc_coverage(
                        summary.summary_rows, oracle_groups, args.h
                    )
                )
                output_rows.append(
                    {
                        "dataset": dataset,
                        "rate": rate,
                        "query_index": query_index,
                        "method": "CAMC",
                        "old_coverage": old.get(
                            (dataset, rate, query_index, "CAMC")
                        ),
                        "corrected_coverage": corrected,
                        "cade_stored_coverage": cade_stored,
                        "cade_recomputed_coverage": cade_recomputed,
                        "predicted_group_count": predicted_count,
                        "oracle_group_count": oracle_count,
                        "overlap_group_count": overlap_count,
                        "h": args.h,
                        "status": "ok",
                    }
                )
                write_rows(args.output, output_rows)
                print(
                    "%s %d%% coverage Q%d complete"
                    % (dataset, rate, query_index),
                    flush=True,
                )
    finally:
        conn.close()
    write_rows(args.output, output_rows)
    print("Saved %d coverage measurements to %s" % (len(output_rows), args.output))


if __name__ == "__main__":
    main()
