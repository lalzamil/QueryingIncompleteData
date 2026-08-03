#!/usr/bin/env python3
"""Run marked-null Q13--Q14 over four normalized relations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import psycopg2

from MCDBPostgresNative import NativeMCDB, qident, relation_mapping
from RunFourRelationAggregateComparisons import FACTOR_FILES
from RunFourRelationSetComparisons import (
    FourRelationNativeMCDB,
    create_aligned_subsets,
    flatten_join_query,
    project_joint_bundle,
    required_missing_for_queries,
)
from RunLikeApxMarkedFullData import (
    TimedRewrittenLikeApx,
    aggregate_quality,
    set_quality,
)
from RunSectionComparisons import (
    CONN,
    clean_name,
    columns,
    direct_coverage,
    execute_truth_set,
    load_factor_map,
    run_analytic_aggregate,
    run_camc_aggregate,
    run_camc_set,
    set_timeout,
    timeout_row,
    write_csv,
)
from RunSectionComparisonsFullData import assign_marked_null_symbols
from RunnerCertainAnswersBagTVD import evaluate_certain
from nonAgg_direct import _load_table, run_direct_per_tuple


SELECTED_QUERIES = (13, 14)


def read_reference(
    path: Path,
    rates: Sequence[int],
) -> Dict[int, Dict[int, str]]:
    result: Dict[int, Dict[int, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            query_index = int(row["query_index"])
            if query_index not in SELECTED_QUERIES:
                continue
            rate = int(row["rate"])
            if rate not in rates:
                continue
            result.setdefault(rate, {})[query_index] = row["query"]
    expected = {
        (rate, query_index)
        for rate in rates
        for query_index in SELECTED_QUERIES
    }
    actual = {
        (rate, query_index)
        for rate, queries in result.items()
        for query_index in queries
    }
    if actual != expected:
        raise ValueError(f"{path} does not contain Q13--Q14 at all rates")
    return result


def create_marked_join_view(
    connection,
    tables: Sequence[str],
    join_key: str,
    name: str,
) -> str:
    cursor = connection.cursor()
    selected = [f"r0.{qident('_rid')} AS {qident('_rid')}"]
    used = {"_rid"}
    for position, table in enumerate(tables):
        alias = f"r{position}"
        for column in columns(connection, table):
            if column == "_rid" or column in used:
                continue
            selected.append(f"{alias}.{qident(column)} AS {qident(column)}")
            used.add(column)
    source = f"{qident(tables[0])} r0"
    for position, table in enumerate(tables[1:], 1):
        source += " JOIN %s r%d ON r0.%s=r%d.%s" % (
            qident(table),
            position,
            qident(join_key),
            position,
            qident(join_key),
        )
    cursor.execute("DROP VIEW IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP VIEW %s AS SELECT %s FROM %s"
        % (qident(name), ", ".join(selected), source)
    )
    connection.commit()
    return name


def decorated_base(
    workload: str,
    dataset: str,
    rate: int,
    query_index: int,
    query: str,
    h: int,
    marked_statistics: Mapping[str, Any],
    sampling_s: float,
    encoding_s: float,
    fallback_symbols: int,
) -> Dict[str, Any]:
    return {
        "workload": workload,
        "dataset": dataset,
        "block": f"{dataset}_marked_four_{rate}",
        "rate": rate,
        "query_index": query_index,
        "query": query,
        "h": h,
        "relation_count": 4,
        "row_limit": 0,
        "total_factor_sampling_s": sampling_s,
        "total_encoding_s": encoding_s,
        "factor_fallback_symbols": fallback_symbols,
        "null_semantics": "marked",
        "marked_null_group_size": 3,
        **marked_statistics,
    }


def timed_out_row(
    base: Mapping[str, Any],
    method: str,
    timeout_s: int,
    error: Exception,
    time_s: float | None = None,
) -> Dict[str, Any]:
    row = timeout_row(base, method, timeout_s, error)
    if time_s is not None:
        row["time_s"] = time_s
    return row


def prepare_rate(
    connection,
    workload: Mapping[str, Any],
    dataset: str,
    rate: int,
    kind: str,
    seed: int,
) -> Dict[str, Any]:
    rate_config = workload["rates"][str(rate)]
    if kind == "set":
        csv_key = "csv"
        table_key = "table"
        factor_csv_key = "factor_csv"
        factor_table_key = "factor_table"
    else:
        csv_key = "aggregate_csv"
        table_key = "aggregate_table"
        factor_csv_key = "aggregate_factor_csv"
        factor_table_key = "aggregate_factor_table"
    for path, table in zip(rate_config[csv_key], rate_config[table_key]):
        _load_table(connection, path, table, force=True)
    _load_table(
        connection,
        rate_config[factor_csv_key],
        rate_config[factor_table_key],
        force=True,
    )
    for path, table in zip(
        rate_config["complete_csv"], rate_config["complete_table"]
    ):
        _load_table(connection, path, table, force=True)
    prefix = clean_name(f"{dataset}_marked4_{kind}_{rate}")
    predicted, truth = create_aligned_subsets(
        connection,
        rate_config[table_key],
        rate_config["complete_table"],
        workload["join_key"],
        prefix,
        0,
        seed,
    )
    marked_statistics = assign_marked_null_symbols(
        connection,
        predicted,
        3,
        42,
    )
    predicted_view = create_marked_join_view(
        connection,
        predicted,
        workload["join_key"],
        clean_name(f"{prefix}_pred_view")[:55],
    )
    truth_view = create_marked_join_view(
        connection,
        truth,
        workload["join_key"],
        clean_name(f"{prefix}_truth_view")[:55],
    )
    return {
        "rate_config": rate_config,
        "predicted": predicted,
        "truth": truth,
        "predicted_view": predicted_view,
        "truth_view": truth_view,
        "factor_table": rate_config[factor_table_key],
        "marked_statistics": marked_statistics,
        "csvs": rate_config[csv_key],
    }


def build_sampling(
    connection,
    workload: Mapping[str, Any],
    dataset: str,
    rate: int,
    kind: str,
    prepared: Mapping[str, Any],
    ordering: Mapping[str, Sequence[str]],
    queries: Sequence[str],
    h: int,
    seed: int,
    strict: bool,
) -> Dict[str, Any]:
    inspector = NativeMCDB(connection)
    actual_missing = list(inspector.missing_attributes(prepared["predicted_view"]))
    missing = required_missing_for_queries(
        queries,
        SELECTED_QUERIES,
        actual_missing,
        ordering,
    )
    partitions = list(workload["partitions"].values())
    engine = FourRelationNativeMCDB(connection, partitions)
    bundle = engine.create_bundle(
        prepared["predicted_view"],
        tuple(missing),
        ordering,
        h,
        seed=float(seed % 1000) / 1000.0,
        prefix=f"marked_{dataset}_{kind}_{rate}",
        strict=strict,
        n_bins=5,
        factor_table=prepared["factor_table"],
    )
    mappings: Dict[str, Any] = {}
    projection_started = time.perf_counter()
    for position, (table, attributes) in enumerate(
        zip(prepared["predicted"], partitions)
    ):
        projected = project_joint_bundle(
            connection,
            bundle,
            attributes,
            workload["join_key"],
            clean_name(f"marked_{dataset}_{kind}_{rate}_r{position}")[:55],
        )
        token = prepared["csvs"][position]
        mappings.update(
            relation_mapping(
                projected,
                token,
                os.path.basename(token),
                os.path.splitext(os.path.basename(token))[0],
            )
        )
    projection_s = time.perf_counter() - projection_started
    return {
        "engine": engine,
        "bundle": bundle,
        "mappings": mappings,
        "sampling_s": bundle.sampling_s,
        "encoding_s": bundle.encoding_s + projection_s,
        "fallback_symbols": sum(bundle.unresolved_draws.values()),
        "missing": missing,
    }


def query_missing_attributes(
    connection,
    prepared: Mapping[str, Any],
    ordering: Mapping[str, Sequence[str]],
    configured: Sequence[str],
) -> List[str]:
    actual_missing = list(
        NativeMCDB(connection).missing_attributes(prepared["predicted_view"])
    )
    return required_missing_for_queries(
        configured,
        SELECTED_QUERIES,
        actual_missing,
        ordering,
    )


def run_set(
    connection,
    workload: Mapping[str, Any],
    dataset: str,
    reference: Mapping[int, Mapping[int, str]],
    method: str,
    h: int,
    timeout_s: int,
    seed: int,
    strict: bool,
    rates: Sequence[int],
    output: Path,
) -> None:
    ordering = {
        str(key).lower(): [str(value).lower() for value in values]
        for key, values in load_factor_map(workload["factor_file"]).items()
    }
    rows: List[Dict[str, Any]] = []
    for rate in rates:
        set_timeout(connection, 0)
        prepared = prepare_rate(connection, workload, dataset, rate, "set", seed)
        configured = list(workload["rates"][str(rate)]["set_queries"])
        for query_index in SELECTED_QUERIES:
            configured[query_index - 11] = reference[rate][query_index]
        relevant_missing = query_missing_attributes(
            connection, prepared, ordering, configured
        )
        sampling = None
        if method in {"CAMC", "QE"}:
            sampling = build_sampling(
                connection,
                workload,
                dataset,
                rate,
                "set",
                prepared,
                ordering,
                configured,
                h,
                seed,
                strict,
            )
        sampling_s = sampling["sampling_s"] if sampling else 0.0
        encoding_s = sampling["encoding_s"] if sampling else 0.0
        fallback = sampling["fallback_symbols"] if sampling else 0
        sampling_share = sampling_s / len(SELECTED_QUERIES)
        encoding_share = encoding_s / len(SELECTED_QUERIES)
        for query_index in SELECTED_QUERIES:
            query = reference[rate][query_index]
            flat_predicted = flatten_join_query(query, prepared["predicted_view"])
            flat_truth = flatten_join_query(query, prepared["truth_view"])
            base = decorated_base(
                "set",
                dataset,
                rate,
                query_index,
                query,
                h,
                prepared["marked_statistics"],
                sampling_s,
                encoding_s if method == "CAMC" else 0.0,
                fallback,
            )
            set_timeout(connection, timeout_s)
            try:
                if method == "CADE":
                    result = run_direct_per_tuple(
                        connection,
                        flat_predicted,
                        prepared["predicted_view"],
                        prepared["truth_view"],
                        relevant_missing,
                        ordering,
                        return_groups=True,
                    )
                    if result.get("error"):
                        raise RuntimeError(result["error"])
                    row = {
                        **base,
                        "method": method,
                        "time_s": result["sql_time_s"],
                        "metric": result.get("tv_prob"),
                        "coverage": direct_coverage(result.get("groups", [])),
                        "delta_w": result.get("delta_w"),
                        "status": "ok",
                    }
                elif method == "Certain Answers":
                    metrics = evaluate_certain(
                        connection.cursor(), flat_predicted, flat_truth
                    )
                    row = {
                        **base,
                        "method": method,
                        "time_s": metrics["time_pred_s"],
                        "metric": metrics["coverage_aware_tvd"],
                        "uniform_set_tvd": metrics["tv_set"],
                        "bag_frequency_tvd": metrics["tv_bag"],
                        "coverage_aware_tvd": metrics["coverage_aware_tvd"],
                        "coverage": metrics["coverage"],
                        "delta_w": None,
                        "status": "ok",
                        "result_rows": metrics["size_pred"],
                    }
                elif method == "CAMC":
                    truth = execute_truth_set(connection, flat_truth)
                    elapsed, metric, coverage, count, delta_w = run_camc_set(
                        sampling["engine"],
                        query,
                        sampling["mappings"],
                        truth,
                        h,
                    )
                    row = {
                        **base,
                        "method": method,
                        "time_s": elapsed + sampling_share + encoding_share,
                        "query_time_s": elapsed,
                        "factor_sampling_share_s": sampling_share,
                        "encoding_share_s": encoding_share,
                        "metric": metric,
                        "coverage": coverage,
                        "delta_w": delta_w,
                        "status": "ok",
                        "result_rows": count,
                    }
                else:
                    rewriter = TimedRewrittenLikeApx(sampling["engine"])
                    relation = rewriter.create_relation(sampling["bundle"])
                    qe_mapping = {
                        prepared["predicted_view"]: relation,
                        prepared["predicted_view"].lower(): relation,
                    }
                    result = rewriter.evaluate_summary(flat_predicted, qe_mapping)
                    truth = execute_truth_set(connection, flat_truth)
                    metric, coverage, delta_w, count = set_quality(result, truth, h)
                    without_sampling = (
                        rewriter.last_rewriting_s + rewriter.last_query_s
                    )
                    row = {
                        **base,
                        "method": "QE",
                        "time_s": without_sampling + sampling_share,
                        "time_with_factor_sampling_s": without_sampling + sampling_share,
                        "time_without_factor_sampling_s": without_sampling,
                        "rewriting_time_s": rewriter.last_rewriting_s,
                        "query_time_s": rewriter.last_query_s,
                        "factor_sampling_share_s": sampling_share,
                        "metric": metric,
                        "coverage": coverage,
                        "delta_w": delta_w,
                        "status": "ok",
                        "result_rows": count,
                        "union_branches": rewriter.last_union_branches,
                    }
                connection.commit()
            except Exception as error:
                connection.rollback()
                extra = sampling_share
                if method == "CAMC":
                    extra += encoding_share
                row = timed_out_row(
                    base,
                    method,
                    timeout_s,
                    error,
                    timeout_s + extra,
                )
            rows.append(row)
            write_csv(str(output), rows)
            print(
                f"{dataset} marked set Q{query_index} at {rate}% {method}: {row['status']}",
                flush=True,
            )


def run_aggregate(
    connection,
    workload: Mapping[str, Any],
    dataset: str,
    reference: Mapping[int, Mapping[int, str]],
    method: str,
    h: int,
    timeout_s: int,
    seed: int,
    strict: bool,
    rates: Sequence[int],
    output: Path,
) -> None:
    factor_file = workload.get("aggregate_factor_file") or FACTOR_FILES.get(dataset)
    if not factor_file:
        raise ValueError(f"No aggregation factor file for {dataset}")
    ordering = {
        str(key).lower(): [str(value).lower() for value in values]
        for key, values in load_factor_map(factor_file).items()
    }
    rows: List[Dict[str, Any]] = []
    for rate in rates:
        set_timeout(connection, 0)
        prepared = prepare_rate(
            connection, workload, dataset, rate, "aggregate", seed
        )
        configured = list(workload["rates"][str(rate)]["aggregate_queries"])
        for query_index in SELECTED_QUERIES:
            configured[query_index - 11] = reference[rate][query_index]
        relevant_missing = query_missing_attributes(
            connection, prepared, ordering, configured
        )
        sampling = None
        if method in {"CAMC", "QE"}:
            sampling = build_sampling(
                connection,
                workload,
                dataset,
                rate,
                "aggregate",
                prepared,
                ordering,
                configured,
                h,
                seed,
                strict,
            )
        sampling_s = sampling["sampling_s"] if sampling else 0.0
        encoding_s = sampling["encoding_s"] if sampling else 0.0
        fallback = sampling["fallback_symbols"] if sampling else 0
        sampling_share = sampling_s / len(SELECTED_QUERIES)
        encoding_share = encoding_s / len(SELECTED_QUERIES)
        for query_index in SELECTED_QUERIES:
            query = reference[rate][query_index]
            flat_predicted = flatten_join_query(query, prepared["predicted_view"])
            flat_truth = flatten_join_query(query, prepared["truth_view"])
            base = decorated_base(
                "aggregate",
                dataset,
                rate,
                query_index,
                query,
                h,
                prepared["marked_statistics"],
                sampling_s,
                encoding_s if method == "CAMC" else 0.0,
                fallback,
            )
            set_timeout(connection, timeout_s)
            try:
                if method == "CADE":
                    elapsed, metric, coverage, count, delta_w = run_analytic_aggregate(
                        connection,
                        flat_predicted,
                        prepared["predicted_view"],
                        flat_truth,
                        ordering,
                        relevant_missing,
                        True,
                    )
                    row = {
                        **base,
                        "method": method,
                        "time_s": elapsed,
                        "metric": metric,
                        "coverage": coverage,
                        "delta_w": delta_w,
                        "result_rows": count,
                        "status": "ok",
                    }
                elif method == "CAMC":
                    elapsed, metric, coverage, count, delta_w = run_camc_aggregate(
                        sampling["engine"],
                        query,
                        sampling["mappings"],
                        connection,
                        flat_truth,
                        h,
                    )
                    row = {
                        **base,
                        "method": method,
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
                else:
                    rewriter = TimedRewrittenLikeApx(sampling["engine"])
                    relation = rewriter.create_relation(sampling["bundle"])
                    qe_mapping = {
                        prepared["predicted_view"]: relation,
                        prepared["predicted_view"].lower(): relation,
                    }
                    result = rewriter.evaluate_summary(flat_predicted, qe_mapping)
                    metric, coverage, delta_w, count = aggregate_quality(
                        result, connection, flat_truth, h
                    )
                    without_sampling = (
                        rewriter.last_rewriting_s + rewriter.last_query_s
                    )
                    row = {
                        **base,
                        "method": "QE",
                        "time_s": without_sampling + sampling_share,
                        "time_with_factor_sampling_s": without_sampling + sampling_share,
                        "time_without_factor_sampling_s": without_sampling,
                        "rewriting_time_s": rewriter.last_rewriting_s,
                        "query_time_s": rewriter.last_query_s,
                        "factor_sampling_share_s": sampling_share,
                        "metric": metric,
                        "coverage": coverage,
                        "delta_w": delta_w,
                        "result_rows": count,
                        "status": "ok",
                        "union_branches": rewriter.last_union_branches,
                    }
                connection.commit()
            except Exception as error:
                connection.rollback()
                extra = sampling_share
                if method == "CAMC":
                    extra += encoding_share
                row = timed_out_row(
                    base,
                    method,
                    timeout_s,
                    error,
                    timeout_s + extra,
                )
            rows.append(row)
            write_csv(str(output), rows)
            print(
                f"{dataset} marked aggregate Q{query_index} at {rate}% {method}: {row['status']}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("bank", "nyc", "bitcoin"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set-reference", required=True, type=Path)
    parser.add_argument("--aggregate-reference", required=True, type=Path)
    parser.add_argument(
        "--method",
        required=True,
        choices=("CAMC", "CADE", "QE", "Certain Answers"),
    )
    parser.add_argument("--workloads", default="set,aggregate")
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--allow-factor-fallback", action="store_true")
    parser.add_argument("--db-host", default=CONN["host"])
    parser.add_argument("--db-port", type=int, default=CONN["port"])
    parser.add_argument("--db-name", default=CONN["dbname"])
    parser.add_argument("--db-user", default=CONN["user"])
    parser.add_argument("--db-password", default=CONN["password"])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.h != 783:
        raise ValueError("The marked comparison uses H=783")
    workload = json.loads(args.config.read_text())
    if workload["dataset"] != args.dataset:
        raise ValueError("Configuration dataset does not match --dataset")
    rates = tuple(int(value) for value in args.rates.split(",") if value.strip())
    if not rates or not set(rates).issubset({5, 10, 20}):
        raise ValueError("--rates must contain one or more of 5, 10, and 20")
    set_reference = read_reference(args.set_reference, rates)
    aggregate_reference = read_reference(args.aggregate_reference, rates)
    workloads = {
        value.strip().lower()
        for value in args.workloads.split(",")
        if value.strip()
    }
    if args.method == "Certain Answers":
        workloads = {"set"}
    if not workloads.issubset({"set", "aggregate"}):
        raise ValueError("--workloads must contain set and/or aggregate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    connection.autocommit = False
    try:
        if "set" in workloads:
            run_set(
                connection,
                workload,
                args.dataset,
                set_reference,
                args.method,
                args.h,
                args.timeout,
                args.seed,
                not args.allow_factor_fallback,
                rates,
                args.output_dir / "set.csv",
            )
        if "aggregate" in workloads:
            run_aggregate(
                connection,
                workload,
                args.dataset,
                aggregate_reference,
                args.method,
                args.h,
                args.timeout,
                args.seed,
                not args.allow_factor_fallback,
                rates,
                args.output_dir / "aggregate.csv",
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
