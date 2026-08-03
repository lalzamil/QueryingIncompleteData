"""Run the union-based LikeApx baseline on the marked-null CAMC queries.

The runner uses the same factor sampler and the same ten queries as the saved
marked-null CAMC experiment.  It records query rewriting, PostgreSQL execution,
and factor sampling separately.  Factor sampling is performed once per
dataset, query type, and missingness rate and is amortized over ten queries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psycopg2

from LikeApxRewrittenPostgres import RewrittenLikeApx
from MCDBPostgresNative import QueryResult
from RunSectionComparisonsFullData import (
    CONN,
    GROUPS,
    SEPARATOR_FILES,
    append_extra_aggregate_queries,
    assign_marked_null_symbols,
    block_for_rate,
    build_camc,
    clean_name,
    create_join_view,
    execute_truth_set,
    expose_group_columns,
    flatten_join_query,
    load_factor_map,
    mean_normalized_width,
    normalize_key,
    output_tvd,
    prepare_block,
    ratio_of_mean_interval_width,
    remove_redundant_join_grouping,
    replace_tokens,
    set_timeout,
    wilson_interval,
    write_csv,
)


METHOD = "LikeApx"


class TimedRewrittenLikeApx(RewrittenLikeApx):
    """Measure SQL construction separately from PostgreSQL execution."""

    def __init__(self, engine):
        super().__init__(engine)
        self.last_rewriting_s = 0.0
        self.last_query_s = 0.0
        self.last_union_branches = 0

    def evaluate_summary(self, query, relations):
        rewrite_started = time.perf_counter()
        spec, sql, summary_columns = self.compile(query, relations)
        self.last_rewriting_s = time.perf_counter() - rewrite_started
        expected = self._resolve_relation(spec.left_token, relations).h
        branches = sql.count("\nUNION ALL\n") + 1
        if branches != expected:
            raise AssertionError(
                "Expected %d valuation-specific queries but generated %d" %
                (expected, branches)
            )
        self.last_union_branches = branches
        cursor = self.connection.cursor()
        query_started = time.perf_counter()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            self.last_query_s = time.perf_counter() - query_started
        return QueryResult(
            query=spec,
            world_rows=[],
            summary_rows=rows,
            world_columns=tuple(),
            summary_columns=summary_columns,
            sql=sql,
            elapsed_s=self.last_query_s,
        )


def read_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_sql(query: str) -> str:
    return " ".join(query.strip().rstrip(";").split()).lower()


def reference_queries(path: str, h: int) -> Dict[Tuple[str, int, int], str]:
    rows = read_rows(path)
    if not rows:
        raise ValueError("No CAMC reference rows found in %s" % path)
    result: Dict[Tuple[str, int, int], str] = {}
    for row in rows:
        if row.get("method") != "CAMC":
            continue
        if int(row["h"]) != h:
            raise ValueError("CAMC reference H does not match H=%d" % h)
        if row.get("null_semantics") != "marked":
            raise ValueError("CAMC reference is not a marked-null experiment")
        if int(row.get("marked_null_group_size") or 0) != 3:
            raise ValueError("CAMC reference does not use marked-null groups of size three")
        if int(row.get("row_limit") or 0) != 0:
            raise ValueError("CAMC reference does not use the full data")
        key = (row["workload"], int(row["rate"]), int(row["query_index"]))
        result[key] = row["query"]
    expected = {
        (workload, rate, query_index)
        for workload in ("set", "aggregate")
        for rate in (5, 10, 20)
        for query_index in range(1, 11)
    }
    missing = sorted(expected - set(result))
    if missing:
        raise ValueError("CAMC reference is missing queries: %s" % missing)
    return result


def assert_reference_query(
    references: Mapping[Tuple[str, int, int], str],
    workload: str,
    rate: int,
    query_index: int,
    query: str,
) -> None:
    expected = references[(workload, rate, query_index)]
    if normalized_sql(query) != normalized_sql(expected):
        raise ValueError(
            "%s %d%% Q%d differs from the saved marked-null CAMC query\n"
            "CAMC: %s\nLikeApx: %s" %
            (workload, rate, query_index, expected, query)
        )


def rewritten_mapping(
    rewriter: TimedRewrittenLikeApx,
    bundle_mapping: Mapping[str, Any],
) -> Dict[str, Any]:
    cache: Dict[int, Any] = {}
    result: Dict[str, Any] = {}
    for token, bundle in bundle_mapping.items():
        identity = id(bundle)
        if identity not in cache:
            cache[identity] = rewriter.create_relation(bundle)
        result[token] = cache[identity]
    return result


def set_quality(
    result: QueryResult,
    truth: Sequence[Tuple[Any, ...]],
    h: int,
) -> Tuple[float, float, Optional[float], int]:
    predicted = {
        normalize_key(row[:-1]): float(row[-1])
        for row in result.summary_rows
    }
    tvd = output_tvd(predicted, truth)
    truth_set = set(truth)
    keys = set(predicted) | truth_set
    covered = 0
    intervals = []
    for key in keys:
        probability = predicted.get(key, 0.0)
        lower, upper = wilson_interval(round(probability * h), h)
        covered += int(
            lower <= (1.0 if key in truth_set else 0.0) <= upper
        )
        if key in predicted:
            intervals.append((probability, lower, upper))
    coverage = covered / len(keys) if keys else 1.0
    delta_w = ratio_of_mean_interval_width(intervals)
    return tvd, coverage, delta_w, len(predicted)


def aggregate_quality(
    result: QueryResult,
    conn,
    truth_query: str,
    h: int,
) -> Tuple[float, float, Optional[float], int]:
    spec = result.query
    n_group = len(spec.group_by)
    cursor = conn.cursor()
    cursor.execute(truth_query)
    truth_rows = cursor.fetchall()
    truth = {
        normalize_key(row[:n_group]): float(row[n_group])
        for row in truth_rows if row[n_group] is not None
    }
    predicted = {}
    for row in result.summary_rows:
        key = normalize_key(row[:n_group])
        estimate = float(row[n_group]) if row[n_group] is not None else 0.0
        sample_sd = (
            float(row[n_group + 1])
            if row[n_group + 1] is not None else 0.0
        )
        predicted[key] = (estimate, sample_sd / (h ** 0.5))
    errors = []
    covered = 0
    for key, value in truth.items():
        if key not in predicted:
            errors.append(1.0)
            continue
        estimate, stderr = predicted[key]
        errors.append(abs(estimate - value) / max(abs(value), 1e-12))
        covered += int(
            estimate - 1.96 * stderr <= value <= estimate + 1.96 * stderr
        )
    metric = sum(errors) / len(errors) if errors else 0.0
    coverage = covered / len(truth) if truth else 1.0
    delta_w = mean_normalized_width(
        (estimate, estimate - 1.96 * stderr, estimate + 1.96 * stderr)
        for estimate, stderr in predicted.values()
    )
    return metric, coverage, delta_w, len(predicted)


def timeout_status(error: Exception) -> str:
    message = str(error).lower()
    if "statement timeout" in message or "canceling statement" in message:
        return "timeout"
    return "error"


def run_queries(
    conn,
    dataset: str,
    block: str,
    workload: str,
    meta: Mapping[str, Any],
    prepared: Mapping[str, Any],
    rewriter: TimedRewrittenLikeApx,
    mappings: Mapping[str, Any],
    references: Mapping[Tuple[str, int, int], str],
    h: int,
    timeout_s: int,
    sampling_s: float,
    completed: Iterable[int],
    checkpoint,
) -> List[Dict[str, Any]]:
    rate = int(re.search(r"(5|10|20)$", block).group(1))
    sampling_share = sampling_s / 10.0
    queries = list(meta["queries"])
    if len(queries) != 10:
        raise ValueError("%s must contain exactly ten queries" % block)
    completed_set = set(completed)
    truth_view = create_join_view(
        conn,
        prepared["full_tables"][1],
        prepared["full_tables"][2],
        prepared["join_key"],
        clean_name("likeapx_truth_%s_%s" % (workload, block)),
    )
    rows: List[Dict[str, Any]] = []
    for query_index, configured_query in enumerate(queries, 1):
        if query_index in completed_set:
            continue
        query = (
            remove_redundant_join_grouping(configured_query)
            if workload == "set"
            else expose_group_columns(configured_query)
        )
        assert_reference_query(
            references, workload, rate, query_index, query
        )
        is_join = " JOIN " in query.upper()
        if is_join:
            truth_query = flatten_join_query(query, truth_view)
        else:
            truth_query = replace_tokens(query, prepared["truth_map"])
        truth_set = None
        set_timeout(conn, 0)
        if workload == "set":
            truth_set = execute_truth_set(conn, truth_query)
        set_timeout(conn, timeout_s)
        base = {
            "workload": workload,
            "dataset": dataset,
            "block": block,
            "rate": rate,
            "query_index": query_index,
            "query": query,
            "h": h,
            "method": METHOD,
        }
        rewriter.last_rewriting_s = 0.0
        rewriter.last_query_s = 0.0
        rewriter.last_union_branches = 0
        try:
            result = rewriter.evaluate_summary(query, mappings)
            if workload == "set":
                metric, coverage, delta_w, n_rows = set_quality(
                    result, truth_set or [], h
                )
            else:
                set_timeout(conn, 0)
                metric, coverage, delta_w, n_rows = aggregate_quality(
                    result, conn, truth_query, h
                )
            without_sampling = (
                rewriter.last_rewriting_s + rewriter.last_query_s
            )
            rows.append({
                **base,
                "time_s": without_sampling + sampling_share,
                "time_with_factor_sampling_s": (
                    without_sampling + sampling_share
                ),
                "time_without_factor_sampling_s": without_sampling,
                "rewriting_time_s": rewriter.last_rewriting_s,
                "query_time_s": rewriter.last_query_s,
                "factor_sampling_share_s": sampling_share,
                "metric": metric,
                "coverage": coverage,
                "delta_w": delta_w,
                "status": "ok",
                "result_rows": n_rows,
                "union_branches": rewriter.last_union_branches,
            })
        except Exception as error:
            conn.rollback()
            without_sampling = (
                rewriter.last_rewriting_s + rewriter.last_query_s
            )
            rows.append({
                **base,
                "time_s": without_sampling + sampling_share,
                "time_with_factor_sampling_s": (
                    without_sampling + sampling_share
                ),
                "time_without_factor_sampling_s": without_sampling,
                "rewriting_time_s": rewriter.last_rewriting_s,
                "query_time_s": rewriter.last_query_s,
                "factor_sampling_share_s": sampling_share,
                "metric": None,
                "coverage": None,
                "delta_w": None,
                "status": timeout_status(error),
                "result_rows": None,
                "union_branches": rewriter.last_union_branches,
                "error": str(error),
            })
        print(
            "%s %s %d%% Q%d %s: rewrite %.3fs, query %.3fs, "
            "factor share %.3fs" % (
                dataset,
                workload,
                rate,
                query_index,
                rows[-1]["status"],
                rewriter.last_rewriting_s,
                rewriter.last_query_s,
                sampling_share,
            ),
            flush=True,
        )
        checkpoint(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-config", default="configs/mnar_set_queries.json")
    parser.add_argument("--agg-config", default="configs/mnar1_agg_inj_query.json")
    parser.add_argument(
        "--agg-extra", default="configs/section_comparison_agg_queries.json"
    )
    parser.add_argument("--db-host", default=CONN["host"])
    parser.add_argument("--db-port", type=int, default=CONN["port"])
    parser.add_argument("--db-name", default=CONN["dbname"])
    parser.add_argument("--db-user", default=CONN["user"])
    parser.add_argument("--db-password", default=CONN["password"])
    parser.add_argument("--datasets", default="bank,nyc,bitcoin")
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--workloads", default="set,aggregate")
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--marked-null-group-size", type=int, default=3)
    parser.add_argument("--marked-null-seed", type=int, default=42)
    parser.add_argument(
        "--allow-factor-fallback",
        action="store_true",
        help=(
            "Use the observed marginal only when a finite-sample "
            "conditioning group has no observed donor"
        ),
    )
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument(
        "--camc-reference-dir",
        default=(
            "psql_results/section_comparisons/"
            "full_data_20260727_camc_per_dataset_isolated_300s/"
            "marked_group3"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "psql_results/section_comparisons/"
            "full_data_20260730_likeapx_marked_group3_300s/"
            "likeapx_marked.csv"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()
    if args.h != 783:
        raise ValueError("This comparison must use H=783")
    if args.rows != 0:
        raise ValueError("This comparison must use the full data")
    if args.marked_null_group_size != 3:
        raise ValueError("This comparison must use marked-null groups of size three")

    with open(args.set_config) as handle:
        set_config = json.load(handle)
    with open(args.agg_config) as handle:
        agg_config = json.load(handle)
    append_extra_aggregate_queries(agg_config, args.agg_extra)

    datasets = [
        value.strip().lower()
        for value in args.datasets.split(",") if value.strip()
    ]
    rates = [
        int(value) for value in args.rates.split(",") if value.strip()
    ]
    workloads = [
        value.strip().lower()
        for value in args.workloads.split(",") if value.strip()
    ]
    results: List[Dict[str, Any]] = (
        [dict(row) for row in read_rows(args.output)]
        if args.resume else []
    )
    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    conn.autocommit = False
    try:
        for dataset in datasets:
            if dataset not in GROUPS:
                raise ValueError("Unknown dataset %s" % dataset)
            reference_path = os.path.join(
                args.camc_reference_dir, "%s.csv" % dataset
            )
            references = reference_queries(reference_path, args.h)
            set_group, agg_group = GROUPS[dataset]
            factor_map = load_factor_map(SEPARATOR_FILES[dataset])
            for rate in rates:
                for workload in workloads:
                    completed = {
                        int(row["query_index"])
                        for row in results
                        if row.get("dataset") == dataset
                        and row.get("workload") == workload
                        and int(row.get("rate", -1)) == rate
                        and row.get("method") == METHOD
                    }
                    if len(completed) == 10:
                        print(
                            "Skipping completed %s %s at %d%%" %
                            (dataset, workload, rate),
                            flush=True,
                        )
                        continue
                    group = (
                        set_config[set_group]
                        if workload == "set" else agg_config[agg_group]
                    )
                    block, meta = block_for_rate(group, rate)
                    print(
                        "Preparing LikeApx %s %s at %d%%" %
                        (dataset, workload, rate),
                        flush=True,
                    )
                    set_timeout(conn, 0)
                    prepared = prepare_block(
                        conn,
                        meta,
                        dataset,
                        block,
                        workload,
                        args.rows,
                        args.seed,
                        force_reload=args.force_reload,
                    )
                    marked_statistics = assign_marked_null_symbols(
                        conn,
                        prepared["pred_tables"],
                        args.marked_null_group_size,
                        args.marked_null_seed,
                    )
                    engine, bundle_mapping, sampling_s, unused_encoding_s, fallback = build_camc(
                        conn,
                        meta,
                        prepared,
                        workload,
                        factor_map,
                        args.h,
                        args.seed,
                        strict=not args.allow_factor_fallback,
                    )
                    if fallback and not args.allow_factor_fallback:
                        raise ValueError(
                            "Factor sampler used %d fallback draws" % fallback
                        )
                    rewriter = TimedRewrittenLikeApx(engine)
                    mappings = rewritten_mapping(rewriter, bundle_mapping)
                    print(
                        "LikeApx factor sampling %.3fs; MCDB encoding %.3fs "
                        "excluded; fallback symbols %d" %
                        (sampling_s, unused_encoding_s, fallback),
                        flush=True,
                    )

                    def checkpoint(partial_rows):
                        decorated = []
                        for partial in partial_rows:
                            row = dict(partial)
                            row["row_limit"] = args.rows
                            row["total_factor_sampling_s"] = sampling_s
                            row["total_encoding_s"] = 0.0
                            row["factor_fallback_symbols"] = fallback
                            row["null_semantics"] = "marked"
                            row["marked_null_group_size"] = (
                                args.marked_null_group_size
                            )
                            row.update(marked_statistics)
                            decorated.append(row)
                        write_csv(args.output, results + decorated)

                    block_rows = run_queries(
                        conn,
                        dataset,
                        block,
                        workload,
                        meta,
                        prepared,
                        rewriter,
                        mappings,
                        references,
                        args.h,
                        args.timeout,
                        sampling_s,
                        completed,
                        checkpoint,
                    )
                    for row in block_rows:
                        row["row_limit"] = args.rows
                        row["total_factor_sampling_s"] = sampling_s
                        row["total_encoding_s"] = 0.0
                        row["factor_fallback_symbols"] = fallback
                        row["null_semantics"] = "marked"
                        row["marked_null_group_size"] = (
                            args.marked_null_group_size
                        )
                        row.update(marked_statistics)
                    results.extend(block_rows)
                    write_csv(args.output, results)
    finally:
        conn.close()
    write_csv(args.output, results)
    print(
        "Saved %d LikeApx measurements to %s" %
        (len(results), args.output),
        flush=True,
    )


if __name__ == "__main__":
    main()
