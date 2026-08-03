"""Run QE after explicitly computing the factor distributions.

The factor distributions are materialized before sampling the H repairs.  QE
then rewrites one query for each repair and combines the queries with UNION ALL.
The runner records factorization separately.  Repair sampling is performed by
the valuation-specific queries and is therefore included in query execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psycopg2

from MCDBPostgresNative import NativeMCDB, QueryResult, qident
from QEFactorDistributionPostgres import (
    FactorDistributionBuilder,
    FactorDistributionQE,
    QERelation,
    relation_mapping as qe_relation_mapping,
)
from RunSectionComparisonsFullData import (
    CONN,
    GROUPS,
    SEPARATOR_FILES,
    append_extra_aggregate_queries,
    assign_marked_null_symbols,
    block_for_rate,
    clean_name,
    columns,
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
    relation_settings,
    required_missing_attributes,
    replace_tokens,
    set_timeout,
    wilson_interval,
    write_csv,
)


METHOD = "QE"


class TimedRewrittenLikeApx(FactorDistributionQE):
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


def build_qe_distributions(
    conn,
    meta: Mapping[str, Any],
    prepared: Mapping[str, Any],
    workload: str,
    factor_map: Mapping[str, Sequence[str]],
    h: int,
    seed: int,
    strict: bool = True,
) -> Tuple[NativeMCDB, Dict[str, Any], float, float, int]:
    """Compute factor distributions, then sample H repairs from them."""
    engine = NativeMCDB(conn)
    builder = FactorDistributionBuilder(engine)
    required_by_position = required_missing_attributes(
        engine, meta, prepared, workload, factor_map
    )
    mappings: Dict[str, Any] = {}
    preprocessing_s = 0.0
    encoding_s = 0.0
    fallback_symbols = 0

    def create(position: int, table: str, missing, ordering):
        available = set(columns(conn, table))
        restricted = {
            attr: tuple(value for value in separators if value in available)
            for attr, separators in ordering.items() if attr in available
        }
        for attr in missing:
            restricted.setdefault(attr, tuple())
        return builder.create_relation(
            table=table,
            factor_table=prepared["tables"][position],
            missing_attributes=missing,
            ordering=restricted,
            h=h,
            seed=seed,
            prefix="qe_dist_%s_%d" % (clean_name(table), h),
            n_bins=5,
            allow_fallback=not strict,
        )

    def project(joint: QERelation, source_table: str, join_key: str,
                name: str) -> QERelation:
        join_key = join_key.lower()
        relation_columns = [
            column for column in columns(conn, source_table)
            if column not in ("_rid", join_key)
        ]
        selected = ["_rid", join_key] + relation_columns
        for attribute in relation_columns:
            symbol_column = "%s_nullsym" % attribute
            if symbol_column in joint.columns:
                selected.append(symbol_column)
        selected = list(dict.fromkeys(selected))
        cursor = conn.cursor()
        projected_name = clean_name(name)
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(projected_name))
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS SELECT %s "
            "FROM %s" % (
                qident(projected_name),
                ", ".join(qident(value) for value in selected),
                qident(joint.base_table),
            )
        )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (%s)" % (
                qident(projected_name), qident("_rid"),
            )
        )
        conn.commit()
        selected_set = set(selected)
        kept_columns = tuple(
            column for column in joint.columns if column in selected_set
        )
        kept_missing = tuple(
            attribute for attribute in joint.missing_attributes
            if attribute in relation_columns
        )
        return QERelation(
            base_table=projected_name,
            h=joint.h,
            columns=kept_columns,
            column_types={
                column: value for column, value in joint.column_types.items()
                if column in selected_set
            },
            missing_attributes=kept_missing,
            distributions={
                attribute: joint.distributions[attribute]
                for attribute in kept_missing
            },
        )

    if workload == "set" and meta.get("semantic_relations"):
        _configured, ordering = relation_settings(meta, 0, workload, factor_map)
        joint = create(0, prepared["pred_tables"][0], required_by_position[0], ordering)
        preprocessing_s += joint.factorization_s
        fallback_symbols += joint.fallback_occurrences
        configured = prepared["tables"][0]
        csv_token = prepared["csvs"][0]
        mappings.update(qe_relation_mapping(
            joint, configured, csv_token, os.path.basename(csv_token),
            os.path.splitext(os.path.basename(csv_token))[0],
        ))
        projection_started = time.perf_counter()
        for position in (1, 2):
            projected = project(
                joint,
                prepared["pred_tables"][position],
                prepared["join_key"],
                clean_name("qe_relation_%d_%s" % (
                    position, prepared["pred_tables"][0]
                ))[:55],
            )
            configured = prepared["tables"][position]
            csv_token = prepared["csvs"][position]
            mappings.update(qe_relation_mapping(
                projected, configured, csv_token, os.path.basename(csv_token),
                os.path.splitext(os.path.basename(csv_token))[0],
            ))
        encoding_s += time.perf_counter() - projection_started
        return engine, mappings, preprocessing_s, encoding_s, fallback_symbols

    for position, table in enumerate(prepared["pred_tables"]):
        _configured, ordering = relation_settings(
            meta, position, workload, factor_map
        )
        bundle = create(
            position, table, required_by_position[position], ordering
        )
        preprocessing_s += bundle.factorization_s
        fallback_symbols += bundle.fallback_occurrences
        configured = prepared["tables"][position]
        csv_token = prepared["csvs"][position]
        mappings.update(qe_relation_mapping(
            bundle, configured, csv_token, os.path.basename(csv_token),
            os.path.splitext(os.path.basename(csv_token))[0],
        ))
    return engine, mappings, preprocessing_s, encoding_s, fallback_symbols


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
    h: int,
    timeout_s: int,
    preprocessing_s: float,
    completed: Iterable[int],
    checkpoint,
) -> List[Dict[str, Any]]:
    rate = int(re.search(r"(5|10|20)$", block).group(1))
    factorization_share = preprocessing_s / 10.0
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
            without_preprocessing = (
                rewriter.last_rewriting_s + rewriter.last_query_s
            )
            rows.append({
                **base,
                "time_s": without_preprocessing + factorization_share,
                "time_with_factorization_s": (
                    without_preprocessing + factorization_share
                ),
                "time_without_factorization_s": without_preprocessing,
                "rewriting_time_s": rewriter.last_rewriting_s,
                "query_time_s": rewriter.last_query_s,
                "factorization_share_s": factorization_share,
                "metric": metric,
                "coverage": coverage,
                "delta_w": delta_w,
                "status": "ok",
                "result_rows": n_rows,
                "union_branches": rewriter.last_union_branches,
            })
        except Exception as error:
            conn.rollback()
            without_preprocessing = (
                rewriter.last_rewriting_s + rewriter.last_query_s
            )
            rows.append({
                **base,
                "time_s": without_preprocessing + factorization_share,
                "time_with_factorization_s": (
                    without_preprocessing + factorization_share
                ),
                "time_without_factorization_s": without_preprocessing,
                "rewriting_time_s": rewriter.last_rewriting_s,
                "query_time_s": rewriter.last_query_s,
                "factorization_share_s": factorization_share,
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
            "factorization share %.3fs" % (
                dataset,
                workload,
                rate,
                query_index,
                rows[-1]["status"],
                rewriter.last_rewriting_s,
                rewriter.last_query_s,
                factorization_share,
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
        "--null-semantics",
        choices=("nonrepeating", "marked"),
        required=True,
    )
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
        "--output",
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()
    if args.h != 783:
        raise ValueError("This comparison must use H=783")
    if args.rows != 0:
        raise ValueError("This comparison must use the full data")
    if args.null_semantics == "marked" and args.marked_null_group_size != 3:
        raise ValueError(
            "The marked-null comparison must use groups of size three"
        )

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
                        "Preparing QE %s %s at %d%%" %
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
                    if args.null_semantics == "marked":
                        marked_statistics = assign_marked_null_symbols(
                            conn,
                            prepared["pred_tables"],
                            args.marked_null_group_size,
                            args.marked_null_seed,
                        )
                    else:
                        marked_statistics = {
                            "marked_null_count": 0,
                            "marked_symbol_count": 0,
                            "marked_repeated_symbol_count": 0,
                        }
                    engine, bundle_mapping, preprocessing_s, unused_encoding_s, fallback = build_qe_distributions(
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
                            "%d null symbols had no factor distribution" % fallback
                        )
                    rewriter = TimedRewrittenLikeApx(engine)
                    mappings = rewritten_mapping(rewriter, bundle_mapping)
                    print(
                        "QE factorization %.3fs; relation projection "
                        "%.3fs excluded; fallback symbols %d" %
                        (preprocessing_s, unused_encoding_s, fallback),
                        flush=True,
                    )

                    def checkpoint(partial_rows):
                        decorated = []
                        for partial in partial_rows:
                            row = dict(partial)
                            row["row_limit"] = args.rows
                            row["total_factorization_s"] = (
                                preprocessing_s
                            )
                            row["total_encoding_s"] = 0.0
                            row["factor_fallback_symbols"] = fallback
                            row["null_semantics"] = args.null_semantics
                            row["marked_null_group_size"] = (
                                args.marked_null_group_size
                                if args.null_semantics == "marked" else 1
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
                        args.h,
                        args.timeout,
                        preprocessing_s,
                        completed,
                        checkpoint,
                    )
                    for row in block_rows:
                        row["row_limit"] = args.rows
                        row["total_factorization_s"] = (
                            preprocessing_s
                        )
                        row["total_encoding_s"] = 0.0
                        row["factor_fallback_symbols"] = fallback
                        row["null_semantics"] = args.null_semantics
                        row["marked_null_group_size"] = (
                            args.marked_null_group_size
                            if args.null_semantics == "marked" else 1
                        )
                        row.update(marked_statistics)
                    results.extend(block_rows)
                    write_csv(args.output, results)
    finally:
        conn.close()
    write_csv(args.output, results)
    print(
        "Saved %d QE measurements to %s" %
        (len(results), args.output),
        flush=True,
    )


if __name__ == "__main__":
    main()
