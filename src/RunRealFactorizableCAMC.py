#!/usr/bin/env python3
"""Run CAMC on ten set and ten aggregation queries per real dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import psycopg2

from MCDBPostgresNative import NativeMCDB, qident, relation_mapping
from RunRealFactorizableTenQueries import (
    create_query_view,
    finite,
    flatten_query,
    relation_names,
    restricted_ordering,
    table_columns,
)
from RunSectionComparisons import (
    CONN,
    mean_normalized_width,
    ratio_of_mean_interval_width,
    set_timeout,
    wilson_interval,
)
from nonAgg_direct import _load_table


FIELDS = [
    "query_type",
    "dataset",
    "query_index",
    "relation_count",
    "method",
    "h",
    "time_s",
    "query_time_s",
    "factor_sampling_share_s",
    "encoding_share_s",
    "total_factor_sampling_s",
    "total_encoding_s",
    "delta_w",
    "result_rows",
    "status",
    "query",
    "error",
]


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: finite(row.get(field)) for field in FIELDS})


def full_relation_query(dataset: Mapping[str, Any]) -> str:
    relations = [relation["name"] for relation in dataset["relations"]]
    query = "SELECT %s FROM %s" % (dataset["join_key"], relations[0])
    for relation in relations[1:]:
        query += " JOIN %s USING (%s)" % (relation, dataset["join_key"])
    return query


def build_bundle(
    connection,
    dataset_key: str,
    dataset: Mapping[str, Any],
    query_type: str,
    h: int,
    seed: int,
):
    relation_map = {
        relation["name"].lower(): relation["table"]
        for relation in dataset["relations"]
    }
    full_view = "real_camc_%s_%s_full" % (dataset_key, query_type)
    create_query_view(
        connection,
        full_relation_query(dataset),
        relation_map,
        dataset["join_key"].lower(),
        full_view,
    )
    working_table = "real_camc_%s_%s_input" % (
        dataset_key,
        query_type,
    )
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(working_table))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
        "SELECT row_number() OVER (ORDER BY source.%s)::bigint AS %s, "
        "source.* FROM %s source"
        % (
            qident(working_table),
            qident(dataset["join_key"].lower()),
            qident("_rid"),
            qident(full_view),
        )
    )
    cursor.execute(
        "CREATE UNIQUE INDEX ON %s (%s)"
        % (qident(working_table), qident("_rid"))
    )
    cursor.execute("ANALYZE %s" % qident(working_table))
    connection.commit()
    engine = NativeMCDB(connection)
    missing = list(engine.missing_attributes(working_table))
    ordering = restricted_ordering(
        {
            key.lower(): [value.lower() for value in values]
            for key, values in dataset["orderings"].items()
        },
        missing,
        table_columns(connection, working_table),
    )
    bundle = engine.create_bundle(
        working_table,
        missing,
        ordering,
        h,
        seed=float(seed % 1000) / 1000.0,
        prefix="real_camc_%s_%s_%d" % (dataset_key, query_type, h),
        strict=True,
        n_bins=5,
        factor_table=working_table,
    )
    mappings = relation_mapping(bundle, full_view, working_table)
    return engine, mappings, full_view, bundle


def set_result(engine: NativeMCDB, mappings, query: str, h: int) -> Dict[str, Any]:
    result = engine.evaluate_summary(query, mappings)
    intervals = []
    for row in result.summary_rows:
        probability = float(row[-1])
        lower, upper = wilson_interval(round(probability * h), h)
        intervals.append((probability, lower, upper))
    return {
        "query_time_s": result.elapsed_s,
        "delta_w": ratio_of_mean_interval_width(intervals),
        "result_rows": len(result.summary_rows),
    }


def aggregate_result(
    engine: NativeMCDB,
    mappings,
    query: str,
    h: int,
) -> Dict[str, Any]:
    result = engine.evaluate_summary(query, mappings)
    group_count = len(result.query.group_by)
    intervals = []
    for row in result.summary_rows:
        estimate = row[group_count]
        if estimate is None:
            continue
        estimate = float(estimate)
        sample_sd = row[group_count + 1]
        sample_sd = float(sample_sd) if sample_sd is not None else 0.0
        half_width = 1.96 * sample_sd / math.sqrt(h)
        intervals.append(
            (estimate, estimate - half_width, estimate + half_width)
        )
    return {
        "query_time_s": result.elapsed_s,
        "delta_w": mean_normalized_width(intervals),
        "result_rows": len(result.summary_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_factorizable_10_queries.json")
    parser.add_argument("--datasets", default="student,aircraft,medical")
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    datasets = json.load(open(args.config))
    selected = [
        value.strip().lower()
        for value in args.datasets.split(",")
        if value.strip()
    ]
    unknown = sorted(set(selected) - set(datasets))
    if unknown:
        raise ValueError("Unknown datasets: %s" % ", ".join(unknown))

    output = Path(args.output)
    rows: List[Dict[str, Any]] = []
    connection = psycopg2.connect(**CONN)
    connection.autocommit = False
    try:
        for dataset_key in selected:
            dataset = datasets[dataset_key]
            for relation in dataset["relations"]:
                _load_table(
                    connection,
                    relation["csv"],
                    relation["table"],
                    force=True,
                )
            for query_type, config_key in (
                ("set", "set_queries"),
                ("aggregation", "aggregate_queries"),
            ):
                set_timeout(connection, 0)
                engine, mappings, full_view, bundle = build_bundle(
                    connection,
                    dataset_key,
                    dataset,
                    query_type,
                    args.h,
                    args.seed,
                )
                query_count = len(dataset[config_key])
                if query_count != 10:
                    raise ValueError(
                        "%s %s defines %d queries, not 10"
                        % (dataset_key, query_type, query_count)
                    )
                sampling_share = bundle.sampling_s / query_count
                encoding_share = bundle.encoding_s / query_count
                print(
                    "%s %s setup: sampling %.6fs; encoding %.6fs"
                    % (
                        dataset_key,
                        query_type,
                        bundle.sampling_s,
                        bundle.encoding_s,
                    ),
                    flush=True,
                )
                for query_index, query in enumerate(dataset[config_key], 1):
                    flat_query = flatten_query(query, full_view)
                    base = {
                        "query_type": query_type,
                        "dataset": dataset_key,
                        "query_index": query_index,
                        "relation_count": len(relation_names(query)),
                        "method": "CAMC",
                        "h": args.h,
                        "factor_sampling_share_s": sampling_share,
                        "encoding_share_s": encoding_share,
                        "total_factor_sampling_s": bundle.sampling_s,
                        "total_encoding_s": bundle.encoding_s,
                        "query": query,
                    }
                    set_timeout(connection, args.timeout)
                    try:
                        if query_type == "set":
                            result = set_result(
                                engine,
                                mappings,
                                flat_query,
                                args.h,
                            )
                        else:
                            result = aggregate_result(
                                engine,
                                mappings,
                                flat_query,
                                args.h,
                            )
                        rows.append(
                            {
                                **base,
                                **result,
                                "time_s": (
                                    result["query_time_s"]
                                    + sampling_share
                                    + encoding_share
                                ),
                                "status": "ok",
                                "error": "",
                            }
                        )
                    except Exception as error:
                        connection.rollback()
                        rows.append(
                            {
                                **base,
                                "time_s": float(args.timeout),
                                "query_time_s": None,
                                "delta_w": None,
                                "result_rows": None,
                                "status": (
                                    "timeout"
                                    if "statement timeout" in str(error).lower()
                                    else "error"
                                ),
                                "error": str(error),
                            }
                        )
                    write_rows(output, rows)
                    print(
                        "%s %s Q%d CAMC: %s"
                        % (
                            dataset_key,
                            query_type,
                            query_index,
                            rows[-1]["status"],
                        ),
                        flush=True,
                    )
    finally:
        connection.close()
    write_rows(output, rows)
    print("Saved %d CAMC measurements to %s" % (len(rows), output))


if __name__ == "__main__":
    main()
