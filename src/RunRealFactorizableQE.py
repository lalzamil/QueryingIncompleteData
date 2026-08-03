#!/usr/bin/env python3
"""Run QE on real-world factorizable MNAR data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import psycopg2

from MCDBPostgresNative import NativeMCDB, parse_query, qident
from QEFactorDistributionPostgres import (
    FactorDistributionBuilder,
    relation_mapping,
)
from RunQEFromFactorDistributionsFullData import TimedRewrittenLikeApx
from RunRealFactorizableCAMC import aggregate_result, set_result
from RunRealFactorizableTenQueries import (
    finite,
    flatten_query,
    relation_names,
    restricted_ordering,
    table_columns,
)
from RunSectionComparisons import set_timeout
from nonAgg_direct import _load_table


FIELDS = [
    "query_type",
    "dataset",
    "query_index",
    "relation_count",
    "method",
    "h",
    "time_s",
    "factorization_s",
    "rewriting_s",
    "query_time_s",
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


def create_qe_view(
    connection,
    query: str,
    relation_map: Mapping[str, str],
    join_key: str,
    view: str,
) -> tuple[str, int]:
    logical = relation_names(query)
    if not logical:
        raise ValueError("The query has no relation")
    physical = [relation_map[name.lower()] for name in logical]
    selected = [
        "r0.%s AS %s" % (qident(join_key), qident("_rid")),
        "r0.%s AS %s" % (qident(join_key), qident(join_key)),
        "1::integer AS %s" % qident("__marginal"),
    ]
    used = {"_rid", join_key.lower(), "__marginal"}
    for position, table in enumerate(physical):
        alias = "r%d" % position
        for column in table_columns(connection, table):
            if column in used:
                continue
            selected.append(
                "%s.%s AS %s" % (alias, qident(column), qident(column))
            )
            used.add(column)
    source = "%s r0" % qident(physical[0])
    for position, table in enumerate(physical[1:], 1):
        source += " JOIN %s r%d ON r0.%s=r%d.%s" % (
            qident(table),
            position,
            qident(join_key),
            position,
            qident(join_key),
        )
    cursor = connection.cursor()
    cursor.execute("DROP VIEW IF EXISTS %s" % qident(view))
    cursor.execute(
        "CREATE TEMP VIEW %s AS SELECT %s FROM %s"
        % (qident(view), ", ".join(selected), source)
    )
    connection.commit()
    return flatten_query(query, view), len(physical)


def required_missing_attributes(
    engine: NativeMCDB,
    query: str,
    view: str,
    ordering: Mapping[str, Sequence[str]],
) -> List[str]:
    missing = set(engine.missing_attributes(view))
    needed = set(engine._needed_columns(parse_query(query)))
    active = {attribute for attribute in needed if attribute in missing}
    changed = True
    while changed:
        changed = False
        for attribute in tuple(active):
            for separator in ordering.get(attribute, ()):
                if separator in missing and separator not in active:
                    active.add(separator)
                    changed = True
    return sorted(active)


def materialize_factor_rows(connection, view: str, table: str) -> float:
    cursor = connection.cursor()
    started = time.perf_counter()
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(table))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS SELECT * FROM %s"
        % (qident(table), qident(view))
    )
    cursor.execute(
        "CREATE UNIQUE INDEX ON %s (%s)"
        % (qident(table), qident("_rid"))
    )
    cursor.execute("ANALYZE %s" % qident(table))
    connection.commit()
    return time.perf_counter() - started


def timeout_status(error: Exception) -> str:
    message = str(error).lower()
    if "statement timeout" in message or "canceling statement" in message:
        return "timeout"
    return "error"


def run_dataset(args, dataset_key: str, dataset: Mapping[str, Any]) -> None:
    output = Path(args.output)
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
        relation_map = {
            relation["name"].lower(): relation["table"]
            for relation in dataset["relations"]
        }
        for relation in dataset["relations"]:
            _load_table(
                connection,
                relation["csv"],
                relation["table"],
                force=True,
            )
        normalized_ordering = {
            key.lower(): [value.lower() for value in values]
            for key, values in dataset["orderings"].items()
        }
        for query_type, config_key in (
            ("set", "set_queries"),
            ("aggregation", "aggregate_queries"),
        ):
            if query_type not in args.query_types:
                continue
            queries = list(dataset[config_key])
            if len(queries) != 10:
                raise ValueError(
                    "%s %s defines %d queries, not 10"
                    % (dataset_key, query_type, len(queries))
                )
            for query_index, query in enumerate(queries, 1):
                if query_index < args.query_start or query_index > args.query_end:
                    continue
                view = "real_qe_%s_%s_q%d" % (
                    dataset_key,
                    query_type,
                    query_index,
                )
                flat_query, relation_count = create_qe_view(
                    connection,
                    query,
                    relation_map,
                    dataset["join_key"].lower(),
                    view,
                )
                engine = NativeMCDB(connection)
                available = table_columns(connection, view)
                all_missing = list(engine.missing_attributes(view))
                ordering = restricted_ordering(
                    normalized_ordering,
                    all_missing,
                    available,
                )
                active = required_missing_attributes(
                    engine,
                    flat_query,
                    view,
                    ordering,
                )
                base = {
                    "query_type": query_type,
                    "dataset": dataset_key,
                    "query_index": query_index,
                    "relation_count": relation_count,
                    "method": "QE",
                    "h": args.h,
                    "query": query,
                }
                try:
                    set_timeout(connection, 0)
                    factor_table = "%s_factors" % view
                    materialization_s = materialize_factor_rows(
                        connection,
                        view,
                        factor_table,
                    )
                    builder = FactorDistributionBuilder(engine)
                    bundle = builder.create_relation(
                        table=view,
                        factor_table=factor_table,
                        missing_attributes=active,
                        ordering=ordering,
                        h=args.h,
                        seed=args.seed,
                        prefix="real_qe_%s_%s_q%d" % (
                            dataset_key,
                            query_type,
                            query_index,
                        ),
                        n_bins=5,
                        allow_fallback=False,
                    )
                    bundle.factorization_s += materialization_s
                    rewriter = TimedRewrittenLikeApx(engine)
                    mappings = relation_mapping(bundle, view)
                    set_timeout(connection, args.timeout)
                    if query_type == "set":
                        result = set_result(
                            rewriter,
                            mappings,
                            flat_query,
                            args.h,
                        )
                    else:
                        result = aggregate_result(
                            rewriter,
                            mappings,
                            flat_query,
                            args.h,
                        )
                    elapsed = (
                        bundle.factorization_s
                        + rewriter.last_rewriting_s
                        + result["query_time_s"]
                    )
                    rows.append(
                        {
                            **base,
                            **result,
                            "time_s": elapsed,
                            "factorization_s": bundle.factorization_s,
                            "rewriting_s": rewriter.last_rewriting_s,
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
                            "factorization_s": None,
                            "rewriting_s": None,
                            "query_time_s": None,
                            "delta_w": None,
                            "result_rows": None,
                            "status": timeout_status(error),
                            "error": str(error),
                        }
                    )
                write_rows(output, rows)
                print(
                    "%s %s Q%d QE: %s, %.6fs"
                    % (
                        dataset_key,
                        query_type,
                        query_index,
                        rows[-1]["status"],
                        float(rows[-1]["time_s"]),
                    ),
                    flush=True,
                )
    finally:
        connection.close()
    write_rows(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_factorizable_10_queries.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--db-host", default=os.environ.get("PGHOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("PGPORT", "5433")))
    parser.add_argument("--db-name", default=os.environ.get("PGDATABASE", "mydb"))
    parser.add_argument("--db-user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--db-password", default=os.environ.get("PGPASSWORD", ""))
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-types", default="set,aggregation")
    parser.add_argument("--query-start", type=int, default=1)
    parser.add_argument("--query-end", type=int, default=10)
    args = parser.parse_args()
    if args.h != 783:
        raise ValueError("QE must use H=783")
    args.query_types = {
        value.strip().lower()
        for value in args.query_types.split(",")
        if value.strip()
    }
    if not args.query_types <= {"set", "aggregation"}:
        raise ValueError("Unknown query type")
    if not 1 <= args.query_start <= args.query_end <= 10:
        raise ValueError("Query indices must be between 1 and 10")
    datasets = json.load(open(args.config))
    dataset_key = args.dataset.strip().lower()
    if dataset_key not in datasets:
        raise ValueError("Unknown dataset %s" % dataset_key)
    run_dataset(args, dataset_key, datasets[dataset_key])


if __name__ == "__main__":
    main()
