#!/usr/bin/env python3
"""Run CAEX and CADE on ten set and ten aggregation queries per real dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import psycopg2

from MCDBPostgresNative import NativeMCDB, qident
from RunSectionComparisons import CONN, aggregate_sql, set_timeout
from RunnerSetQueriy import evaluate_query_with_groups
from SetQueryRewriterExecuter import QueryExecutor as CAEXSetExecutor
from nonAgg_direct import _load_table, run_direct_per_tuple


def relation_names(query: str) -> List[str]:
    return re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", query, flags=re.IGNORECASE)


def flatten_query(query: str, view: str) -> str:
    return re.sub(
        r"\bFROM\s+.+?(?=\s+WHERE\s+|\s+GROUP\s+BY\s+|\s+HAVING\s+|$)",
        "FROM " + view,
        " ".join(query.strip().rstrip(";").split()),
        count=1,
        flags=re.IGNORECASE,
    )


def table_columns(connection, table: str) -> List[str]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT a.attname
        FROM pg_catalog.pg_attribute a
        WHERE a.attrelid = %s::regclass
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """,
        (table,),
    )
    return [row[0].lower() for row in cursor.fetchall()]


def create_query_view(
    connection,
    query: str,
    relation_map: Mapping[str, str],
    join_key: str,
    view: str,
) -> Tuple[str, int]:
    logical = relation_names(query)
    if not logical:
        raise ValueError("The query has no relation")
    physical = [relation_map[name.lower()] for name in logical]
    selected = [
        "r0.%s AS %s" % (qident(join_key), qident(join_key)),
        "1::integer AS %s" % qident("__marginal"),
    ]
    used = {join_key.lower(), "__marginal"}
    for position, table in enumerate(physical):
        alias = "r%d" % position
        for column in table_columns(connection, table):
            if column in used:
                continue
            selected.append("%s.%s AS %s" % (alias, qident(column), qident(column)))
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


def restricted_ordering(
    ordering: Mapping[str, Sequence[str]],
    active_missing: Sequence[str],
    available: Iterable[str],
) -> Dict[str, List[str]]:
    columns = {column.lower() for column in available}
    result: Dict[str, List[str]] = {}
    for attribute in active_missing:
        separators = [
            value.lower()
            for value in ordering.get(attribute, ())
            if value.lower() in columns and value.lower() != attribute
        ]
        result[attribute] = separators or ["__marginal"]
    return result


def finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def set_result(
    connection,
    flat_query: str,
    view: str,
    ordering: Mapping[str, Sequence[str]],
    missing: Sequence[str],
    method: str,
) -> Dict[str, Any]:
    if method == "CAEX":
        meta = {"real": {"csv": [view], "table": [view]}}
        executor = CAEXSetExecutor(connection, meta, skip_prepare=True)
        executor.interval_mode = "delta"
        executor.interval_alpha = 0.05
        executor._ordering_T = dict(ordering)
        executor._missing_T = list(missing)
        metrics, _groups, _truth = evaluate_query_with_groups(
            executor, flat_query, flat_query, dict(ordering), list(missing)
        )
        return {
            "time_s": metrics["time_pred_s"],
            "delta_w": metrics.get("normalized_interval_width"),
            "result_rows": metrics.get("size_pred_all"),
        }
    direct = run_direct_per_tuple(
        connection,
        flat_query,
        view,
        view,
        list(missing),
        dict(ordering),
        return_groups=True,
    )
    if direct.get("error"):
        raise RuntimeError(direct["error"])
    return {
        "time_s": direct["sql_time_s"],
        "delta_w": direct.get("delta_w"),
        "result_rows": len(direct.get("groups", ())),
    }


def aggregate_result(
    connection,
    flat_query: str,
    view: str,
    ordering: Mapping[str, Sequence[str]],
    missing: Sequence[str],
    direct: bool,
) -> Dict[str, Any]:
    sql, group_columns = aggregate_sql(
        flat_query, view, ordering, missing, direct=direct
    )
    cursor = connection.cursor()
    started = time.perf_counter()
    cursor.execute(sql)
    rows = cursor.fetchall()
    elapsed = time.perf_counter() - started
    estimates = []
    for row in rows:
        estimate = row[group_columns]
        stderr = row[group_columns + 1]
        if estimate is None:
            continue
        estimate = float(estimate)
        stderr = float(stderr or 0.0)
        estimates.append((estimate, stderr))
    widths = [3.92 * stderr for estimate, stderr in estimates]
    mean_estimate = (
        sum(abs(estimate) for estimate, _stderr in estimates) / len(estimates)
        if estimates
        else 0.0
    )
    mean_width = sum(widths) / len(widths) if widths else 0.0
    return {
        "time_s": elapsed,
        "delta_w": mean_width / mean_estimate if mean_estimate > 1e-12 else None,
        "result_rows": len(estimates),
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_type",
        "dataset",
        "query_index",
        "relation_count",
        "method",
        "time_s",
        "delta_w",
        "result_rows",
        "status",
        "query",
        "error",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: finite(row.get(field)) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_factorizable_10_queries.json")
    parser.add_argument(
        "--output",
        default="psql_results/real_factorizable_10/real_factorizable_10_results.csv",
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    datasets = json.load(open(args.config))
    output = Path(args.output)
    rows: List[Dict[str, Any]] = []
    connection = psycopg2.connect(**CONN)
    try:
        for dataset_key, dataset in datasets.items():
            relation_map = {
                relation["name"].lower(): relation["table"]
                for relation in dataset["relations"]
            }
            for relation in dataset["relations"]:
                _load_table(connection, relation["csv"], relation["table"], force=True)
            for query_type, config_key in (
                ("set", "set_queries"),
                ("aggregation", "aggregate_queries"),
            ):
                for query_index, query in enumerate(dataset[config_key], 1):
                    view = "real_%s_%s_q%d" % (dataset_key, query_type, query_index)
                    flat_query, relation_count = create_query_view(
                        connection,
                        query,
                        relation_map,
                        dataset["join_key"].lower(),
                        view,
                    )
                    inspector = NativeMCDB(connection)
                    missing = list(inspector.missing_attributes(view))
                    ordering = restricted_ordering(
                        {
                            key.lower(): [value.lower() for value in values]
                            for key, values in dataset["orderings"].items()
                        },
                        missing,
                        table_columns(connection, view),
                    )
                    base = {
                        "query_type": query_type,
                        "dataset": dataset_key,
                        "query_index": query_index,
                        "relation_count": relation_count,
                        "query": query,
                    }
                    for method in ("CAEX", "CADE"):
                        set_timeout(connection, args.timeout)
                        try:
                            if query_type == "set":
                                result = set_result(
                                    connection, flat_query, view, ordering, missing, method
                                )
                            else:
                                result = aggregate_result(
                                    connection,
                                    flat_query,
                                    view,
                                    ordering,
                                    missing,
                                    direct=(method == "CADE"),
                                )
                            rows.append(
                                {
                                    **base,
                                    **result,
                                    "method": method,
                                    "status": "ok",
                                    "error": "",
                                }
                            )
                        except Exception as error:
                            connection.rollback()
                            rows.append(
                                {
                                    **base,
                                    "method": method,
                                    "time_s": args.timeout,
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
                            "%s %s Q%d %s: %s"
                            % (
                                dataset_key,
                                query_type,
                                query_index,
                                method,
                                rows[-1]["status"],
                            ),
                            flush=True,
                        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
