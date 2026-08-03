"""Run the PostgreSQL-native MCDB implementation on experiment workloads.

The source relations, random draws, sampled-value arrays, and query operators
remain in PostgreSQL. Python selects a workload, issues SQL statements, and
records timings. Input preparation is reported separately from MCDB sampling,
encoding, and query evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from MCDBPostgresNative import BundleRelation, NativeMCDB, relation_mapping


def find_block(config: Mapping[str, object], block_name: str):
    matches = []
    for group_name, group in config.items():
        if not isinstance(group, dict):
            continue
        for candidate, metadata in group.items():
            if candidate == block_name and isinstance(metadata, dict):
                matches.append((group_name, metadata))
    if not matches:
        raise KeyError("No workload block named %s" % block_name)
    if len(matches) > 1:
        raise ValueError("Workload block %s occurs in multiple groups" % block_name)
    return matches[0]


def parse_query_indices(value: str, query_count: int) -> List[int]:
    if value.lower() == "all":
        return list(range(query_count))
    result = []
    for token in value.split(","):
        position = int(token.strip())
        if position < 1 or position > query_count:
            raise ValueError("Query index %d is outside 1..%d" % (position, query_count))
        result.append(position - 1)
    return result


def normalized_ordering(value: object) -> Dict[str, Tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(attribute).lower(): tuple(str(item).lower() for item in dependencies)
        for attribute, dependencies in value.items()
    }


def resolve_table(engine: NativeMCDB, configured_name: str) -> str:
    cursor = engine.connection.cursor()
    cursor.execute(
        "SELECT c.relname FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = ANY (current_schemas(false)) "
        "AND lower(c.relname) = lower(%s) "
        "AND c.relkind IN ('r', 'p', 'v', 'm') "
        "ORDER BY CASE WHEN c.relname = %s THEN 0 ELSE 1 END LIMIT 1",
        (configured_name, configured_name),
    )
    row = cursor.fetchone()
    if not row:
        raise ValueError("PostgreSQL relation %s does not exist" % configured_name)
    return row[0]


def relation_settings(metadata: Mapping[str, object], position: int):
    if position == 0:
        missing_key = "missing_attrs_single"
        ordering_key = "ordering_single"
    elif position == 1:
        missing_key = "missing_attrs_T"
        ordering_key = "ordering_T"
    else:
        missing_key = "missing_attrs_S"
        ordering_key = "ordering_S"
    missing = tuple(str(value).lower() for value in metadata.get(missing_key, []))
    ordering = normalized_ordering(metadata.get(ordering_key, {}))
    return missing, ordering


def add_relation_tokens(target: Dict[str, BundleRelation],
                        bundle: BundleRelation,
                        configured_table: str,
                        csv_token: str) -> None:
    target.update(relation_mapping(bundle, configured_table, csv_token))
    target.update(relation_mapping(bundle, os.path.basename(csv_token)))


def unresolved_text(bundles: Mapping[int, BundleRelation]) -> str:
    values = []
    for position, bundle in sorted(bundles.items()):
        for attribute, count in sorted(bundle.unresolved_draws.items()):
            if count:
                values.append("R%d.%s=%d" % (position, attribute, count))
    return ";".join(values)


def write_rows(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnar_set_queries.json")
    parser.add_argument("--block", required=True)
    parser.add_argument("--queries", default="all",
                        help="One-based comma-separated positions, or 'all'")
    parser.add_argument("--h", type=int, default=100)
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--allow-marginal-fallback", action="store_true")
    parser.add_argument("--statement-timeout", type=int, default=300)
    parser.add_argument("--work-mem", default="256MB")
    parser.add_argument("--output", default="mcdb_postgres_native_results.csv")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = json.load(handle)
    group_name, metadata = find_block(config, args.block)
    queries = list(metadata.get("queries", []))
    query_indices = parse_query_indices(args.queries, len(queries))
    active_positions = set()
    for query_index in query_indices:
        if " JOIN " in (" " + queries[query_index].upper() + " "):
            active_positions.update((1, 2))
        else:
            active_positions.add(0)
    configured_tables = list(metadata.get("table", []))
    csv_tokens = list(metadata.get("csv", []))
    if not configured_tables or len(configured_tables) != len(csv_tokens):
        raise ValueError("The workload block requires matching table and csv lists")

    engine = NativeMCDB.connect()
    cursor = engine.connection.cursor()
    cursor.execute("SELECT set_config('statement_timeout', %s, false)",
                   (str(args.statement_timeout * 1000),))
    cursor.execute("SELECT set_config('work_mem', %s, false)", (args.work_mem,))

    input_started = time.perf_counter()
    working_tables = {}
    actual_tables = {}
    for position in sorted(active_positions):
        configured_table = configured_tables[position]
        actual_table = resolve_table(engine, configured_table)
        actual_tables[position] = actual_table
        working_tables[position] = engine.prepare_relation(
            actual_table,
            "native_%s_r%d" % (args.block.lower(), position),
        )
    input_s = time.perf_counter() - input_started

    bundles = {}
    relations: Dict[str, BundleRelation] = {}
    encode_started = time.perf_counter()
    encode_by_relation = {}
    for position, working_table in sorted(working_tables.items()):
        configured_missing, ordering = relation_settings(metadata, position)
        discovered_missing = engine.missing_attributes(working_table)
        missing = tuple(dict.fromkeys(configured_missing + discovered_missing))
        relation_started = time.perf_counter()
        bundle = engine.create_bundle(
            table=working_table,
            missing_attributes=missing,
            ordering=ordering,
            h=args.h,
            seed=args.seed,
            prefix="native_%s_h%d_r%d" % (args.block.lower(), args.h, position),
            strict=not args.allow_marginal_fallback,
            n_bins=args.n_bins,
        )
        relation_elapsed = time.perf_counter() - relation_started
        bundles[position] = bundle
        encode_by_relation[position] = relation_elapsed
        add_relation_tokens(
            relations, bundle, configured_tables[position], csv_tokens[position]
        )
    encode_s = time.perf_counter() - encode_started

    rows = []
    fallback = unresolved_text(bundles)
    print("block=%s H=%d input=%.3fs encode=%.3fs fallback=%s" % (
        args.block, args.h, input_s, encode_s, fallback or "none"
    ), flush=True)
    for query_index in query_indices:
        query = queries[query_index]
        result = engine.evaluate_summary(query, relations)
        row = {
            "config": args.config,
            "group": group_name,
            "block": args.block,
            "query_index": query_index + 1,
            "query": query,
            "query_kind": "aggregation" if result.query.is_aggregation else "set",
            "h": args.h,
            "seed": args.seed,
            "n_bins": args.n_bins,
            "marginal_fallback_allowed": args.allow_marginal_fallback,
            "fallback_draw_groups": fallback,
            "input_prepare_s": input_s,
            "sample_encode_s": encode_s,
            "query_s": result.elapsed_s,
            "standalone_total_s": encode_s + result.elapsed_s,
            "summary_rows": len(result.summary_rows),
            "source_tables": ";".join(
                "R%d=%s" % item for item in sorted(actual_tables.items())
            ),
            "relation_encode_s": ";".join(
                "R%d=%.6f" % item for item in sorted(encode_by_relation.items())
            ),
        }
        rows.append(row)
        print("Q%d %-11s query=%.3fs rows=%d" % (
            query_index + 1, row["query_kind"], result.elapsed_s,
            len(result.summary_rows),
        ), flush=True)

    write_rows(args.output, rows)
    print("saved %s" % args.output, flush=True)
    engine.close()


if __name__ == "__main__":
    main()
