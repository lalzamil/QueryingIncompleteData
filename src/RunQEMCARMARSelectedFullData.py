"""Run QE on the MCAR or MAR queries defined in the JSON files.

The four non-aggregation queries come from mcar_set_queries.json or
mar_set_queries.json. The five aggregation queries come from all_queries.json.
An internal join query prepares the normalized relations but is not evaluated.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from typing import Any, Dict, List, Mapping, Sequence

import psycopg2

import RunQEFromFactorDistributionsFullData as qe
from RunSectionComparisonsFullData import (
    GROUPS,
    block_for_rate,
    prepare_block,
    query_attributes,
    set_timeout,
    write_csv,
)


SOURCE_GROUPS = {
    "MCAR": {
        "bank": ("bank_mcar_set", "bank_mcar"),
        "nyc": ("nyc_mcar_set", "nyc_mcar"),
        "bitcoin": ("bitcoin_mcar_set", "bit_macr"),
    },
    "MAR": {
        "bank": ("bank_mar_set", "bank_mar"),
        "nyc": ("nyc_mar_set", "nyc_mar"),
        "bitcoin": ("bitcoin_mar_set", "bit_mar"),
    },
}


def rate_of(name: str) -> int:
    match = re.search(r"(5|10|20)%?$", name)
    if not match:
        raise ValueError("Cannot infer missingness rate from %s" % name)
    return int(match.group(1))


def block_at_rate(group: Mapping[str, Any], rate: int) -> Mapping[str, Any]:
    for name, meta in group.items():
        if rate_of(name) == rate:
            return meta
    raise KeyError("No block found at %d%%" % rate)


def source_token(query: str) -> str:
    match = re.search(r"\bFROM\s+(\S+)", query, re.IGNORECASE)
    if not match:
        raise ValueError("Query has no FROM token: %s" % query)
    return match.group(1)


def replace_source(query: str, replacement: str) -> str:
    token = source_token(query)
    return re.sub(re.escape(token), replacement, query, count=1, flags=re.IGNORECASE)


def remove_avg(query: str) -> str:
    rewritten, count = re.subn(
        r"\bAVG\s*\(\s*([A-Za-z_]\w*)\s*\)",
        r"\1",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise ValueError("Expected one AVG expression in %s" % query)
    return rewritten


def attribute_ordering(
    queries: Sequence[str],
    causes: Sequence[str],
) -> Dict[str, List[str]]:
    if not causes:
        return {}
    attributes = []
    for query in queries:
        for attribute in query_attributes(query):
            value = str(attribute).lower()
            if value not in attributes and value not in causes:
                attributes.append(value)
    return {attribute: list(causes) for attribute in attributes}


def duplicate_for_runner(queries: Sequence[str]) -> List[str]:
    if len(queries) != 5:
        raise ValueError("Expected five selected queries")
    return list(queries) + list(queries)


def build_selected_configs(mechanism: str):
    with open("configs/all_queries.json") as handle:
        aggregate_source = json.load(handle)
    set_path = "configs/mcar_set_queries.json" if mechanism == "MCAR" else "configs/mar_set_queries.json"
    with open(set_path) as handle:
        set_source = json.load(handle)

    set_config: Dict[str, Any] = {}
    aggregate_config: Dict[str, Any] = {}
    for dataset, (set_target, aggregate_target) in GROUPS.items():
        source_set_group, source_aggregate_group = SOURCE_GROUPS[mechanism][dataset]
        set_config[set_target] = {}
        aggregate_config[aggregate_target] = {}
        for rate in (5, 10, 20):
            original_set = copy.deepcopy(
                block_at_rate(set_source[source_set_group], rate)
            )
            original_aggregate = copy.deepcopy(
                block_at_rate(aggregate_source[source_aggregate_group], rate)
            )
            aggregate_queries = list(original_aggregate.get("queries", []))
            set_queries = list(original_set.get("queries", []))
            if len(aggregate_queries) != 5:
                raise ValueError("%s %s aggregation must have five queries" % (dataset, rate))
            if len(set_queries) != 4:
                raise ValueError("%s %s non-aggregation must have four source queries" % (dataset, rate))

            aggregate_meta = copy.deepcopy(original_aggregate)
            aggregate_meta["queries"] = duplicate_for_runner(aggregate_queries)
            causes_by_relation = original_aggregate.get("Cause", [[], [], []])
            aggregate_causes = [
                str(value).lower()
                for value in (causes_by_relation[0] if causes_by_relation else [])
            ]
            aggregate_meta["factor_map"] = attribute_ordering(
                aggregate_queries, aggregate_causes
            )
            aggregate_config[aggregate_target][
                "%s_mnar_%d" % (dataset, rate)
            ] = aggregate_meta

            first_source = original_aggregate["csv"][0]
            normalized_set_queries = [
                replace_source(query, first_source) for query in set_queries
            ]
            join_query = remove_avg(aggregate_queries[4])
            selected_set_queries = normalized_set_queries + [join_query]
            set_meta = copy.deepcopy(original_aggregate)
            set_meta["queries"] = duplicate_for_runner(selected_set_queries)
            set_meta["missing_attrs_single"] = original_set.get(
                "missing_attrs_single", []
            )
            set_meta["ordering_single"] = original_set.get(
                "ordering_single", {}
            )
            left_causes = [
                str(value).lower()
                for value in (
                    causes_by_relation[1]
                    if len(causes_by_relation) > 1 else []
                )
            ]
            right_causes = [
                str(value).lower()
                for value in (
                    causes_by_relation[2]
                    if len(causes_by_relation) > 2 else []
                )
            ]
            set_meta["missing_attrs_T"] = []
            set_meta["missing_attrs_S"] = []
            set_meta["ordering_T"] = attribute_ordering(
                [join_query], left_causes
            )
            set_meta["ordering_S"] = attribute_ordering(
                [join_query], right_causes
            )
            set_config[set_target][
                "%s_mnar_%d" % (dataset, rate)
            ] = set_meta
    return set_config, aggregate_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=("MCAR", "MAR"), required=True)
    parser.add_argument("--db-host", default=os.environ.get("PGHOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("PGPORT", "5433")))
    parser.add_argument("--db-name", default=os.environ.get("PGDATABASE", "mydb"))
    parser.add_argument("--db-user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--db-password", default=os.environ.get("PGPASSWORD", ""))
    parser.add_argument("--datasets", default="bank,nyc,bitcoin")
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--workloads", default="set,aggregate")
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()
    if args.h != 783:
        raise ValueError("This comparison must use H=783")

    set_config, aggregate_config = build_selected_configs(args.mechanism)
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
        [dict(row) for row in qe.read_rows(args.output)]
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
            set_group, aggregate_group = GROUPS[dataset]
            for rate in rates:
                for workload in workloads:
                    target_count = 4 if workload == "set" else 5
                    target_queries = set(range(1, target_count + 1))
                    completed = {
                        int(row["query_index"])
                        for row in results
                        if row.get("dataset") == dataset
                        and row.get("workload") == workload
                        and int(row.get("rate", -1)) == rate
                        and row.get("method") == qe.METHOD
                    }
                    if len(completed & target_queries) == target_count:
                        print(
                            "Skipping completed %s %s at %d%%"
                            % (dataset, workload, rate),
                            flush=True,
                        )
                        continue
                    group = (
                        set_config[set_group]
                        if workload == "set"
                        else aggregate_config[aggregate_group]
                    )
                    block, meta = block_for_rate(group, rate)
                    print(
                        "Preparing QE %s %s %s at %d%%"
                        % (args.mechanism, dataset, workload, rate),
                        flush=True,
                    )
                    set_timeout(conn, 0)
                    prepared = prepare_block(
                        conn,
                        meta,
                        dataset,
                        block,
                        workload,
                        0,
                        args.seed,
                        force_reload=args.force_reload,
                    )
                    factor_map = {
                        str(key).lower(): tuple(
                            str(value).lower() for value in values
                        )
                        for key, values in meta.get("factor_map", {}).items()
                    }
                    engine, bundle_mapping, preprocessing_s, projection_s, fallback = qe.build_qe_distributions(
                        conn,
                        meta,
                        prepared,
                        workload,
                        factor_map,
                        args.h,
                        args.seed,
                        strict=False,
                    )
                    rewriter = qe.TimedRewrittenLikeApx(engine)
                    mappings = qe.rewritten_mapping(rewriter, bundle_mapping)
                    print(
                        "QE factorization %.3fs; relation projection %.3fs excluded"
                        % (preprocessing_s, projection_s),
                        flush=True,
                    )

                    def checkpoint(partial_rows):
                        decorated = []
                        for partial in partial_rows:
                            row = dict(partial)
                            row["mechanism"] = args.mechanism
                            row["row_limit"] = 0
                            row["total_factorization_s"] = preprocessing_s
                            row["total_encoding_s"] = 0.0
                            row["factor_fallback_symbols"] = fallback
                            row["null_semantics"] = "nonrepeating"
                            decorated.append(row)
                        write_csv(args.output, results + decorated)

                    block_rows = qe.run_queries(
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
                        preprocessing_s * (10.0 / target_count),
                        completed | set(range(target_count + 1, 11)),
                        checkpoint,
                    )
                    for row in block_rows:
                        row["mechanism"] = args.mechanism
                        row["row_limit"] = 0
                        row["total_factorization_s"] = preprocessing_s
                        row["total_encoding_s"] = 0.0
                        row["factor_fallback_symbols"] = fallback
                        row["null_semantics"] = "nonrepeating"
                    results.extend(block_rows)
                    write_csv(args.output, results)
    finally:
        conn.close()
    write_csv(args.output, results)
    print(
        "Saved %d QE measurements to %s" % (len(results), args.output),
        flush=True,
    )


if __name__ == "__main__":
    main()
