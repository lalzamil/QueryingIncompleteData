"""Run CAEX on the selected full-data MCAR or MAR queries."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import psycopg2

from RunQEMCARMARSelectedFullData import build_selected_configs
from RunSectionComparisonsFullData import (
    GROUPS,
    block_for_rate,
    prepare_block,
    read_csv,
    run_agg_block,
    run_set_block,
    set_timeout,
    write_csv,
)


METHOD = "CAEX"


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
    parser.add_argument("--query-indices", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()

    set_config, aggregate_config = build_selected_configs(args.mechanism)
    datasets = [value.strip().lower() for value in args.datasets.split(",") if value.strip()]
    rates = [int(value) for value in args.rates.split(",") if value.strip()]
    workloads = [value.strip().lower() for value in args.workloads.split(",") if value.strip()]
    requested_queries = {int(value) for value in args.query_indices.split(",") if value.strip()}
    results: List[Dict[str, Any]] = read_csv(args.output) if args.resume else []

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
                    completed = {
                        int(row["query_index"])
                        for row in results
                        if row.get("dataset") == dataset
                        and row.get("workload") == workload
                        and int(row.get("rate", -1)) == rate
                        and row.get("method") == METHOD
                    }
                    if len(completed & set(range(1, target_count + 1))) == target_count:
                        print(
                            "Skipping completed %s %s at %d%%" % (dataset, workload, rate),
                            flush=True,
                        )
                        continue

                    group = set_config[set_group] if workload == "set" else aggregate_config[aggregate_group]
                    block, meta = block_for_rate(group, rate)
                    print(
                        "Preparing CAEX %s %s %s at %d%%" % (args.mechanism, dataset, workload, rate),
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
                        str(key).lower(): tuple(str(value).lower() for value in values)
                        for key, values in meta.get("factor_map", {}).items()
                    }

                    def checkpoint(partial_rows):
                        decorated = []
                        for partial in partial_rows:
                            row = dict(partial)
                            row["mechanism"] = args.mechanism
                            row["row_limit"] = 0
                            row["null_semantics"] = "nonrepeating"
                            decorated.append(row)
                        write_csv(args.output, results + decorated)

                    skipped = completed | set(range(target_count + 1, 11))
                    if requested_queries:
                        skipped |= set(range(1, target_count + 1)) - requested_queries
                    if workload == "set":
                        block_rows = run_set_block(
                            conn,
                            dataset,
                            block,
                            meta,
                            prepared,
                            783,
                            args.timeout,
                            0.0,
                            0.0,
                            None,
                            {},
                            target_count,
                            skipped,
                            checkpoint,
                            (METHOD,),
                        )
                    else:
                        block_rows = run_agg_block(
                            conn,
                            dataset,
                            block,
                            meta,
                            prepared,
                            factor_map,
                            783,
                            args.timeout,
                            0.0,
                            0.0,
                            None,
                            {},
                            target_count,
                            skipped,
                            checkpoint,
                            (METHOD,),
                        )
                    for row in block_rows:
                        row["mechanism"] = args.mechanism
                        row["row_limit"] = 0
                        row["null_semantics"] = "nonrepeating"
                    results.extend(block_rows)
                    write_csv(args.output, results)
    finally:
        conn.close()

    write_csv(args.output, results)
    print("Saved %d CAEX measurements to %s" % (len(results), args.output), flush=True)


if __name__ == "__main__":
    main()
