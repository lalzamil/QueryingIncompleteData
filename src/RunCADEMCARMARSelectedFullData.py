"""Run CADE with the selected five MCAR or MAR queries.

This wrapper supplies the mechanism-specific factorization maps to the current
full-data CADE runner. All remaining arguments are passed to that runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import RunSectionComparisonsFullData as runner


def write_factor_maps(mechanism: str, aggregate_path: str):
    with open(aggregate_path) as handle:
        config = json.load(handle)
    paths = {}
    for dataset, (_set_group, aggregate_group) in runner.GROUPS.items():
        blocks = config[aggregate_group]
        first = next(iter(blocks.values()))
        factor_map = first.get("factor_map", {})
        path = os.path.abspath(
            "selected5_%s_%s_separators.csv" % (mechanism.lower(), dataset)
        )
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("Y", "Xi_minimal"))
            writer.writeheader()
            for attribute, separators in sorted(factor_map.items()):
                writer.writerow(
                    {"Y": attribute, "Xi_minimal": repr(list(separators))}
                )
        paths[dataset] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mechanism", choices=("MCAR", "MAR", "MNAR"), required=True)
    parser.add_argument("--agg-config", required=True)
    known, remaining = parser.parse_known_args()
    runner.SEPARATOR_FILES = write_factor_maps(
        known.mechanism, known.agg_config
    )
    sys.argv = [sys.argv[0], "--agg-config", known.agg_config] + remaining
    runner.main()


if __name__ == "__main__":
    main()
