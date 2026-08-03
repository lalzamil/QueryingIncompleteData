#!/usr/bin/env python3
"""Create a lossless four-relation Bank workload without changing missingness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("data")
INCOMPLETE_DIR = ROOT / "Mnar1FourRelationsData"
COMPLETE_DIR = ROOT / "CompleteFourRelationsData"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "bank_four_relation_queries.json"
JOIN_KEY = "customer_id"

PARTITIONS: Dict[str, List[str]] = {
    "r1": [
        "job", "marital", "housing", "loan", "contact", "day", "month",
        "duration", "pdays", "previous", "poutcome", "y",
    ],
    "r2": ["age", "balance"],
    "r3": ["campaign", "default"],
    "r4": ["education"],
}


def project_csv(source: Path, destination: Path, attributes: Iterable[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    attributes = list(attributes)
    with source.open(newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no header")
        absent = [attribute for attribute in attributes if attribute not in reader.fieldnames]
        if absent:
            raise ValueError(f"{source} is missing columns: {absent}")
        with destination.open("w", newline="") as destination_handle:
            writer = csv.DictWriter(destination_handle, fieldnames=[JOIN_KEY] + attributes)
            writer.writeheader()
            for customer_id, row in enumerate(reader, 1):
                projected = {JOIN_KEY: customer_id}
                projected.update({attribute: row[attribute] for attribute in attributes})
                writer.writerow(projected)


def relation_paths(directory: Path, prefix: str) -> List[str]:
    return [str(directory / f"{prefix}_{relation}.csv") for relation in PARTITIONS]


def from_clause(paths: List[str]) -> str:
    clause = paths[0]
    for path in paths[1:]:
        clause += f" JOIN {path} USING ({JOIN_KEY})"
    return clause


def set_queries(paths: List[str]) -> List[str]:
    source = from_clause(paths)
    return [
        f"SELECT day FROM {source} WHERE balance > 444 AND campaign > 1 AND education = 'secondary'",
        f"SELECT housing, contact FROM {source} WHERE age > 39 AND \"default\" = 'no' AND education != 'primary' GROUP BY housing, contact",
        f"SELECT duration FROM {source} WHERE balance < 0 AND campaign > 2 AND education = 'secondary'",
        f"SELECT day, duration FROM {source} WHERE age > 50 AND campaign > 2 AND education = 'tertiary' GROUP BY day, duration",
        f"SELECT contact, housing FROM {source} WHERE balance > 1000 AND campaign > 1 AND education != 'unknown' GROUP BY contact, housing",
    ]


def aggregate_queries(paths: List[str]) -> List[str]:
    source = from_clause(paths)
    return [
        f"SELECT job, AVG(duration) AS avg_duration FROM {source} WHERE balance > 0 AND campaign > 1 AND education = 'secondary' GROUP BY job",
        f"SELECT poutcome, AVG(day) AS avg_day FROM {source} WHERE age > 39 AND \"default\" = 'no' AND education != 'primary' GROUP BY poutcome",
        f"SELECT marital, AVG(duration) AS avg_duration FROM {source} WHERE balance < 0 AND campaign > 1 AND education = 'secondary' GROUP BY marital",
        f"SELECT y, AVG(day) AS avg_day FROM {source} WHERE age > 50 AND campaign > 2 AND education = 'tertiary' GROUP BY y",
        f"SELECT loan, AVG(duration) AS avg_duration FROM {source} WHERE balance > 1000 AND campaign > 1 AND education != 'unknown' GROUP BY loan",
    ]


def main() -> None:
    source_complete = ROOT / "rwDatasets" / "bank_complete.csv"
    complete_paths = relation_paths(COMPLETE_DIR, "bank_complete")
    for (relation, attributes), destination in zip(PARTITIONS.items(), complete_paths):
        project_csv(source_complete, Path(destination), attributes)

    config = {
        "dataset": "bank",
        "join_key": JOIN_KEY,
        "set_config_key": "bank_manr1_set",
        "factor_file": (
            "data/MNAR1Data/bank/"
            "bank_mnar1_minimal_separators_Xi.csv"
        ),
        "aggregate_factor_file": (
            "data/MNAR1Data/bank/"
            "bank_agg_mnar1_minimal_separators_Xi.csv"
        ),
        "partitions": PARTITIONS,
        "rates": {},
    }
    for rate in (5, 10, 20):
        source = ROOT / "MNAR1Data" / "bank" / f"bank_mnar1_{rate}.csv"
        paths = relation_paths(INCOMPLETE_DIR, f"bank_mnar1_{rate}")
        for (relation, attributes), destination in zip(PARTITIONS.items(), paths):
            project_csv(source, Path(destination), attributes)
        aggregate_source = (
            ROOT / "MNAR1Data" / "bank" / f"bank_agg_mnar1_{rate}.csv"
        )
        aggregate_paths = relation_paths(
            INCOMPLETE_DIR,
            f"bank_agg_mnar1_{rate}",
        )
        for (relation, attributes), destination in zip(
            PARTITIONS.items(),
            aggregate_paths,
        ):
            project_csv(aggregate_source, Path(destination), attributes)
        config["rates"][str(rate)] = {
            "csv": paths,
            "table": [f"bank4_mnar_{rate}_{relation}" for relation in PARTITIONS],
            "factor_csv": str(source),
            "factor_table": f"bank4_mnar_{rate}_factor",
            "aggregate_csv": aggregate_paths,
            "aggregate_table": [
                f"bank4_agg_mnar_{rate}_{relation}"
                for relation in PARTITIONS
            ],
            "aggregate_factor_csv": str(aggregate_source),
            "aggregate_factor_table": f"bank4_agg_mnar_{rate}_factor",
            "complete_csv": complete_paths,
            "complete_table": [f"bank4_complete_{relation}" for relation in PARTITIONS],
            "set_queries": set_queries(paths),
            "aggregate_queries": aggregate_queries(aggregate_paths),
        }

    with CONFIG_PATH.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(CONFIG_PATH)


if __name__ == "__main__":
    main()
