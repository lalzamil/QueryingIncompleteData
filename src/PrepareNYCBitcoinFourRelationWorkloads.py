#!/usr/bin/env python3
"""Create lossless four-relation NYC and Bitcoin workloads without changing missingness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("data")
INCOMPLETE_DIR = ROOT / "Mnar1FourRelationsData"
COMPLETE_DIR = ROOT / "CompleteFourRelationsData"
JOIN_KEY = "tuple_id"

SPECS = {
    "nyc": {
        "complete": ROOT / "rwDatasets" / "nyc_complete.csv",
        "incomplete": ROOT / "MNAR1Data" / "nyc" / "nyc_mnar1_{rate}.csv",
        "aggregate_incomplete": (
            ROOT / "MNAR1Data" / "nyc" / "nyc_agg_mnar1_{rate}.csv"
        ),
        "source_key": "id",
        "set_config_key": "nyc_manr1_set",
        "factor_file": "data/MNAR1Data/nyc/nyc_mnar1_minimal_separators_Xi.csv",
        "aggregate_factor_file": (
            "data/MNAR1Data/nyc/"
            "nyc_agg_mnar1_minimal_separators_Xi.csv"
        ),
        "partitions": {
            "r1": ["vendor_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"],
            "r2": ["passenger_count"],
            "r3": ["store_and_fwd_flag"],
            "r4": ["id", "pickup_datetime", "dropoff_datetime"],
        },
    },
    "bitcoin": {
        "complete": ROOT / "rwDatasets" / "BitcoinHeistData_complete.csv",
        "incomplete": ROOT / "MNAR1Data" / "BitcoinHeistData" / "BitcoinHeistData_mnar1_{rate}.csv",
        "aggregate_incomplete": (
            ROOT
            / "MNAR1Data"
            / "BitcoinHeistData"
            / "BitcoinHeistData_agg_mnar1_{rate}.csv"
        ),
        "source_key": None,
        "set_config_key": "bit_manr1_set",
        "factor_file": "data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_mnar1_minimal_separators_Xi.csv",
        "aggregate_factor_file": (
            "data/MNAR1Data/BitcoinHeistData/"
            "BitcoinHeistData_agg_mnar1_minimal_separators_Xi.csv"
        ),
        "partitions": {
            "r1": ["label", "weight", "day", "length", "neighbors", "count", "looped"],
            "r2": ["income"],
            "r3": ["year"],
            "r4": ["address"],
        },
    },
}


def stable_score(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def selected_rows(
    source: Path,
    key_column: Optional[str],
    limit: int,
    seed: int,
) -> Dict[int, str]:
    if limit <= 0:
        selected: Dict[int, str] = {}
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{source} has no header")
            if key_column is not None and key_column not in reader.fieldnames:
                raise ValueError(f"{source} does not contain {key_column}")
            for row_index, row in enumerate(reader, 1):
                selected[row_index] = (
                    row[key_column] if key_column is not None else str(row_index)
                )
        if len(set(selected.values())) != len(selected):
            raise ValueError(f"{key_column} is not unique in {source}")
        return selected
    heap: List[Tuple[int, int, str]] = []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no header")
        if key_column is not None and key_column not in reader.fieldnames:
            raise ValueError(f"{source} does not contain {key_column}")
        for row_index, row in enumerate(reader, 1):
            key = row[key_column] if key_column is not None else str(row_index)
            score = stable_score(key, seed)
            candidate = (-score, -row_index, key)
            if len(heap) < limit:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    selected = { -row_index: key for _score, row_index, key in heap }
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} rows from {source}, found {len(selected)}")
    if len(set(selected.values())) != len(selected):
        raise ValueError(f"{key_column} is not unique in the selected rows of {source}")
    return selected


def project_partitions(source: Path, destinations: Sequence[Path], partitions: Mapping[str, Sequence[str]],
                       source_key: Optional[str], selected: Mapping[int, str]) -> None:
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    handles = [destination.open("w", newline="") for destination in destinations]
    try:
        with source.open(newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise ValueError(f"{source} has no header")
            requested = [attribute for values in partitions.values() for attribute in values]
            absent = [attribute for attribute in requested if attribute not in reader.fieldnames]
            if absent:
                raise ValueError(f"{source} is missing columns: {absent}")
            writers = []
            for handle, attributes in zip(handles, partitions.values()):
                writer = csv.DictWriter(handle, fieldnames=[JOIN_KEY] + list(attributes))
                writer.writeheader()
                writers.append(writer)
            tuple_ids = {row_index: tuple_id for tuple_id, row_index in enumerate(sorted(selected), 1)}
            written = 0
            for row_index, row in enumerate(reader, 1):
                if row_index not in selected:
                    continue
                if (
                    source_key is not None
                    and row[source_key] != selected[row_index]
                ):
                    raise ValueError(f"Row alignment differs at row {row_index} in {source}")
                for writer, attributes in zip(writers, partitions.values()):
                    projected = {JOIN_KEY: tuple_ids[row_index]}
                    projected.update({attribute: row[attribute] for attribute in attributes})
                    writer.writerow(projected)
                written += 1
            if written != len(selected):
                raise ValueError(f"Expected {len(selected)} rows from {source}, wrote {written}")
    finally:
        for handle in handles:
            handle.close()


def relation_paths(directory: Path, prefix: str) -> List[Path]:
    return [directory / f"{prefix}_{relation}.csv" for relation in ("r1", "r2", "r3", "r4")]


def from_clause(paths: Iterable[Path]) -> str:
    values = [str(path) for path in paths]
    return values[0] + "".join(f" JOIN {path} USING ({JOIN_KEY})" for path in values[1:])


def set_queries(dataset: str, paths: Sequence[Path]) -> List[str]:
    source = from_clause(paths)
    if dataset == "nyc":
        return [
            f"SELECT vendor_id FROM {source} WHERE passenger_count = 1 AND store_and_fwd_flag = 'N'",
            f"SELECT pickup_longitude, passenger_count FROM {source} WHERE store_and_fwd_flag = 'N' GROUP BY pickup_longitude, passenger_count",
            f"SELECT trip_duration FROM {source} WHERE passenger_count > 1 AND store_and_fwd_flag = 'N'",
            f"SELECT dropoff_latitude FROM {source} WHERE passenger_count = 1 AND store_and_fwd_flag != 'Y'",
            f"SELECT pickup_longitude, trip_duration FROM {source} WHERE pickup_latitude > 40.5 AND store_and_fwd_flag = 'N' GROUP BY pickup_longitude, trip_duration",
        ]
    return [
        f"SELECT neighbors FROM {source} WHERE income > 100000000 AND year != 2016",
        f"SELECT weight, looped FROM {source} WHERE income > 130000000 AND year = 2011 GROUP BY weight, looped",
        f"SELECT length FROM {source} WHERE income > 100000000 AND year = 2014",
        f"SELECT count FROM {source} WHERE income > 100000000 AND year = 2016",
        f"SELECT looped, neighbors FROM {source} WHERE income > 100000000 AND year != 2016 GROUP BY looped, neighbors",
    ]


def prepare(dataset: str, row_limit: int, seed: int) -> Path:
    spec = SPECS[dataset]
    partitions = spec["partitions"]
    selected = selected_rows(spec["complete"], spec["source_key"], row_limit, seed)
    complete_paths = relation_paths(COMPLETE_DIR, f"{dataset}_complete")
    project_partitions(spec["complete"], complete_paths, partitions, spec["source_key"], selected)
    config = {
        "dataset": dataset,
        "join_key": JOIN_KEY,
        "set_config_key": spec["set_config_key"],
        "factor_file": spec["factor_file"],
        "aggregate_factor_file": spec["aggregate_factor_file"],
        "normalized_rows": len(selected),
        "partitions": partitions,
        "rates": {},
    }
    for rate in (5, 10, 20):
        source = Path(str(spec["incomplete"]).format(rate=rate))
        paths = relation_paths(INCOMPLETE_DIR, f"{dataset}_mnar1_{rate}")
        project_partitions(source, paths, partitions, spec["source_key"], selected)
        aggregate_source = Path(
            str(spec["aggregate_incomplete"]).format(rate=rate)
        )
        aggregate_paths = relation_paths(
            INCOMPLETE_DIR,
            f"{dataset}_agg_mnar1_{rate}",
        )
        project_partitions(
            aggregate_source,
            aggregate_paths,
            partitions,
            spec["source_key"],
            selected,
        )
        config["rates"][str(rate)] = {
            "csv": [str(path) for path in paths],
            "table": [f"{dataset}4_mnar_{rate}_r{position}" for position in range(1, 5)],
            "factor_csv": str(source),
            "factor_table": f"{dataset}4_mnar_{rate}_factor",
            "aggregate_csv": [str(path) for path in aggregate_paths],
            "aggregate_table": [
                f"{dataset}4_agg_mnar_{rate}_r{position}"
                for position in range(1, 5)
            ],
            "aggregate_factor_csv": str(aggregate_source),
            "aggregate_factor_table": (
                f"{dataset}4_agg_mnar_{rate}_factor"
            ),
            "complete_csv": [str(path) for path in complete_paths],
            "complete_table": [f"{dataset}4_complete_r{position}" for position in range(1, 5)],
            "set_queries": set_queries(dataset, paths),
        }
    config_path = REPOSITORY_ROOT / "configs" / f"{dataset}_four_relation_queries.json"
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="nyc,bitcoin")
    parser.add_argument(
        "--rows",
        type=int,
        default=20000,
        help="Number of rows to retain; use 0 for the complete relation",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        if dataset not in SPECS:
            raise ValueError(f"Unknown dataset: {dataset}")
        print(prepare(dataset, args.rows, args.seed))


if __name__ == "__main__":
    main()
