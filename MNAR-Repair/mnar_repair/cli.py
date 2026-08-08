"""Command-line interface for MNAR-Repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .pipeline import repair_relation

OUTPUT_NAMES = (
    "repaired_relation.csv",
    "repaired_mgraph.json",
    "repair_set.json",
)


def _json_object(path: Path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and impute an MNAR-Repair of an incomplete relation."
    )
    parser.add_argument("relation", type=Path, help="Incomplete relation in CSV format.")
    parser.add_argument("mgraph", type=Path, help="M-graph metadata in JSON format.")
    parser.add_argument("--costs", type=Path, help="Optional repair costs in JSON format.")
    parser.add_argument(
        "--imputer",
        choices=("mbi", "simple"),
        default="mbi",
        help="Imputation method for the selected attributes (default: mbi).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repair_output"),
        help="Directory for the repaired relation and metadata.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace MNAR-Repair output files that already exist.",
    )
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    relation = pd.read_csv(arguments.relation)
    mgraph = _json_object(arguments.mgraph)
    costs = _json_object(arguments.costs) if arguments.costs else None

    output_paths = [arguments.output_dir / name for name in OUTPUT_NAMES]
    existing = [path for path in output_paths if path.exists()]
    if existing and not arguments.force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing MNAR-Repair output: {names}. "
            "Use --force to replace it."
        )

    result = repair_relation(
        relation,
        mgraph,
        costs=costs,
        imputer=arguments.imputer,
        random_state=arguments.random_state,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    result.relation.to_csv(output_paths[0], index=False)
    with output_paths[1].open("w", encoding="utf-8") as handle:
        json.dump(result.mgraph, handle, indent=2)
        handle.write("\n")
    with output_paths[2].open("w", encoding="utf-8") as handle:
        json.dump(list(result.repair_set), handle, indent=2)
        handle.write("\n")

    print("Repair set:", ", ".join(result.repair_set) or "empty")
    for path in output_paths:
        print(f"Wrote {path}")
    return 0
