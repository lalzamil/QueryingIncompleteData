import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_KEYS = ("bank_mcar", "bank_mar", "nyc_mcar", "nyc_mar", "bit_macr", "bit_mar")
DATASET_NAMES = {"bank": "Bank Marketing", "nyc": "NYC Taxi Trips", "bit": "Bitcoin Heist"}
QUERY_PATTERN = re.compile(
    r"SELECT\s+AVG\((?P<attribute>\w+)\)\s+FROM\s+(?P<path>\S+)"
    r"(?:\s+WHERE\s+(?P<where>.+?))?"
    r"(?:\s+GROUP\s+BY\s+(?P<group>.+))?$",
    re.IGNORECASE,
)
CONDITION_PATTERN = re.compile(r"^\s*(?P<column>\w+)\s*(?P<operator><=|>=|!=|=|<|>)\s*(?P<value>.+?)\s*$")
Z_95 = 1.96


def parse_query(query: str) -> dict:
    match = QUERY_PATTERN.match(query.strip().rstrip(";"))
    if not match:
        raise ValueError(f"Unsupported query: {query}")
    result = match.groupdict()
    result["attribute"] = result["attribute"].lower()
    result["group"] = [value.strip().strip('"').lower() for value in result["group"].split(",")] if result["group"] else []
    return result


def conditions(where_clause: str | None) -> list[tuple[str, str, object]]:
    if not where_clause:
        return []
    result = []
    for expression in re.split(r"\s+AND\s+", where_clause, flags=re.IGNORECASE):
        match = CONDITION_PATTERN.match(expression)
        if not match:
            raise ValueError(f"Unsupported condition: {expression}")
        value = match.group("value").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            parsed_value = value[1:-1]
        else:
            parsed_value = float(value)
        result.append((match.group("column").lower(), match.group("operator"), parsed_value))
    return result


def filter_frame(frame: pd.DataFrame, parsed_conditions: list[tuple[str, str, object]]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, operator, value in parsed_conditions:
        series = frame[column]
        if operator == "=":
            current = series.eq(value)
        elif operator == "!=":
            current = series.ne(value)
        elif operator == ">":
            current = series.gt(value)
        elif operator == "<":
            current = series.lt(value)
        elif operator == ">=":
            current = series.ge(value)
        elif operator == "<=":
            current = series.le(value)
        else:
            raise ValueError(operator)
        mask &= current.fillna(False)
    return frame.loc[mask]


def normalized_width(estimate: float, variance: float) -> float:
    if not np.isfinite(estimate) or estimate == 0 or not np.isfinite(variance) or variance < 0:
        return np.nan
    return 2.0 * Z_95 * math.sqrt(variance) / abs(estimate)


def mcar_result(frame: pd.DataFrame, attribute: str, group_columns: list[str]) -> tuple[float, int]:
    widths = []
    groups = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [(None, frame)]
    for _, group in groups:
        values = pd.to_numeric(group[attribute], errors="coerce").dropna()
        if len(values) < 2:
            continue
        estimate = float(values.mean())
        variance = float(values.var(ddof=1) / len(values))
        width = normalized_width(estimate, variance)
        if np.isfinite(width):
            widths.append(width)
    return (float(np.mean(widths)), len(widths)) if widths else (np.nan, 0)


def mar_group_result(frame: pd.DataFrame, attribute: str, causes: list[str]) -> float:
    total = len(frame)
    if total == 0:
        return np.nan
    grouped = frame.groupby(causes, dropna=False, sort=False)[attribute].agg(
        n_all="size",
        n_obs="count",
        mean="mean",
        variance="var",
    )
    grouped = grouped[grouped["n_obs"].ge(2) & grouped["mean"].notna() & grouped["variance"].notna()]
    if grouped.empty or int(grouped["n_all"].sum()) != total:
        return np.nan
    weights = grouped["n_all"].astype(float) / float(total)
    estimate = float((grouped["mean"] * weights).sum())
    variance = float(((grouped["variance"] / grouped["n_obs"]) * weights.pow(2)).sum())
    return normalized_width(estimate, variance)


def mar_result(frame: pd.DataFrame, attribute: str, causes: list[str], group_columns: list[str]) -> tuple[float, int]:
    if not causes:
        return mcar_result(frame, attribute, group_columns)
    widths = []
    groups = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [(None, frame)]
    for _, group in groups:
        width = mar_group_result(group, attribute, causes)
        if np.isfinite(width):
            widths.append(width)
    return (float(np.mean(widths)), len(widths)) if widths else (np.nan, 0)


def rate_from_name(name: str) -> int:
    match = re.search(r"_(5|10|20)%?$", name)
    if not match:
        raise ValueError(f"Cannot determine missingness rate from {name}")
    return int(match.group(1))


def dataset_from_key(config_key: str) -> str:
    prefix = config_key.split("_", 1)[0]
    return DATASET_NAMES[prefix]


def run(config_path: Path, data_root: Path, output: Path) -> None:
    configuration = json.loads(config_path.read_text())
    rows = []
    for config_key in CONFIG_KEYS:
        mechanism = "MCAR" if "mcar" in config_key or "macr" in config_key else "MAR"
        dataset = dataset_from_key(config_key)
        for block_name, metadata in configuration[config_key].items():
            parsed_queries = [parse_query(query) for query in metadata["queries"]]
            paths = {query["path"] for query in parsed_queries}
            if len(paths) != 1:
                raise ValueError(f"{block_name} refers to multiple input relations")
            relative_path = next(iter(paths))
            input_path = data_root / relative_path
            cause_metadata = metadata.get("Cause", [])
            input_index = list(metadata["csv"]).index(relative_path)
            causes = [value.lower() for value in (cause_metadata[input_index] or [])]
            required_columns = set(causes)
            for query in parsed_queries:
                required_columns.add(query["attribute"])
                required_columns.update(query["group"])
                required_columns.update(column for column, _, _ in conditions(query["where"]))
            frame = pd.read_csv(input_path, usecols=sorted(required_columns), low_memory=False)
            frame.columns = frame.columns.str.lower()
            for query_index, (query_text, query) in enumerate(zip(metadata["queries"], parsed_queries), start=1):
                filtered = filter_frame(frame, conditions(query["where"]))
                if mechanism == "MCAR":
                    width, output_groups = mcar_result(filtered, query["attribute"], query["group"])
                else:
                    width, output_groups = mar_result(filtered, query["attribute"], causes, query["group"])
                rows.append(
                    {
                        "dataset": dataset,
                        "mechanism": mechanism,
                        "rate": rate_from_name(block_name),
                        "query_index": query_index,
                        "query": query_text,
                        "delta_w": width,
                        "status": "ok" if np.isfinite(width) else "undefined",
                        "output_groups": output_groups,
                        "input_rows": len(frame),
                        "selected_rows": len(filtered),
                    }
                )
            del frame
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    summary = (
        result[result["status"].eq("ok")]
        .groupby(["dataset", "rate"], as_index=False)
        .agg(delta_w=("delta_w", "mean"), completed_queries=("query_index", "size"))
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.data_root, arguments.output)


if __name__ == "__main__":
    main()
