"""Compare CAEX, CAMC, and CADE on the common MNAR workloads.

Each configured query is executed once. PostgreSQL enforces an independent
statement timeout for every measured query. CAMC uses H=783 repairs and the
factor sampler; factor sampling is setup and is not included in query time.
MCDB relation encoding is amortized across the ten queries in a block.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import itertools
import json
import math
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psycopg2

from CadeDiscretization import (
    discretized_orderings,
    materialized_cade_source,
    observed_grouping_patterns,
    prepare_cade_bins,
    source_cte,
)
from MCDBPostgresNative import NativeMCDB, literal, parse_query, qident, relation_mapping
from RunnerSetQueriy import evaluate_query_with_groups
from SetQueryRewriterExecuter import QueryExecutor as CAEXSetExecutor
from nonAgg_direct import _load_table, run_direct_per_tuple


CONN = dict(host=os.environ.get("PGHOST", "localhost"), port=int(os.environ.get("PGPORT", "5433")), dbname=os.environ.get("PGDATABASE", "mydb"), user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD", ""))
GROUPS = {
    "bank": ("bank_manr1_set", "bank_manr1_agg"),
    "nyc": ("nyc_manr1_set", "nyc_manr1_agg"),
    "bitcoin": ("bit_manr1_set", "bit_manr1_agg"),
}
SEPARATOR_FILES = {
    "bank": "data/MNAR1Data/bank/bank_agg_mnar1_minimal_separators_Xi.csv",
    "nyc": "data/MNAR1Data/nyc/nyc_agg_mnar1_minimal_separators_Xi.csv",
    "bitcoin": "data/MNAR1Data/BitcoinHeistData/BitcoinHeistData_agg_mnar1_minimal_separators_Xi.csv",
}
METHODS = ("CAEX", "CAMC", "CADE")


def clean_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def normalize_key(values: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(normalize_value(value) for value in values)


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def set_timeout(conn, seconds: int) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT set_config('statement_timeout', %s, false)", (str(int(seconds) * 1000),))
    conn.commit()


def ensure_loaded(
    conn,
    csv_path: str,
    table: str,
    force: bool = False,
) -> None:
    if force:
        _load_table(conn, csv_path, table, force=True)
        return
    cursor = conn.cursor()
    cursor.execute("SELECT to_regclass(%s)", (table,))
    exists = cursor.fetchone()[0] is not None
    count = 0
    if exists:
        cursor.execute("SELECT count(*) FROM %s" % qident(table))
        count = int(cursor.fetchone()[0])
    conn.commit()
    if count == 0:
        _load_table(conn, csv_path, table, force=True)


def raw_columns(conn, table: str) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT a.attname FROM pg_catalog.pg_attribute a "
        "JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
        "WHERE c.relname=%s AND pg_catalog.pg_table_is_visible(c.oid) "
        "AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum",
        (table,),
    )
    return [row[0] for row in cursor.fetchall()]


def columns(conn, table: str) -> List[str]:
    return [value.lower() for value in raw_columns(conn, table)]


def source_columns(conn, table: str) -> List[Tuple[str, str]]:
    return [(value, value.lower()) for value in raw_columns(conn, table)]


def resolve_column(conn, table: str, requested: str) -> str:
    matches = [value for value in raw_columns(conn, table) if value.lower() == requested.lower()]
    if not matches:
        raise ValueError("Column %s is absent from %s" % (requested, table))
    return matches[0]


def create_aligned_single_subsets(conn, source: str, truth: str, prefix: str,
                                  row_limit: int, seed: int) -> Tuple[str, str]:
    cursor = conn.cursor()
    key_table = clean_name(prefix + "_row_keys")[:55]
    pred_table = clean_name(prefix + "_pred0")[:55]
    truth_table = clean_name(prefix + "_truth0")[:55]
    source_cols = [pair for pair in source_columns(conn, source) if pair[1] != "_rid"]
    truth_cols = [pair for pair in source_columns(conn, truth) if pair[1] != "_rid"]
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(key_table))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
        "WITH numbered AS (SELECT row_number() OVER (ORDER BY ctid)::bigint AS rn FROM %s) "
        "SELECT rn FROM numbered ORDER BY hashtextextended(rn::text, %d) LIMIT %d" % (
            qident(key_table), qident(source), int(seed), int(row_limit)
        )
    )
    cursor.execute("CREATE UNIQUE INDEX ON %s (rn)" % qident(key_table))

    def make(source_table: str, target_table: str, selected: Sequence[Tuple[str, str]]) -> None:
        projection = ", ".join(
            "n.%s AS %s" % (qident(actual), qident(normalized)) for actual, normalized in selected
        )
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(target_table))
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "WITH numbered AS (SELECT row_number() OVER (ORDER BY ctid)::bigint AS rn, s.* FROM %s s) "
            "SELECT n.rn AS %s, %s FROM numbered n JOIN %s k USING (rn)" % (
                qident(target_table), qident(source_table), qident("_rid"), projection,
                qident(key_table),
            )
        )
        cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (qident(target_table), qident("_rid")))
        cursor.execute("ANALYZE %s" % qident(target_table))

    make(source, pred_table, source_cols)
    make(truth, truth_table, truth_cols)
    conn.commit()
    return pred_table, truth_table


def create_aligned_join_subsets(conn, pred_left: str, pred_right: str,
                                truth_left: str, truth_right: str, key: str,
                                prefix: str, key_limit: int, seed: int) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    cursor = conn.cursor()
    key_table = clean_name(prefix + "_join_keys")[:55]
    pred_left_key = resolve_column(conn, pred_left, key)
    pred_right_key = resolve_column(conn, pred_right, key)
    truth_left_key = resolve_column(conn, truth_left, key)
    truth_right_key = resolve_column(conn, truth_right, key)
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(key_table))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
        "SELECT l.%s AS join_key FROM %s l JOIN %s r ON l.%s=r.%s "
        "JOIN %s fl ON fl.%s=l.%s JOIN %s fr ON fr.%s=l.%s "
        "WHERE l.%s IS NOT NULL GROUP BY l.%s "
        "ORDER BY hashtextextended(l.%s::text, %d) LIMIT %d" % (
            qident(key_table), qident(pred_left_key), qident(pred_left), qident(pred_right),
            qident(pred_left_key), qident(pred_right_key), qident(truth_left), qident(truth_left_key),
            qident(pred_left_key), qident(truth_right), qident(truth_right_key), qident(pred_left_key),
            qident(pred_left_key), qident(pred_left_key), qident(pred_left_key), int(seed), int(key_limit),
        )
    )
    cursor.execute("CREATE UNIQUE INDEX ON %s (join_key)" % qident(key_table))

    def make(source_table: str, target_table: str) -> None:
        selected = [pair for pair in source_columns(conn, source_table) if pair[1] != "_rid"]
        projection = ", ".join(
            "s.%s AS %s" % (qident(actual), qident(normalized)) for actual, normalized in selected
        )
        source_key = resolve_column(conn, source_table, key)
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(target_table))
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT row_number() OVER (ORDER BY s.ctid)::bigint AS %s, %s "
            "FROM %s s JOIN %s k ON s.%s=k.join_key" % (
                qident(target_table), qident("_rid"), projection, qident(source_table),
                qident(key_table), qident(source_key),
            )
        )
        cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (qident(target_table), qident("_rid")))
        cursor.execute("CREATE INDEX ON %s (%s)" % (qident(target_table), qident(key)))
        cursor.execute("ANALYZE %s" % qident(target_table))

    pl = clean_name(prefix + "_pred1")[:55]
    pr = clean_name(prefix + "_pred2")[:55]
    tl = clean_name(prefix + "_truth1")[:55]
    tr = clean_name(prefix + "_truth2")[:55]
    make(pred_left, pl)
    make(pred_right, pr)
    make(truth_left, tl)
    make(truth_right, tr)
    conn.commit()
    return (pl, pr), (tl, tr)


def token_map(csvs: Sequence[str], configured_tables: Sequence[str], actual_tables: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for csv_path, configured, actual in zip(csvs, configured_tables, actual_tables):
        base = os.path.basename(csv_path)
        stem = os.path.splitext(base)[0]
        directory = os.path.basename(os.path.dirname(csv_path))
        normalized = csv_path.replace("\\", "/")
        parts = [value for value in normalized.split("/") if value]
        suffixes = ["/".join(parts[position:]) for position in range(len(parts))]
        tokens = [csv_path, base, stem, configured, "%s/%s" % (directory, base),
                  "%s/%s" % (directory, stem)]
        tokens.extend(suffixes)
        for token in tokens:
            result[token] = actual
    return result


def replace_tokens(sql: str, replacements: Mapping[str, str]) -> str:
    result = " ".join(sql.strip().rstrip(";").split())
    for token in sorted(replacements, key=len, reverse=True):
        result = re.sub(re.escape(token), replacements[token], result, flags=re.IGNORECASE)
    return result


def relation_settings(meta: Mapping[str, Any], position: int, workload: str,
                      factor_map: Mapping[str, Sequence[str]]) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    if workload == "set":
        if position == 0:
            missing = meta.get("missing_attrs_single", [])
            ordering = meta.get("ordering_single", {})
        elif position == 1:
            missing = meta.get("missing_attrs_T", [])
            ordering = meta.get("ordering_T", {})
        else:
            missing = meta.get("missing_attrs_S", [])
            ordering = meta.get("ordering_S", {})
        return tuple(str(value).lower() for value in (missing or [])), {
            str(key).lower(): tuple(str(value).lower() for value in values)
            for key, values in (ordering or {}).items()
        }
    return tuple(), {str(key).lower(): tuple(str(value).lower() for value in values)
                     for key, values in factor_map.items()}


def load_factor_map(path: str) -> Dict[str, Tuple[str, ...]]:
    result: Dict[str, Tuple[str, ...]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            attr = str(row.get("Y", "")).strip().lower()
            raw = row.get("Xi_minimal") or "[]"
            try:
                values = ast.literal_eval(raw)
            except Exception:
                values = []
            result[attr] = tuple(str(value).lower() for value in values)
    return result


def create_join_view(conn, left: str, right: str, key: str, name: str) -> str:
    cursor = conn.cursor()
    left_cols = columns(conn, left)
    right_cols = columns(conn, right)
    selected = ["l.%s AS %s" % (qident(column), qident(column)) for column in left_cols if column != "_rid"]
    used = set(column for column in left_cols if column != "_rid")
    selected.extend(
        "r.%s AS %s" % (qident(column), qident(column))
        for column in right_cols if column not in used and column != "_rid"
    )
    cursor.execute("DROP VIEW IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP VIEW %s AS SELECT %s FROM %s l JOIN %s r ON l.%s=r.%s" % (
            qident(name), ", ".join(selected), qident(left), qident(right), qident(key), qident(key)
        )
    )
    conn.commit()
    return name


def create_deduplicated_right(conn, left: str, right: str, key: str, name: str) -> str:
    cursor = conn.cursor()
    left_names = set(columns(conn, left))
    selected = []
    for actual, normalized in source_columns(conn, right):
        if normalized in ("_rid", key.lower()) or normalized not in left_names:
            selected.append("r.%s AS %s" % (qident(actual), qident(normalized)))
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS SELECT %s FROM %s r" % (
            qident(name), ", ".join(selected), qident(right)
        )
    )
    cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (qident(name), qident("_rid")))
    cursor.execute("CREATE INDEX ON %s (%s)" % (qident(name), qident(key.lower())))
    cursor.execute("ANALYZE %s" % qident(name))
    conn.commit()
    return name


def project_joint_bundle(conn, bundle, source_table: str, join_key: str,
                         name: str):
    """Project a jointly sampled bundle onto one physical relation."""
    join_key = join_key.lower()
    relation_columns = [
        column for column in columns(conn, source_table)
        if column not in ("_rid", join_key)
    ]
    input_columns = set(bundle.columns)
    selected = ["_rid", join_key] + relation_columns
    for attribute in relation_columns:
        symbol_column = "%s_nullsym" % attribute
        if symbol_column in input_columns:
            selected.append(symbol_column)
        if attribute in bundle.missing_attributes:
            selected.append(bundle.sample_column(attribute))
    selected.append("__present")
    selected = list(dict.fromkeys(selected))
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(name))
    cursor.execute(
        "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS SELECT %s FROM %s" % (
            qident(name), ", ".join(qident(column) for column in selected),
            qident(bundle.bundle_table),
        )
    )
    cursor.execute(
        "CREATE UNIQUE INDEX ON %s (%s)" %
        (qident(name), qident("_rid"))
    )
    cursor.execute(
        "CREATE UNIQUE INDEX ON %s (%s)" %
        (qident(name), qident(join_key))
    )
    cursor.execute("ANALYZE %s" % qident(name))
    conn.commit()
    selected_set = set(selected)
    kept_columns = tuple(
        column for column in bundle.columns if column in selected_set
    )
    kept_types = {
        column: value for column, value in bundle.column_types.items()
        if column in selected_set
    }
    kept_missing = tuple(
        attribute for attribute in bundle.missing_attributes
        if attribute in relation_columns
    )
    return dataclasses.replace(
        bundle,
        base_table=name,
        bundle_table=name,
        columns=kept_columns,
        column_types=kept_types,
        missing_attributes=kept_missing,
        sampling_s=0.0,
        encoding_s=0.0,
    )


def flatten_join_query(query: str, view: str) -> str:
    return re.sub(
        r"\bFROM\s+\S+\s+JOIN\s+\S+\s+USING\s*\(\s*[A-Za-z_]\w*\s*\)",
        "FROM %s" % view,
        " ".join(query.strip().rstrip(";").split()), count=1, flags=re.IGNORECASE,
    )


def expose_group_columns(query: str) -> str:
    spec = parse_query(query)
    if not spec.group_by:
        return query
    match = re.search(r"\bSELECT\s+(?P<select>.*?)\s+FROM\s+", query, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Unable to parse the aggregation SELECT list")
    select_list = match.group("select")
    selected = [value.strip().strip('"').lower() for value in select_list.split(",")]
    missing_groups = [value for value in spec.group_by if value.lower() not in selected]
    if not missing_groups:
        return query
    replacement = ", ".join(missing_groups + [select_list])
    return query[:match.start("select")] + replacement + query[match.end("select"):]


def remove_redundant_join_grouping(query: str) -> str:
    if " JOIN " not in query.upper():
        return query
    return re.sub(r"\s+GROUP\s+BY\s+[^;]+\s*;?\s*$", "", query, flags=re.IGNORECASE)


def is_timeout(error: Exception) -> bool:
    return "statement timeout" in str(error).lower() or "canceling statement" in str(error).lower()


def timeout_row(base: Mapping[str, Any], method: str, timeout_s: int, error: Exception) -> Dict[str, Any]:
    return {**base, "method": method, "time_s": float(timeout_s), "metric": None,
            "coverage": None, "delta_w": None,
            "status": "timeout" if is_timeout(error) else "error", "error": str(error)}


def direct_coverage(groups: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not groups:
        return None
    covered = 0
    for row in groups:
        lo = max(0.0, float(row["p_hat"]) - float(row["ci_half"]))
        hi = min(1.0, float(row["p_hat"]) + float(row["ci_half"]))
        covered += int(lo <= float(row["p_oracle"]) <= hi)
    return covered / len(groups)


def wilson_interval(successes: float, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = max(0.0, min(1.0, float(successes) / trials))
    d = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / d
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / d
    return max(0.0, center - half), min(1.0, center + half)


def mean_normalized_width(estimates: Iterable[Tuple[float, float, float]]) -> Optional[float]:
    """Average ``(upper-lower)/abs(estimate)`` over nonzero point estimates."""
    widths = [
        max(0.0, upper - lower) / abs(estimate)
        for estimate, lower, upper in estimates
        if abs(estimate) > 1e-12
    ]
    return sum(widths) / len(widths) if widths else None


def ratio_of_mean_interval_width(estimates: Iterable[Tuple[float, float, float]]) -> Optional[float]:
    """Mean interval width divided by the absolute mean point estimate."""
    rows = list(estimates)
    if not rows:
        return None
    mean_estimate = sum(abs(estimate) for estimate, _lower, _upper in rows) / len(rows)
    mean_width = sum(max(0.0, upper - lower) for _estimate, lower, upper in rows) / len(rows)
    return mean_width / mean_estimate if mean_estimate > 1e-12 else None


def output_tvd(predicted: Mapping[Tuple[Any, ...], float], truth: Iterable[Tuple[Any, ...]]) -> float:
    truth_set = set(truth)
    keys = set(predicted) | truth_set
    zp = sum(max(0.0, value) for value in predicted.values())
    zt = float(len(truth_set))
    return 0.5 * sum(
        abs((max(0.0, predicted.get(key, 0.0)) / zp if zp else 0.0) -
            (1.0 / zt if key in truth_set and zt else 0.0))
        for key in keys
    )


def execute_truth_set(conn, query: str) -> List[Tuple[Any, ...]]:
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT * FROM (%s) truth_query" % query)
    return [normalize_key(row) for row in cursor.fetchall()]


def run_camc_set(engine: NativeMCDB, query: str, mappings: Mapping[str, Any],
                 truth: Sequence[Tuple[Any, ...]], h: int) -> Tuple[float, float, float, int, Optional[float]]:
    result = engine.evaluate_summary(query, mappings)
    predicted = {normalize_key(row[:-1]): float(row[-1]) for row in result.summary_rows}
    tvd = output_tvd(predicted, truth)
    truth_set = set(truth)
    keys = set(predicted) | truth_set
    covered = 0
    intervals = []
    for key in keys:
        p = predicted.get(key, 0.0)
        lo, hi = wilson_interval(round(p * h), h)
        covered += int(lo <= (1.0 if key in truth_set else 0.0) <= hi)
        if key in predicted:
            intervals.append((p, lo, hi))
    coverage = covered / len(keys) if keys else 1.0
    delta_w = ratio_of_mean_interval_width(intervals)
    return result.elapsed_s, tvd, coverage, len(predicted), delta_w


def predicate_sql(alias: str, predicate) -> str:
    operator = "<>" if predicate.operator == "!=" else predicate.operator
    return "%s.%s %s %s" % (alias, qident(predicate.column), operator, literal(predicate.value))


def matching_condition(base_alias: str, stats_alias: str, separators: Sequence[str]) -> str:
    if not separators:
        return "TRUE"
    return " AND ".join(
        "(%s.%s IS NULL OR %s.%s=%s.%s)" % (
            base_alias, qident(column), stats_alias, qident(column), base_alias, qident(column)
        ) for column in separators
    )


def grouping_sets(
    separators: Sequence[str],
    value: Optional[str] = None,
    masks: Optional[Sequence[int]] = None,
) -> str:
    groups = []
    if masks is None:
        subsets = (
            subset
            for size in range(len(separators) + 1)
            for subset in itertools.combinations(separators, size)
        )
    else:
        subsets = (
            tuple(
                column
                for index, column in enumerate(separators)
                if not (
                    int(mask)
                    & (1 << (len(separators) - index - 1))
                )
            )
            for mask in masks
        )
    for subset in subsets:
        columns = list(subset)
        if value is not None:
            columns.append(value)
        groups.append("(" + ", ".join(qident(column) for column in columns) + ")")
    return ", ".join(groups)


def grouping_mask(alias: str, separators: Sequence[str]) -> str:
    if not separators:
        return "0"
    terms = [
        "CASE WHEN %s.%s IS NULL THEN %d ELSE 0 END" % (
            alias, qident(column), 1 << (len(separators) - index - 1)
        )
        for index, column in enumerate(separators)
    ]
    return " + ".join(terms)


def grouped_match(base_alias: str, stats_alias: str, separators: Sequence[str]) -> str:
    terms = ["%s.gmask=(%s)" % (stats_alias, grouping_mask(base_alias, separators))]
    terms.extend(
        "COALESCE(%s.%s::text,'__CADE_NULL__')=COALESCE(%s.%s::text,'__CADE_NULL__')" % (
            stats_alias, qident(column), base_alias, qident(column)
        )
        for column in separators
    )
    return " AND ".join(terms)


def aggregate_sql(query: str, relation: str, orderings: Mapping[str, Sequence[str]],
                  active_missing: Sequence[str], direct: bool,
                  bin_boundaries: Optional[Mapping[str, Sequence[float]]] = None,
                  grouping_patterns: Optional[Mapping[Tuple[str, ...], Sequence[int]]] = None,
                  summarize_direct: bool = True) -> Tuple[str, int]:
    spec = parse_query(query)
    aggregate_items = [item for item in spec.select_items if item.aggregate]
    if len(aggregate_items) != 1 or aggregate_items[0].aggregate != "avg" or not aggregate_items[0].column:
        raise ValueError("The aggregation comparison currently supports one AVG expression")
    y = aggregate_items[0].column
    group_cols = list(spec.group_by)
    missing = set(value.lower() for value in active_missing)
    ctes: List[str] = []
    working_relation = relation
    effective_orderings = {
        str(attribute).lower(): tuple(str(value).lower() for value in separators)
        for attribute, separators in orderings.items()
    }
    if direct and bin_boundaries:
        ctes.append(source_cte(relation, bin_boundaries))
        working_relation = "cade_source"
        effective_orderings = discretized_orderings(orderings, bin_boundaries)
    joins: List[str] = []
    weight_terms: List[str] = []

    by_attr: Dict[str, List[Any]] = {}
    for predicate in spec.predicates:
        by_attr.setdefault(predicate.column, []).append(predicate)
    for index, (attr, predicates) in enumerate(by_attr.items()):
        observed_expr = " AND ".join(predicate_sql("b", predicate) for predicate in predicates)
        if attr not in missing:
            weight_terms.append("CASE WHEN %s THEN 1.0 ELSE 0.0 END" % observed_expr)
            continue
        separators = [value for value in effective_orderings.get(attr, ()) if value != attr]
        alias = "qp%d" % index
        cols = ", ".join(qident(value) for value in separators)
        select_prefix = cols + ", " if cols else ""
        mask = "GROUPING(%s)" % cols if cols else "0"
        group_suffix = " GROUP BY GROUPING SETS (%s)" % grouping_sets(
            separators,
            masks=(grouping_patterns or {}).get(tuple(separators)),
        ) if cols else ""
        stat_predicate = " AND ".join(predicate_sql("s", predicate) for predicate in predicates)
        sep_not_null = " AND ".join("s.%s IS NOT NULL" % qident(value) for value in separators)
        where = "s.%s IS NOT NULL" % qident(attr)
        if sep_not_null:
            where += " AND " + sep_not_null
        ctes.append(
            "%s AS MATERIALIZED (SELECT %s%s AS gmask, "
            "SUM(CASE WHEN %s THEN 1.0 ELSE 0.0 END)/NULLIF(COUNT(*)::double precision,0) AS p "
            "FROM %s s WHERE %s%s)" % (
                alias, select_prefix, mask, stat_predicate, working_relation, where, group_suffix
            )
        )
        lateral = "lp%d" % index
        joins.append(
            "LEFT JOIN %s %s ON %s" % (alias, lateral, grouped_match("b", lateral, separators))
        )
        weight_terms.append(
            "CASE WHEN b.%s IS NULL THEN COALESCE(%s.p,0.0) ELSE CASE WHEN %s THEN 1.0 ELSE 0.0 END END" % (
                qident(attr), lateral, observed_expr
            )
        )

    group_values: List[str] = []
    for index, attr in enumerate(group_cols):
        if attr not in missing:
            group_values.append("b.%s" % qident(attr))
            continue
        separators = [value for value in effective_orderings.get(attr, ()) if value != attr]
        alias = "qg%d" % index
        cols = ", ".join(qident(value) for value in separators)
        select_prefix = cols + ", " if cols else ""
        mask = "GROUPING(%s)" % cols if cols else "0"
        group_suffix = " GROUP BY GROUPING SETS (%s)" % grouping_sets(
            separators,
            attr,
            (grouping_patterns or {}).get(tuple(separators)),
        )
        sep_not_null = " AND ".join("s.%s IS NOT NULL" % qident(value) for value in separators)
        where = "s.%s IS NOT NULL" % qident(attr)
        if sep_not_null:
            where += " AND " + sep_not_null
        ctes.append(
            "%s_raw AS MATERIALIZED (SELECT %s%s AS value, %s AS gmask, COUNT(*)::double precision AS n "
            "FROM %s s WHERE %s%s), "
            "%s AS MATERIALIZED (SELECT *, SUM(n) OVER (PARTITION BY gmask%s) AS total FROM %s_raw)" % (
                alias, select_prefix, qident(attr), mask, working_relation, where, group_suffix,
                alias, (", " + cols) if cols else "", alias
            )
        )
        lateral = "lg%d" % index
        joins.append(
            "LEFT JOIN %s %s ON b.%s IS NULL AND %s" % (
                alias, lateral, qident(attr), grouped_match("b", lateral, separators)
            )
        )
        weight_terms.append(
            "CASE WHEN b.%s IS NULL THEN COALESCE(%s.n/NULLIF(%s.total,0),0.0) ELSE 1.0 END" % (
                qident(attr), lateral, lateral
            )
        )
        group_values.append("CASE WHEN b.%s IS NULL THEN %s.value ELSE b.%s END" % (
            qident(attr), lateral, qident(attr)
        ))

    y_separators = [value for value in effective_orderings.get(y, ()) if value != y]
    y_cols = ", ".join(qident(value) for value in y_separators)
    y_select_prefix = y_cols + ", " if y_cols else ""
    y_sep_not_null = " AND ".join("s.%s IS NOT NULL" % qident(value) for value in y_separators)
    y_where = "s.%s IS NOT NULL" % qident(y)
    if y_sep_not_null:
        y_where += " AND " + y_sep_not_null

    if direct and y not in missing:
        y_value = "b.%s::double precision" % qident(y)
    elif direct:
        y_mask = "GROUPING(%s)" % y_cols if y_cols else "0"
        y_group_suffix = " GROUP BY GROUPING SETS (%s)" % grouping_sets(
            y_separators,
            masks=(grouping_patterns or {}).get(tuple(y_separators)),
        ) if y_cols else ""
        ctes.append(
            "qy AS MATERIALIZED (SELECT %s%s AS gmask, SUM(s.%s::double precision) AS sy, "
            "COUNT(*)::double precision AS n FROM %s s WHERE %s%s)" % (
                y_select_prefix, y_mask, qident(y), working_relation, y_where, y_group_suffix
            )
        )
        joins.append(
            "LEFT JOIN qy ly ON %s" % grouped_match("b", "ly", y_separators)
        )
        y_value = "CASE WHEN b.%s IS NULL THEN ly.sy/NULLIF(ly.n,0) ELSE b.%s::double precision END" % (qident(y), qident(y))
    else:
        group_for_y = ", ".join(([qident(value) for value in y_separators] + [qident(y)]))
        ctes.append(
            "qy AS MATERIALIZED (SELECT %s%s::double precision AS value, COUNT(*)::double precision AS n "
            "FROM %s s WHERE %s GROUP BY %s)" % (
                y_select_prefix, qident(y), working_relation, y_where, group_for_y
            )
        )
        joins.append(
            "JOIN LATERAL (SELECT b.%s::double precision AS value, 1.0 AS p WHERE b.%s IS NOT NULL "
            "UNION ALL SELECT d.value, SUM(d.n)/SUM(SUM(d.n)) OVER () AS p FROM qy d "
            "WHERE b.%s IS NULL AND %s GROUP BY d.value) ly ON TRUE" % (
                qident(y), qident(y), qident(y), matching_condition("b", "d", y_separators)
            )
        )
        weight_terms.append("ly.p")
        y_value = "ly.value"

    weight = " * ".join("(%s)" % value for value in weight_terms) if weight_terms else "1.0"
    group_select = ", ".join("%s AS %s" % (value, qident(attr)) for value, attr in zip(group_values, group_cols))
    prefix = group_select + ", " if group_select else ""
    group_names = ", ".join(qident(value) for value in group_cols)
    group_prefix = group_names + ", " if group_names else ""
    group_clause = " GROUP BY " + group_names if group_names else ""
    summarize_base = summarize_direct and direct and y not in by_attr and y not in group_cols
    if summarize_base:
        base_columns: List[str] = []
        for column in group_cols + list(by_attr):
            if column != y and column not in base_columns:
                base_columns.append(column)
        relevant_factor_attributes = list(by_attr) + group_cols + [y]
        for attribute in relevant_factor_attributes:
            separators = effective_orderings.get(attribute, ())
            for column in separators:
                if column not in base_columns:
                    base_columns.append(column)
        base_select = ", ".join("s.%s" % qident(column) for column in base_columns)
        base_prefix = base_select + ", " if base_select else ""
        base_group = " GROUP BY " + base_select if base_select else ""
        ctes.append(
            "base_stats AS MATERIALIZED (SELECT %sCOUNT(*)::double precision AS n_rows, "
            "COUNT(s.%s)::double precision AS n_y_obs, "
            "SUM(s.%s::double precision) AS sum_y_obs, "
            "SUM(POWER(s.%s::double precision,2)) AS sum_y2_obs "
            "FROM %s s%s)" % (
                base_prefix,
                qident(y),
                qident(y),
                qident(y),
                working_relation,
                base_group,
            )
        )
        if y in missing:
            imputed_y = "ly.sy/NULLIF(ly.n,0)"
            valid_n = (
                "b.n_y_obs + CASE WHEN %s IS NOT NULL "
                "THEN b.n_rows-b.n_y_obs ELSE 0 END" % imputed_y
            )
            valid_sum = (
                "COALESCE(b.sum_y_obs,0) + CASE WHEN %s IS NOT NULL "
                "THEN (b.n_rows-b.n_y_obs)*(%s) ELSE 0 END"
                % (imputed_y, imputed_y)
            )
            valid_sum2 = (
                "COALESCE(b.sum_y2_obs,0) + CASE WHEN %s IS NOT NULL "
                "THEN (b.n_rows-b.n_y_obs)*POWER(%s,2) ELSE 0 END"
                % (imputed_y, imputed_y)
            )
        else:
            valid_n = "b.n_y_obs"
            valid_sum = "COALESCE(b.sum_y_obs,0)"
            valid_sum2 = "COALESCE(b.sum_y2_obs,0)"
        ctes.append(
            "annotated AS (SELECT %s(%s)*(%s) AS sw, "
            "(%s)*POWER((%s),2) AS sw2, "
            "(%s)*(%s) AS swy, "
            "(%s)*(%s) AS swy2 "
            "FROM base_stats b %s)" % (
                prefix,
                valid_n,
                weight,
                valid_n,
                weight,
                valid_sum,
                weight,
                valid_sum2,
                weight,
                " ".join(joins),
            )
        )
        ctes.append(
            "moments AS (SELECT %sSUM(sw) AS sw, SUM(sw2) AS sw2, "
            "SUM(swy) AS swy, SUM(swy2) AS swy2 "
            "FROM annotated WHERE sw>0%s)" % (
                group_prefix,
                group_clause,
            )
        )
    else:
        ctes.append(
            "annotated AS (SELECT %s%s AS y_value, (%s)::double precision AS w "
            "FROM %s b %s)" % (
                prefix,
                y_value,
                weight,
                working_relation,
                " ".join(joins),
            )
        )
        ctes.append(
            "moments AS (SELECT %sSUM(w) AS sw, SUM(w*w) AS sw2, "
            "SUM(w*y_value) AS swy, SUM(w*y_value*y_value) AS swy2 "
            "FROM annotated WHERE w>0 AND y_value IS NOT NULL%s)" % (
                group_prefix,
                group_clause,
            )
        )
    final_prefix = group_names + ", " if group_names else ""
    sql = (
        "WITH " + ",\n".join(ctes) + "\nSELECT " + final_prefix +
        "swy/NULLIF(sw,0) AS estimate, "
        "SQRT(GREATEST(0.0, swy2/NULLIF(sw,0)-POWER(swy/NULLIF(sw,0),2)) / "
        "NULLIF(POWER(sw,2)/NULLIF(sw2,0),0)) AS stderr FROM moments ORDER BY " +
        (group_names if group_names else "1")
    )
    return sql, len(group_cols)


def run_analytic_aggregate(conn, query: str, relation: str, truth_query: str,
                           orderings: Mapping[str, Sequence[str]], active_missing: Sequence[str],
                           direct: bool) -> Tuple[float, float, float, int, Optional[float]]:
    working_relation = relation
    effective_orderings = orderings
    grouping_patterns = {}
    if direct:
        working_relation, effective_orderings = materialized_cade_source(
            conn, relation, orderings
        )
        grouping_patterns = observed_grouping_patterns(
            conn, working_relation, effective_orderings
        )
    sql, n_group = aggregate_sql(
        query,
        working_relation,
        effective_orderings,
        active_missing,
        direct,
        {},
        grouping_patterns,
    )
    cursor = conn.cursor()
    started = time.perf_counter()
    cursor.execute(sql)
    predicted_rows = cursor.fetchall()
    elapsed = time.perf_counter() - started
    cursor.execute(truth_query)
    truth_rows = cursor.fetchall()
    predicted = {normalize_key(row[:n_group]): (float(row[n_group]), float(row[n_group + 1] or 0.0))
                 for row in predicted_rows if row[n_group] is not None}
    truth = {normalize_key(row[:n_group]): float(row[n_group])
             for row in truth_rows if row[n_group] is not None}
    errors: List[float] = []
    covered = 0
    for key, value in truth.items():
        if key not in predicted:
            errors.append(1.0)
            continue
        estimate, stderr = predicted[key]
        errors.append(abs(estimate - value) / max(abs(value), 1e-12))
        covered += int(estimate - 1.96 * stderr <= value <= estimate + 1.96 * stderr)
    metric = sum(errors) / len(errors) if errors else 0.0
    coverage = covered / len(truth) if truth else 1.0
    delta_w = mean_normalized_width(
        (estimate, estimate - 1.96 * stderr, estimate + 1.96 * stderr)
        for estimate, stderr in predicted.values()
    )
    return elapsed, metric, coverage, len(predicted), delta_w


def run_camc_aggregate(engine: NativeMCDB, query: str, mappings: Mapping[str, Any],
                       conn, truth_query: str, h: int) -> Tuple[float, float, float, int, Optional[float]]:
    result = engine.evaluate_summary(query, mappings)
    spec = result.query
    n_group = len(spec.group_by)
    cursor = conn.cursor()
    cursor.execute(truth_query)
    truth_rows = cursor.fetchall()
    truth = {normalize_key(row[:n_group]): float(row[n_group])
             for row in truth_rows if row[n_group] is not None}
    predicted: Dict[Tuple[Any, ...], Tuple[float, float]] = {}
    for row in result.summary_rows:
        key = normalize_key(row[:n_group])
        expected = float(row[n_group]) if row[n_group] is not None else 0.0
        sample_sd = float(row[n_group + 1]) if row[n_group + 1] is not None else 0.0
        predicted[key] = (expected, sample_sd / math.sqrt(h))
    errors: List[float] = []
    covered = 0
    for key, value in truth.items():
        if key not in predicted:
            errors.append(1.0)
            continue
        estimate, stderr = predicted[key]
        errors.append(abs(estimate - value) / max(abs(value), 1e-12))
        covered += int(estimate - 1.96 * stderr <= value <= estimate + 1.96 * stderr)
    metric = sum(errors) / len(errors) if errors else 0.0
    coverage = covered / len(truth) if truth else 1.0
    delta_w = mean_normalized_width(
        (estimate, estimate - 1.96 * stderr, estimate + 1.96 * stderr)
        for estimate, stderr in predicted.values()
    )
    return result.elapsed_s, metric, coverage, len(predicted), delta_w


def append_extra_aggregate_queries(config: Dict[str, Any], templates_path: str) -> None:
    with open(templates_path) as handle:
        templates = json.load(handle)
    for dataset, (_set_group, agg_group) in GROUPS.items():
        for block_name, meta in config[agg_group].items():
            rate_match = re.search(r"(?:mnar_?|_)(5|10|20)$", block_name, re.IGNORECASE)
            if not rate_match:
                raise ValueError("Cannot infer missingness rate from %s" % block_name)
            rate = int(rate_match.group(1))
            extra = [query.format(rate=rate) for query in templates[dataset]]
            queries = list(meta.get("queries", []))
            if len(queries) == 5:
                queries.extend(extra)
            if len(queries) != 10:
                raise ValueError("%s must contain exactly ten aggregation queries" % block_name)
            meta["queries"] = queries


def prepare_block(conn, meta: Mapping[str, Any], dataset: str, block: str,
                  workload: str, row_limit: int, seed: int,
                  force_reload: bool = False) -> Dict[str, Any]:
    csvs = list(meta["csv"])
    tables = list(meta["table"])
    truth_csvs = list(meta["complete_csv"])
    truth_tables = list(meta["complete_table"])
    for path, table in zip(csvs, tables):
        ensure_loaded(conn, path, table, force=force_reload)
    for path, table in zip(truth_csvs, truth_tables):
        ensure_loaded(conn, path, table)
    prefix = "sec_cmp_%s_%s_%s" % (workload, dataset, clean_name(block))
    pred0, truth0 = create_aligned_single_subsets(
        conn, tables[0], truth_tables[0], prefix, row_limit, seed
    )
    join_spec = next((parse_query(query) for query in meta["queries"] if " JOIN " in query.upper()), None)
    if join_spec is None or not join_spec.join_column:
        raise ValueError("%s has no join query" % block)
    (pred1, pred2), (truth1, truth2) = create_aligned_join_subsets(
        conn, tables[1], tables[2], truth_tables[1], truth_tables[2], join_spec.join_column,
        prefix, row_limit, seed,
    )
    pred2_clean = create_deduplicated_right(
        conn, pred1, pred2, join_spec.join_column, clean_name(prefix + "_pred2_unique")[:55]
    )
    truth2_clean = create_deduplicated_right(
        conn, truth1, truth2, join_spec.join_column, clean_name(prefix + "_truth2_unique")[:55]
    )
    pred_tables = [pred0, pred1, pred2_clean]
    full_tables = [truth0, truth1, truth2_clean]
    pred_map = token_map(csvs, tables, pred_tables)
    truth_map = token_map(csvs, tables, full_tables)
    return {"csvs": csvs, "tables": tables, "truth_tables": truth_tables,
            "pred_tables": pred_tables, "full_tables": full_tables,
            "pred_map": pred_map, "truth_map": truth_map,
            "join_key": join_spec.join_column}


def build_camc(conn, meta: Mapping[str, Any], prepared: Mapping[str, Any],
               workload: str, factor_map: Mapping[str, Sequence[str]], h: int,
               seed: int) -> Tuple[NativeMCDB, Dict[str, Any], float, float, int]:
    engine = NativeMCDB(conn)
    mappings: Dict[str, Any] = {}
    sampling_s = 0.0
    encoding_s = 0.0
    fallback_symbols = 0
    if workload == "set" and meta.get("semantic_relations"):
        missing, ordering = relation_settings(meta, 0, workload, factor_map)
        available = set(columns(conn, prepared["pred_tables"][0]))
        ordering = {
            attr: tuple(value for value in separators if value in available)
            for attr, separators in ordering.items() if attr in available
        }
        for attr in missing:
            ordering.setdefault(attr, tuple())
        joint_bundle = engine.create_bundle(
            prepared["pred_tables"][0], missing, ordering, h,
            seed=float(seed % 1000) / 1000.0,
            prefix="camc_%s_%d" %
            (clean_name(prepared["pred_tables"][0]), h),
            strict=True,
            n_bins=5,
            factor_table=prepared["tables"][0],
        )
        sampling_s = joint_bundle.sampling_s
        encoding_s = joint_bundle.encoding_s
        fallback_symbols = sum(joint_bundle.unresolved_draws.values())
        projection_started = time.perf_counter()
        configured = prepared["tables"][0]
        csv_token = prepared["csvs"][0]
        mappings.update(relation_mapping(
            joint_bundle, configured, csv_token,
            os.path.basename(csv_token),
            os.path.splitext(os.path.basename(csv_token))[0],
        ))
        for position in (1, 2):
            projected = project_joint_bundle(
                conn,
                joint_bundle,
                prepared["pred_tables"][position],
                prepared["join_key"],
                clean_name(
                    "camc_relation_%d_%s" %
                    (position, prepared["pred_tables"][0])
                )[:55],
            )
            configured = prepared["tables"][position]
            csv_token = prepared["csvs"][position]
            mappings.update(relation_mapping(
                projected, configured, csv_token,
                os.path.basename(csv_token),
                os.path.splitext(os.path.basename(csv_token))[0],
            ))
        encoding_s += time.perf_counter() - projection_started
        return (
            engine, mappings, sampling_s, encoding_s, fallback_symbols
        )
    for position, table in enumerate(prepared["pred_tables"]):
        _configured_missing, ordering = relation_settings(meta, position, workload, factor_map)
        available = set(columns(conn, table))
        ordering = {attr: tuple(value for value in separators if value in available)
                    for attr, separators in ordering.items() if attr in available}
        missing = engine.missing_attributes(table)
        for attr in missing:
            ordering.setdefault(attr, tuple())
        bundle = engine.create_bundle(
            table, missing, ordering, h, seed=float(seed % 1000) / 1000.0,
            prefix="camc_%s_%d" % (clean_name(table), h), strict=True,
            n_bins=5,
            factor_table=prepared["tables"][position],
        )
        sampling_s += bundle.sampling_s
        encoding_s += bundle.encoding_s
        fallback_symbols += sum(bundle.unresolved_draws.values())
        configured = prepared["tables"][position]
        csv_token = prepared["csvs"][position]
        mappings.update(relation_mapping(bundle, configured, csv_token, os.path.basename(csv_token),
                                         os.path.splitext(os.path.basename(csv_token))[0]))
    return engine, mappings, sampling_s, encoding_s, fallback_symbols


def run_set_block(conn, dataset: str, block: str, meta: Mapping[str, Any],
                  prepared: Mapping[str, Any], h: int, timeout_s: int,
                  encoding_share: float, camc_engine: NativeMCDB,
                  camc_mappings: Mapping[str, Any], query_limit: Optional[int] = None,
                  completed_queries: Optional[Iterable[int]] = None,
                  checkpoint=None) -> List[Dict[str, Any]]:
    executor_meta = {block: {"csv": prepared["csvs"], "table": prepared["pred_tables"],
                             "complete_csv": meta["complete_csv"], "complete_table": prepared["full_tables"]}}
    caex = CAEXSetExecutor(conn, executor_meta, skip_prepare=True)
    caex.interval_mode = "delta"
    caex.interval_alpha = 0.05
    caex._ordering_T = meta.get("ordering_single") or {}
    caex._missing_T = meta.get("missing_attrs_single") or []
    caex._ordering_S = meta.get("ordering_S") or {}
    caex._missing_S = meta.get("missing_attrs_S") or []
    combined_ordering = {str(key).lower(): [str(value).lower() for value in values]
                         for key, values in (meta.get("ordering_T") or {}).items()}
    combined_ordering.update({str(key).lower(): [str(value).lower() for value in values]
                              for key, values in (meta.get("ordering_S") or {}).items()})
    combined_missing = [str(value).lower() for value in (meta.get("missing_attrs_T") or [])]
    combined_missing.extend(str(value).lower() for value in (meta.get("missing_attrs_S") or []))
    join_pred_view = create_join_view(conn, prepared["pred_tables"][1], prepared["pred_tables"][2],
                                      prepared["join_key"], clean_name("cade_set_" + block))
    join_truth_view = create_join_view(conn, prepared["full_tables"][1], prepared["full_tables"][2],
                                       prepared["join_key"], clean_name("cade_set_truth_" + block))
    join_executor_meta = {
        block: {"csv": [join_pred_view], "table": [join_pred_view],
                "complete_csv": [join_truth_view], "complete_table": [join_truth_view]}
    }
    caex_join = CAEXSetExecutor(conn, join_executor_meta, skip_prepare=True)
    caex_join.interval_mode = "delta"
    caex_join.interval_alpha = 0.05
    caex_join._ordering_T = combined_ordering
    caex_join._missing_T = combined_missing
    rows: List[Dict[str, Any]] = []
    queries = list(meta["queries"])
    if query_limit is not None:
        queries = queries[:query_limit]
    completed = set(completed_queries or [])
    for query_index, configured_query in enumerate(queries, 1):
        if query_index in completed:
            continue
        query = remove_redundant_join_grouping(configured_query)
        is_join = " JOIN " in query.upper()
        base = {"workload": "set", "dataset": dataset, "block": block,
                "rate": int(re.search(r"(5|10|20)$", block).group(1)),
                "query_index": query_index, "query": query, "h": h}
        pred_query = replace_tokens(query, prepared["pred_map"])
        truth_query = replace_tokens(query, prepared["truth_map"])
        if is_join:
            caex_executor = caex_join
            caex_pred_query = flatten_join_query(query, join_pred_view)
            grouped_truth_query = flatten_join_query(query, join_truth_view)
            caex_ordering = combined_ordering
            caex_missing = combined_missing
        else:
            caex_executor = caex
            caex_pred_query = pred_query
            grouped_truth_query = truth_query
            caex_ordering = caex._ordering_T
            caex_missing = caex._missing_T
        set_timeout(conn, 0)
        truth_set = execute_truth_set(conn, grouped_truth_query)
        set_timeout(conn, timeout_s)

        try:
            metrics, _pred_groups, _gt_groups = evaluate_query_with_groups(
                caex_executor, caex_pred_query, grouped_truth_query, caex_ordering, caex_missing
            )
            rows.append({**base, "method": "CAEX", "time_s": metrics["time_pred_s"],
                         "metric": metrics.get("tv_cond", metrics.get("tv_prob")),
                         "coverage": metrics.get("interval_coverage"),
                         "delta_w": metrics.get("normalized_interval_width"),
                         "status": "ok", "result_rows": None})
        except Exception as error:
            conn.rollback()
            rows.append(timeout_row(base, "CAEX", timeout_s, error))

        try:
            if is_join:
                cade_query = flatten_join_query(query, join_pred_view)
                cade_truth_query = flatten_join_query(query, join_truth_view)
                result = run_direct_per_tuple(conn, cade_query, join_pred_view, join_truth_view,
                                              combined_missing, combined_ordering, return_groups=True)
            else:
                result = run_direct_per_tuple(
                    conn, pred_query, prepared["pred_tables"][0], prepared["full_tables"][0],
                    [str(value).lower() for value in meta.get("missing_attrs_single", [])],
                    {str(key).lower(): [str(value).lower() for value in values]
                     for key, values in (meta.get("ordering_single") or {}).items()}, return_groups=True,
                )
            if result.get("error"):
                raise RuntimeError(result["error"])
            groups = result.get("groups", [])
            rows.append({**base, "method": "CADE", "time_s": result["sql_time_s"],
                         "metric": result.get("tv_prob"), "coverage": direct_coverage(groups),
                         "delta_w": result.get("delta_w"),
                         "status": "ok", "result_rows": result.get("n_pred")})
        except Exception as error:
            conn.rollback()
            rows.append(timeout_row(base, "CADE", timeout_s, error))

        try:
            elapsed, metric, coverage, n_rows, delta_w = run_camc_set(
                camc_engine, query, camc_mappings, truth_set, h
            )
            rows.append({**base, "method": "CAMC", "time_s": elapsed + encoding_share,
                         "query_time_s": elapsed, "encoding_share_s": encoding_share,
                         "metric": metric, "coverage": coverage, "delta_w": delta_w,
                         "status": "ok", "result_rows": n_rows})
        except Exception as error:
            conn.rollback()
            rows.append(timeout_row(base, "CAMC", timeout_s, error))
        print("%s set Q%d complete" % (block, query_index), flush=True)
        if checkpoint:
            checkpoint(rows)
    return rows


def run_agg_block(conn, dataset: str, block: str, meta: Mapping[str, Any],
                  prepared: Mapping[str, Any], factor_map: Mapping[str, Sequence[str]],
                  h: int, timeout_s: int, encoding_share: float,
                  camc_engine: NativeMCDB, camc_mappings: Mapping[str, Any],
                  query_limit: Optional[int] = None,
                  completed_queries: Optional[Iterable[int]] = None,
                  checkpoint=None) -> List[Dict[str, Any]]:
    pred_view = create_join_view(conn, prepared["pred_tables"][1], prepared["pred_tables"][2],
                                 prepared["join_key"], clean_name("agg_pred_" + block))
    truth_view = create_join_view(conn, prepared["full_tables"][1], prepared["full_tables"][2],
                                  prepared["join_key"], clean_name("agg_truth_" + block))
    rows: List[Dict[str, Any]] = []
    queries = list(meta["queries"])
    if query_limit is not None:
        queries = queries[:query_limit]
    completed = set(completed_queries or [])
    for query_index, configured_query in enumerate(queries, 1):
        if query_index in completed:
            continue
        query = expose_group_columns(configured_query)
        is_join = " JOIN " in query.upper()
        base = {"workload": "aggregate", "dataset": dataset, "block": block,
                "rate": int(re.search(r"(5|10|20)$", block).group(1)),
                "query_index": query_index, "query": query, "h": h}
        if is_join:
            analytic_query = flatten_join_query(query, pred_view)
            truth_query = flatten_join_query(query, truth_view)
            relation = pred_view
        else:
            analytic_query = replace_tokens(query, prepared["pred_map"])
            truth_query = replace_tokens(query, prepared["truth_map"])
            relation = prepared["pred_tables"][0]
        available = set(columns(conn, relation))
        ordering = {attr: tuple(value for value in separators if value in available)
                    for attr, separators in factor_map.items() if attr in available}
        engine = NativeMCDB(conn)
        active_missing = engine.missing_attributes(relation)
        set_timeout(conn, timeout_s)
        try:
            elapsed, metric, coverage, n_rows, delta_w = run_analytic_aggregate(
                conn, analytic_query, relation, truth_query, ordering, active_missing, False
            )
            rows.append({**base, "method": "CAEX", "time_s": elapsed, "metric": metric,
                         "coverage": coverage, "delta_w": delta_w,
                         "status": "ok", "result_rows": n_rows})
        except Exception as error:
            conn.rollback()
            rows.append(timeout_row(base, "CAEX", timeout_s, error))
        try:
            elapsed, metric, coverage, n_rows, delta_w = run_analytic_aggregate(
                conn, analytic_query, relation, truth_query, ordering, active_missing, True
            )
            rows.append({**base, "method": "CADE", "time_s": elapsed, "metric": metric,
                         "coverage": coverage, "delta_w": delta_w,
                         "status": "ok", "result_rows": n_rows})
        except Exception as error:
            conn.rollback()
            rows.append(timeout_row(base, "CADE", timeout_s, error))
        try:
            elapsed, metric, coverage, n_rows, delta_w = run_camc_aggregate(
                camc_engine, query, camc_mappings, conn, truth_query, h
            )
            rows.append({**base, "method": "CAMC", "time_s": elapsed + encoding_share,
                         "query_time_s": elapsed, "encoding_share_s": encoding_share,
                         "metric": metric, "coverage": coverage, "delta_w": delta_w,
                         "status": "ok", "result_rows": n_rows})
        except Exception as error:
            conn.rollback()
            rows.append(timeout_row(base, "CAMC", timeout_s, error))
        print("%s aggregate Q%d complete" % (block, query_index), flush=True)
        if checkpoint:
            checkpoint(rows)
    return rows


def block_for_rate(group: Mapping[str, Any], rate: int) -> Tuple[str, Mapping[str, Any]]:
    for name, meta in group.items():
        if re.search(r"(?:_|mnar)(%d)$" % rate, name, re.IGNORECASE):
            return name, meta
    raise KeyError("No block found for rate %d" % rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-config", default="configs/mnar_set_queries.json")
    parser.add_argument("--agg-config", default="configs/mnar1_agg_inj_query.json")
    parser.add_argument("--agg-extra", default="configs/section_comparison_agg_queries.json")
    parser.add_argument("--datasets", default="bank,nyc,bitcoin")
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--workloads", default="set,aggregate")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--h", type=int, default=783)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--output", default="psql_results/section_comparisons/section_comparison_results.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()
    datasets = [value.strip().lower() for value in args.datasets.split(",") if value.strip()]
    rates = [int(value) for value in args.rates.split(",") if value.strip()]
    workloads = [value.strip().lower() for value in args.workloads.split(",") if value.strip()]
    with open(args.set_config) as handle:
        set_config = json.load(handle)
    with open(args.agg_config) as handle:
        agg_config = json.load(handle)
    append_extra_aggregate_queries(agg_config, args.agg_extra)
    results: List[Dict[str, Any]] = read_csv(args.output) if args.resume else []
    conn = psycopg2.connect(**CONN)
    conn.autocommit = False
    try:
        for dataset in datasets:
            if dataset not in GROUPS:
                raise ValueError("Unknown dataset %s" % dataset)
            set_group, agg_group = GROUPS[dataset]
            factor_map = load_factor_map(SEPARATOR_FILES[dataset])
            for rate in rates:
                for workload in workloads:
                    group = set_config[set_group] if workload == "set" else agg_config[agg_group]
                    block, meta = block_for_rate(group, rate)
                    completed_queries = {
                        int(row["query_index"]) for row in results
                        if row.get("workload") == workload and row.get("dataset") == dataset
                        and int(row.get("rate", -1)) == rate
                        and sum(
                            1 for candidate in results
                            if candidate.get("workload") == workload and candidate.get("dataset") == dataset
                            and int(candidate.get("rate", -1)) == rate
                            and int(candidate.get("query_index", -1)) == int(row["query_index"])
                            and candidate.get("method") in METHODS
                        ) == len(METHODS)
                    }
                    expected_queries = min(args.query_limit or 10, 10)
                    if len(completed_queries) == expected_queries:
                        print("Skipping completed %s %s at %d%%" % (dataset, workload, rate), flush=True)
                        continue
                    print("Preparing %s %s at %d%%" % (dataset, workload, rate), flush=True)
                    set_timeout(conn, 0)
                    prepared = prepare_block(
                        conn,
                        meta,
                        dataset,
                        block,
                        workload,
                        args.rows,
                        args.seed,
                        force_reload=args.force_reload,
                    )
                    camc_engine, camc_mapping, sampling_s, encoding_s, fallback_symbols = build_camc(
                        conn, meta, prepared, workload, factor_map, args.h, args.seed
                    )
                    encoding_share = encoding_s / 10.0
                    print("CAMC setup: factor-sampling %.3fs excluded; encoding %.3fs amortized" %
                          (sampling_s, encoding_s), flush=True)
                    def checkpoint(partial_rows):
                        decorated = []
                        for partial in partial_rows:
                            item = dict(partial)
                            item["row_limit"] = args.rows
                            item["factor_sampling_s_excluded"] = sampling_s
                            item["total_encoding_s"] = encoding_s
                            item["factor_fallback_symbols"] = fallback_symbols
                            decorated.append(item)
                        write_csv(args.output, results + decorated)
                    if workload == "set":
                        block_rows = run_set_block(
                            conn, dataset, block, meta, prepared, args.h, args.timeout,
                            encoding_share, camc_engine, camc_mapping, args.query_limit,
                            completed_queries, checkpoint,
                        )
                    else:
                        block_rows = run_agg_block(
                            conn, dataset, block, meta, prepared, factor_map, args.h,
                            args.timeout, encoding_share, camc_engine, camc_mapping,
                            args.query_limit, completed_queries, checkpoint,
                        )
                    for row in block_rows:
                        row["row_limit"] = args.rows
                        row["factor_sampling_s_excluded"] = sampling_s
                        row["total_encoding_s"] = encoding_s
                        row["factor_fallback_symbols"] = fallback_symbols
                    results.extend(block_rows)
                    write_csv(args.output, results)
    finally:
        conn.close()
    write_csv(args.output, results)
    print("Saved %d measurements to %s" % (len(results), args.output), flush=True)


if __name__ == "__main__":
    main()
