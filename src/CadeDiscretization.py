"""Discretization helpers for continuous separating-set attributes.

Section 4 requires a continuous attribute in a separating set to be
discretized before a factor query groups by it.  The helpers here compute
equal-frequency boundaries once per PostgreSQL relation and expose temporary
bin columns to the CADE aggregation SQL.  Query predicates, output columns,
and aggregate values continue to use the original columns.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Dict, Mapping, Sequence, Tuple


CADE_N_BINS = 20
NUMERIC_TYPES = {
    "int2",
    "int4",
    "int8",
    "float4",
    "float8",
    "numeric",
}

_BIN_CACHE: Dict[
    Tuple[int, str, Tuple[Tuple[str, Tuple[str, ...]], ...], int],
    Dict[str, Tuple[float, ...]],
] = {}
_MATERIALIZED_CACHE: Dict[
    Tuple[int, str, Tuple[Tuple[str, Tuple[str, ...]], ...], int],
    Tuple[str, Dict[str, Tuple[str, ...]]],
] = {}
_PATTERN_CACHE: Dict[
    Tuple[int, str, Tuple[Tuple[str, ...], ...]],
    Dict[Tuple[str, ...], Tuple[int, ...]],
] = {}


def qident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def bin_column(attribute: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", attribute).strip("_").lower()
    return "__cade_bin_" + safe


def _relation_columns(connection, relation: str) -> Dict[str, str]:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM %s LIMIT 0" % qident(relation))
    descriptions = cursor.description or ()
    type_oids = sorted({int(column.type_code) for column in descriptions})
    if not type_oids:
        return {}
    cursor.execute(
        "SELECT oid, typname FROM pg_type WHERE oid = ANY(%s)",
        (type_oids,),
    )
    names = {int(oid): str(name) for oid, name in cursor.fetchall()}
    return {
        str(column.name).lower(): names.get(int(column.type_code), "")
        for column in descriptions
    }


def _unique_boundaries(values) -> Tuple[float, ...]:
    result = []
    for value in values or ():
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        if not result or not math.isclose(
            numeric,
            result[-1],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            result.append(numeric)
    return tuple(result)


def prepare_cade_bins(
    connection,
    relation: str,
    orderings: Mapping[str, Sequence[str]],
    n_bins: int = CADE_N_BINS,
) -> Dict[str, Tuple[float, ...]]:
    """Return equal-frequency boundaries for numeric separating attributes."""
    normalized = tuple(
        sorted(
            (
                str(attribute).lower(),
                tuple(str(value).lower() for value in separators),
            )
            for attribute, separators in orderings.items()
        )
    )
    cache_key = (id(connection), relation, normalized, int(n_bins))
    if cache_key in _BIN_CACHE:
        return _BIN_CACHE[cache_key]

    column_types = _relation_columns(connection, relation)
    separators = sorted(
        {
            str(separator).lower()
            for _attribute, values in normalized
            for separator in values
            if str(separator).lower() in column_types
        }
    )
    probabilities = [index / float(n_bins) for index in range(1, n_bins)]
    probability_sql = "ARRAY[%s]::double precision[]" % ", ".join(
        "%.12g" % value for value in probabilities
    )
    result: Dict[str, Tuple[float, ...]] = {}
    started = time.perf_counter()
    cursor = connection.cursor()
    for attribute in separators:
        if column_types.get(attribute) not in NUMERIC_TYPES:
            continue
        identifier = qident(attribute)
        cursor.execute(
            "SELECT COUNT(DISTINCT {column}), "
            "percentile_cont({probabilities}) WITHIN GROUP "
            "(ORDER BY {column}::double precision) "
            "FROM {relation} WHERE {column} IS NOT NULL".format(
                column=identifier,
                probabilities=probability_sql,
                relation=qident(relation),
            )
        )
        distinct_count, boundaries = cursor.fetchone()
        if int(distinct_count or 0) > n_bins:
            unique = _unique_boundaries(boundaries)
            if unique:
                result[attribute] = unique
    elapsed = time.perf_counter() - started
    _BIN_CACHE[cache_key] = result
    print(
        "Prepared CADE separator bins for %s in %.3fs: %s"
        % (
            relation,
            elapsed,
            ", ".join(
                "%s=%d" % (attribute, len(boundaries) + 1)
                for attribute, boundaries in sorted(result.items())
            )
            or "none",
        ),
        flush=True,
    )
    return result


def discretized_orderings(
    orderings: Mapping[str, Sequence[str]],
    boundaries: Mapping[str, Sequence[float]],
) -> Dict[str, Tuple[str, ...]]:
    return {
        str(attribute).lower(): tuple(
            bin_column(str(separator).lower())
            if str(separator).lower() in boundaries
            else str(separator).lower()
            for separator in separators
        )
        for attribute, separators in orderings.items()
    }


def source_cte(
    relation: str,
    boundaries: Mapping[str, Sequence[float]],
    name: str = "cade_source",
) -> str:
    expressions = ["s.*"]
    for attribute, values in sorted(boundaries.items()):
        column = "s.%s" % qident(attribute)
        clauses = [
            "WHEN {column}::double precision <= {boundary:.17g} THEN {index}".format(
                column=column,
                boundary=float(boundary),
                index=index,
            )
            for index, boundary in enumerate(values)
        ]
        expressions.append(
            "CASE WHEN {column} IS NULL THEN NULL {clauses} ELSE {last} END AS {alias}".format(
                column=column,
                clauses=" ".join(clauses),
                last=len(values),
                alias=qident(bin_column(attribute)),
            )
        )
    return "%s AS NOT MATERIALIZED (SELECT %s FROM %s s)" % (
        name,
        ", ".join(expressions),
        qident(relation),
    )


def materialized_cade_source(
    connection,
    relation: str,
    orderings: Mapping[str, Sequence[str]],
    n_bins: int = CADE_N_BINS,
) -> Tuple[str, Dict[str, Tuple[str, ...]]]:
    """Materialize reusable bin columns for one prepared relation.

    The returned relation retains every original column. Numeric separating
    attributes additionally have equal-frequency bin columns. The returned
    orderings refer to those columns, so factor queries do not reevaluate the
    bin expressions while scanning and joining the relation.
    """
    normalized = tuple(
        sorted(
            (
                str(attribute).lower(),
                tuple(str(value).lower() for value in separators),
            )
            for attribute, separators in orderings.items()
        )
    )
    cache_key = (id(connection), relation, normalized, int(n_bins))
    if cache_key in _MATERIALIZED_CACHE:
        return _MATERIALIZED_CACHE[cache_key]

    boundaries = prepare_cade_bins(connection, relation, orderings, n_bins)
    effective_orderings = discretized_orderings(orderings, boundaries)
    if not boundaries:
        result = (relation, effective_orderings)
        _MATERIALIZED_CACHE[cache_key] = result
        return result

    signature = repr((relation, normalized, int(n_bins))).encode("utf-8")
    digest = hashlib.sha1(signature).hexdigest()[:10]
    safe_relation = re.sub(r"[^a-zA-Z0-9_]+", "_", relation).strip("_").lower()
    prepared_relation = ("cade_bins_%s_%s" % (safe_relation[:38], digest))[:63]

    expressions = [
        "s.%s" % qident(column)
        for column in _relation_columns(connection, relation)
        if not column.endswith("_nullsym")
    ]
    for attribute, values in sorted(boundaries.items()):
        column = "s.%s" % qident(attribute)
        clauses = [
            "WHEN {column}::double precision <= {boundary:.17g} THEN {index}".format(
                column=column,
                boundary=float(boundary),
                index=index,
            )
            for index, boundary in enumerate(values)
        ]
        expressions.append(
            "CASE WHEN {column} IS NULL THEN NULL {clauses} ELSE {last} END AS {alias}".format(
                column=column,
                clauses=" ".join(clauses),
                last=len(values),
                alias=qident(bin_column(attribute)),
            )
        )

    cursor = connection.cursor()
    started = time.perf_counter()
    cursor.execute("DROP TABLE IF EXISTS %s" % qident(prepared_relation))
    cursor.execute(
        "CREATE UNLOGGED TABLE %s AS SELECT %s FROM %s s"
        % (
            qident(prepared_relation),
            ", ".join(expressions),
            qident(relation),
        )
    )

    index_columns = set()
    for attribute, separators in effective_orderings.items():
        columns = tuple(dict.fromkeys(tuple(separators) + (attribute,)))
        if columns:
            index_columns.add(columns)
    for columns in sorted(index_columns):
        cursor.execute(
            "CREATE INDEX ON %s (%s)"
            % (
                qident(prepared_relation),
                ", ".join(qident(column) for column in columns),
            )
        )
    cursor.execute("ANALYZE %s" % qident(prepared_relation))
    connection.commit()
    elapsed = time.perf_counter() - started
    print(
        "Materialized CADE separator bins for %s as %s in %.3fs"
        % (relation, prepared_relation, elapsed),
        flush=True,
    )
    result = (prepared_relation, effective_orderings)
    _MATERIALIZED_CACHE[cache_key] = result
    return result


def observed_grouping_patterns(
    connection,
    relation: str,
    orderings: Mapping[str, Sequence[str]],
) -> Dict[Tuple[str, ...], Tuple[int, ...]]:
    """Return the separator-missingness patterns present in a relation."""
    separator_sets = tuple(
        sorted(
            {
                tuple(str(value).lower() for value in separators)
                for separators in orderings.values()
                if separators
            }
        )
    )
    cache_key = (id(connection), relation, separator_sets)
    if cache_key in _PATTERN_CACHE:
        return _PATTERN_CACHE[cache_key]
    if not separator_sets:
        _PATTERN_CACHE[cache_key] = {}
        return {}

    expressions = []
    for separators in separator_sets:
        terms = [
            "CASE WHEN %s IS NULL THEN %d ELSE 0 END"
            % (
                qident(column),
                1 << (len(separators) - index - 1),
            )
            for index, column in enumerate(separators)
        ]
        expressions.append(
            "ARRAY_AGG(DISTINCT (%s))" % (" + ".join(terms) or "0")
        )
    cursor = connection.cursor()
    started = time.perf_counter()
    cursor.execute(
        "SELECT %s FROM %s" % (
            ", ".join(expressions),
            qident(relation),
        )
    )
    values = cursor.fetchone()
    result = {
        separators: tuple(sorted(int(value) for value in (patterns or (0,))))
        for separators, patterns in zip(separator_sets, values)
    }
    elapsed = time.perf_counter() - started
    print(
        "Prepared CADE separator patterns for %s in %.3fs: %s"
        % (
            relation,
            elapsed,
            ", ".join(
                "%d columns=%d patterns" % (len(separators), len(patterns))
                for separators, patterns in result.items()
            ),
        ),
        flush=True,
    )
    _PATTERN_CACHE[cache_key] = result
    return result
