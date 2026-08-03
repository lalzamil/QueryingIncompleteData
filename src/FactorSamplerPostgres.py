"""Exact PostgreSQL implementation of the paper's FactorSampler algorithm.

Python controls the ordered factors and recursion. Every random choice is
performed by PostgreSQL over the base relation with ``ORDER BY random()``.
The implementation does not discretize conditioning attributes and does not
fall back to an unconditional donor distribution.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from MCDBPostgresNative import (
    BundleRelation,
    CONN,
    _clean_name,
    _temp_name,
    qident,
)


class FactorSamplerError(RuntimeError):
    """Base class for errors raised by FactorSampler."""


class InvalidFactorization(FactorSamplerError):
    """The supplied factors cannot be processed in reverse order."""


class NoMatchingDonor(FactorSamplerError):
    """A conditional factor has no observed tuple to sample."""


@dataclasses.dataclass(frozen=True)
class Factor:
    """A factor P(Z | X), represented by its Z and X attributes."""

    z: Tuple[str, ...]
    x: Tuple[str, ...] = tuple()

    def __post_init__(self):
        object.__setattr__(self, "z", _attribute_tuple(self.z))
        object.__setattr__(self, "x", _attribute_tuple(self.x))
        if not self.z:
            raise InvalidFactorization("A factor must contain a Z attribute")
        if len(set(self.z)) != len(self.z):
            raise InvalidFactorization("A factor contains a duplicate Z attribute")
        if len(set(self.x)) != len(self.x):
            raise InvalidFactorization("A factor contains a duplicate X attribute")


@dataclasses.dataclass(frozen=True)
class FactorStep:
    """One donor selection made while processing a factor."""

    target: str
    factor_index: int
    sampled_attributes: Tuple[str, ...]
    conditioning_values: Tuple[Tuple[str, Any], ...]
    donor_rid: Any


@dataclasses.dataclass(frozen=True)
class FactorSample:
    """The sampled target value and the temporary tuple used to obtain it."""

    value: Any
    state: Mapping[str, Any]
    steps: Tuple[FactorStep, ...]


@dataclasses.dataclass(frozen=True)
class FactorSampleTables:
    """Per-attribute sample tables compatible with MCDB's bundle encoding."""

    base_table: str
    h: int
    attributes: Tuple[str, ...]
    sample_tables: Mapping[str, str]
    sampling_s: float


@dataclasses.dataclass(frozen=True)
class _Context:
    table: str
    columns: Tuple[str, ...]
    column_types: Mapping[str, str]
    factorizations: Mapping[str, Tuple[Factor, ...]]


def _attribute_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    if isinstance(values, str):
        return (values.lower(),)
    return tuple(str(value).lower() for value in values)


def factors(*pairs: Tuple[Sequence[str], Sequence[str]]) -> Tuple[Factor, ...]:
    """Build an ordered factorization from ``(Z, X)`` pairs."""

    return tuple(Factor(tuple(z), tuple(x)) for z, x in pairs)


class PostgresFactorSampler:
    """Run FactorSampler against one PostgreSQL relation at a time."""

    def __init__(self, connection):
        self.connection = connection

    @classmethod
    def connect(cls, **overrides):
        settings = dict(CONN)
        settings.update(overrides)
        return cls(psycopg2.connect(**settings))

    def close(self):
        self.connection.close()

    def set_seed(self, seed: float):
        if not -1.0 <= seed <= 1.0:
            raise ValueError("PostgreSQL random seed must be in [-1, 1]")
        cursor = self.connection.cursor()
        cursor.execute("SELECT setseed(%s)", (float(seed),))

    def sample_from_tuple(
        self,
        table: str,
        rid: Any,
        target: str,
        factorization: Sequence[Factor],
        factorization_by_attribute: Optional[Mapping[str, Sequence[Factor]]] = None,
    ) -> FactorSample:
        """Run FactorSampler for the target cell in one input tuple."""

        context = self._context(
            table, target, factorization, factorization_by_attribute or {}
        )
        row = self._row_by_rid(context, rid)
        if row[target.lower()] is not None:
            raise FactorSamplerError(
                "%s.%s is observed in tuple %s" % (table, target, rid)
            )
        steps: List[FactorStep] = []
        state = dict(row)
        value = self._sample_state(
            context, state, target.lower(), set(), steps
        )
        return FactorSample(value=value, state=dict(state), steps=tuple(steps))

    def sample_marked_null(
        self,
        table: str,
        attribute: str,
        symbol: str,
        factorization: Sequence[Factor],
        h: int,
        factorization_by_attribute: Optional[Mapping[str, Sequence[Factor]]] = None,
        seed: Optional[float] = None,
    ) -> Tuple[Any, ...]:
        """Sample one shared value for a marked null in each of H repairs."""

        if h <= 0:
            raise ValueError("h must be positive")
        attribute = attribute.lower()
        context = self._context(
            table, attribute, factorization, factorization_by_attribute or {}
        )
        if seed is not None:
            self.set_seed(seed)
        return self._sample_marked_null(context, attribute, str(symbol), h)

    def create_sample_tables(
        self,
        table: str,
        factorization_by_attribute: Mapping[str, Sequence[Factor]],
        h: int,
        prefix: Optional[str] = None,
        seed: Optional[float] = None,
    ) -> FactorSampleTables:
        """Sample every requested marked null and store one typed array per symbol."""

        if h <= 0:
            raise ValueError("h must be positive")
        if not factorization_by_attribute:
            raise ValueError("At least one target factorization is required")
        first_key = next(iter(factorization_by_attribute))
        first_target = first_key.lower()
        context = self._context(
            table,
            first_target,
            factorization_by_attribute[first_key],
            factorization_by_attribute,
        )
        attributes = tuple(
            attribute.lower()
            for attribute in factorization_by_attribute
            if self._has_missing_values(table, attribute.lower())
        )
        prefix = prefix or _temp_name("factor_sampler", table, str(h))
        if seed is not None:
            self.set_seed(seed)

        started = time.perf_counter()
        sample_tables: Dict[str, str] = {}
        cursor = self.connection.cursor()
        for attribute in attributes:
            symbols = self._symbols(context, attribute)
            sample_table = _temp_name(prefix, "samples", attribute)
            cursor.execute("DROP TABLE IF EXISTS %s" % qident(sample_table))
            cursor.execute(
                "CREATE TEMP TABLE %s ("
                "symbol text PRIMARY KEY, samples %s[] NOT NULL, "
                "used_fallback boolean NOT NULL DEFAULT false"
                ") ON COMMIT PRESERVE ROWS" % (
                    qident(sample_table), context.column_types[attribute]
                )
            )
            for symbol in symbols:
                sampled_values = self._sample_marked_null(
                    context, attribute, symbol, h
                )
                cursor.execute(
                    "INSERT INTO %s (symbol, samples, used_fallback) "
                    "VALUES (%%s, %%s::%s[], false)" % (
                        qident(sample_table), context.column_types[attribute]
                    ),
                    (symbol, list(sampled_values)),
                )
            sample_tables[attribute] = sample_table
        self.connection.commit()
        return FactorSampleTables(
            base_table=table,
            h=h,
            attributes=attributes,
            sample_tables=sample_tables,
            sampling_s=time.perf_counter() - started,
        )

    def create_bundle(
        self,
        table: str,
        factorization_by_attribute: Mapping[str, Sequence[Factor]],
        h: int,
        prefix: Optional[str] = None,
        seed: Optional[float] = None,
    ) -> BundleRelation:
        """Create an MCDB bundle using samples returned by FactorSampler."""

        prefix = prefix or _temp_name("factor_sampler", table, str(h))
        sampled = self.create_sample_tables(
            table, factorization_by_attribute, h, prefix, seed
        )
        first_key = next(iter(factorization_by_attribute))
        context = self._context(
            table,
            first_key.lower(),
            factorization_by_attribute[first_key],
            factorization_by_attribute,
        )
        cursor = self.connection.cursor()
        bundle_table = _temp_name(prefix, "bundle")
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(bundle_table))
        sample_columns: List[str] = []
        sample_joins: List[str] = []
        for attribute in sampled.attributes:
            alias = "sample_%s" % _clean_name(attribute)
            sample_columns.append(
                "%s.samples AS %s" % (
                    alias, qident("__samples_%s" % attribute)
                )
            )
            sample_joins.append(
                "LEFT JOIN %s %s ON b.%s IS NULL AND %s.symbol = %s" % (
                    qident(sampled.sample_tables[attribute]),
                    alias,
                    qident(attribute),
                    alias,
                    self._symbol_expression("b", attribute, context.columns),
                )
            )
        suffix = ", " + ", ".join(sample_columns) if sample_columns else ""
        encoding_started = time.perf_counter()
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT b.*%s, int4multirange(int4range(1, %d)) AS %s "
            "FROM %s b %s" % (
                qident(bundle_table),
                suffix,
                h + 1,
                qident("__present"),
                qident(table),
                " ".join(sample_joins),
            )
        )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (%s)" % (
                qident(bundle_table), qident("_rid")
            )
        )
        self.connection.commit()
        return BundleRelation(
            base_table=table,
            bundle_table=bundle_table,
            h=h,
            columns=context.columns,
            column_types=dict(context.column_types),
            missing_attributes=sampled.attributes,
            sample_tables=dict(sampled.sample_tables),
            unresolved_draws={attribute: 0 for attribute in sampled.attributes},
            sampling_s=sampled.sampling_s,
            encoding_s=time.perf_counter() - encoding_started,
        )

    def _context(
        self,
        table: str,
        target: str,
        factorization: Sequence[Factor],
        factorization_by_attribute: Mapping[str, Sequence[Factor]],
    ) -> _Context:
        columns, column_types = self._columns(table)
        registry = {
            attribute.lower(): self._normalize_factors(value)
            for attribute, value in factorization_by_attribute.items()
        }
        registry[target.lower()] = self._normalize_factors(factorization)
        for attribute, ordered_factors in registry.items():
            self._validate_factorization(
                attribute, ordered_factors, columns
            )
        return _Context(
            table=table,
            columns=columns,
            column_types=column_types,
            factorizations=registry,
        )

    @staticmethod
    def _normalize_factors(value: Sequence[Factor]) -> Tuple[Factor, ...]:
        normalized = []
        for item in value:
            if isinstance(item, Factor):
                normalized.append(item)
            else:
                z, x = item
                normalized.append(Factor(tuple(z), tuple(x)))
        if not normalized:
            raise InvalidFactorization("A factorization cannot be empty")
        return tuple(normalized)

    @staticmethod
    def _validate_factorization(
        target: str,
        ordered_factors: Sequence[Factor],
        columns: Sequence[str],
    ):
        available = set(columns)
        z_seen = set()
        for factor in ordered_factors:
            unknown = (set(factor.z) | set(factor.x)) - available
            if unknown:
                raise InvalidFactorization(
                    "Unknown factor attributes: %s" % ", ".join(sorted(unknown))
                )
            overlap = z_seen & set(factor.z)
            if overlap:
                raise InvalidFactorization(
                    "Z attributes occur in multiple factors: %s" %
                    ", ".join(sorted(overlap))
                )
            z_seen.update(factor.z)
        if target not in ordered_factors[0].z:
            raise InvalidFactorization(
                "Target %s must occur in the first factor" % target
            )
        for index, factor in enumerate(ordered_factors):
            later_z = {
                attribute
                for later in ordered_factors[index + 1:]
                for attribute in later.z
            }
            missing = set(factor.x) - later_z
            if missing:
                raise InvalidFactorization(
                    "Conditioning attributes must occur in later factors: %s" %
                    ", ".join(sorted(missing))
                )

    def _sample_marked_null(
        self,
        context: _Context,
        attribute: str,
        symbol: str,
        h: int,
    ) -> Tuple[Any, ...]:
        values: List[Any] = []
        for _index in range(1, h + 1):
            occurrence = self._select_occurrence(context, attribute, symbol)
            state = dict(occurrence)
            values.append(
                self._sample_state(context, state, attribute, set(), [])
            )
        return tuple(values)

    def _sample_state(
        self,
        context: _Context,
        state: Dict[str, Any],
        target: str,
        recursion: set,
        steps: List[FactorStep],
    ) -> Any:
        ordered_factors = context.factorizations.get(target)
        if ordered_factors is None:
            raise InvalidFactorization(
                "No factorization was supplied for %s" % target
            )
        for reverse_position, factor in enumerate(reversed(ordered_factors)):
            factor_index = len(ordered_factors) - reverse_position
            missing_z = [attribute for attribute in factor.z if state[attribute] is None]
            for attribute in tuple(missing_z):
                if attribute == target:
                    continue
                symbol = self._cell_symbol(state, attribute)
                if self._occurrence_count(context, attribute, symbol) <= 1:
                    continue
                key = (context.table, attribute, symbol)
                if key in recursion:
                    raise InvalidFactorization(
                        "Recursive marked-null dependency at %s.%s=%s" % key
                    )
                occurrence = self._select_occurrence(context, attribute, symbol)
                nested_state = dict(occurrence)
                state[attribute] = self._sample_state(
                    context,
                    nested_state,
                    attribute,
                    recursion | {key},
                    steps,
                )

            missing_z = [attribute for attribute in factor.z if state[attribute] is None]
            if not missing_z:
                continue
            observed_z = [attribute for attribute in factor.z if state[attribute] is not None]
            unresolved_x = [attribute for attribute in factor.x if state[attribute] is None]
            if unresolved_x:
                raise InvalidFactorization(
                    "Reverse factor order did not resolve: %s" %
                    ", ".join(unresolved_x)
                )
            donor = self._select_donor(
                context, missing_z, factor.x, observed_z, state
            )
            for attribute in missing_z:
                state[attribute] = donor[attribute]
            conditions = tuple(
                (attribute, state[attribute])
                for attribute in tuple(factor.x) + tuple(observed_z)
            )
            steps.append(
                FactorStep(
                    target=target,
                    factor_index=factor_index,
                    sampled_attributes=tuple(missing_z),
                    conditioning_values=conditions,
                    donor_rid=donor["_rid"],
                )
            )
        if state[target] is None:
            raise InvalidFactorization(
                "Factorization did not produce a value for %s" % target
            )
        return state[target]

    def _select_donor(
        self,
        context: _Context,
        missing_z: Sequence[str],
        x_attributes: Sequence[str],
        observed_z: Sequence[str],
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        predicates = [
            "b.%s IS NOT NULL" % qident(attribute)
            for attribute in missing_z
        ]
        parameters: List[Any] = []
        for attribute in tuple(x_attributes) + tuple(observed_z):
            predicates.append(
                "b.%s IS NOT DISTINCT FROM %%s" % qident(attribute)
            )
            parameters.append(state[attribute])
        selected = ["b.%s AS %s" % (qident("_rid"), qident("_rid"))]
        selected.extend(
            "b.%s AS %s" % (qident(attribute), qident(attribute))
            for attribute in missing_z
        )
        sql = (
            "SELECT %s FROM %s b WHERE %s ORDER BY random() LIMIT 1" % (
                ", ".join(selected),
                qident(context.table),
                " AND ".join(predicates),
            )
        )
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql, tuple(parameters))
        row = cursor.fetchone()
        if row is None:
            conditions = [
                "%s=%r" % (attribute, state[attribute])
                for attribute in tuple(x_attributes) + tuple(observed_z)
            ]
            raise NoMatchingDonor(
                "No donor for %s with %s" % (
                    ",".join(missing_z),
                    ", ".join(conditions) if conditions else "no conditions",
                )
            )
        return dict(row)

    def _select_occurrence(
        self, context: _Context, attribute: str, symbol: str
    ) -> Mapping[str, Any]:
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT b.* FROM %s b WHERE b.%s IS NULL AND %s = %%s "
            "ORDER BY random() LIMIT 1" % (
                qident(context.table),
                qident(attribute),
                self._symbol_expression("b", attribute, context.columns),
            ),
            (symbol,),
        )
        row = cursor.fetchone()
        if row is None:
            raise FactorSamplerError(
                "No occurrence of %s in %s.%s" % (
                    symbol, context.table, attribute
                )
            )
        return dict(row)

    def _occurrence_count(
        self, context: _Context, attribute: str, symbol: str
    ) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT count(*) FROM %s b WHERE b.%s IS NULL AND %s = %%s" % (
                qident(context.table),
                qident(attribute),
                self._symbol_expression("b", attribute, context.columns),
            ),
            (symbol,),
        )
        return int(cursor.fetchone()[0])

    def _symbols(self, context: _Context, attribute: str) -> Tuple[str, ...]:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT DISTINCT %s AS symbol FROM %s b "
            "WHERE b.%s IS NULL ORDER BY symbol" % (
                self._symbol_expression("b", attribute, context.columns),
                qident(context.table),
                qident(attribute),
            )
        )
        return tuple(str(row[0]) for row in cursor.fetchall())

    def _row_by_rid(self, context: _Context, rid: Any) -> Mapping[str, Any]:
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT * FROM %s WHERE %s = %%s" % (
                qident(context.table), qident("_rid")
            ),
            (rid,),
        )
        row = cursor.fetchone()
        if row is None:
            raise FactorSamplerError(
                "Tuple %s does not exist in %s" % (rid, context.table)
            )
        return dict(row)

    def _columns(self, table: str) -> Tuple[Tuple[str, ...], Dict[str, str]]:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod)
            FROM pg_catalog.pg_attribute a
            WHERE a.attrelid = %s::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (table,),
        )
        rows = cursor.fetchall()
        if not rows:
            raise FactorSamplerError("Relation %s does not exist" % table)
        columns = tuple(row[0].lower() for row in rows)
        if "_rid" not in columns:
            raise FactorSamplerError(
                "Relation %s requires a unique _rid column" % table
            )
        return columns, {row[0].lower(): row[1] for row in rows}

    def _has_missing_values(self, table: str, attribute: str) -> bool:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM %s WHERE %s IS NULL)" % (
                qident(table), qident(attribute)
            )
        )
        return bool(cursor.fetchone()[0])

    @staticmethod
    def _symbol_expression(
        alias: str, attribute: str, columns: Sequence[str]
    ) -> str:
        symbol_column = "%s_nullsym" % attribute
        row_symbol = "'__row_' || %s.%s::text" % (alias, qident("_rid"))
        if symbol_column in columns:
            return "COALESCE(NULLIF(%s.%s::text, ''), %s)" % (
                alias, qident(symbol_column), row_symbol
            )
        return row_symbol

    @staticmethod
    def _cell_symbol(state: Mapping[str, Any], attribute: str) -> str:
        symbol = state.get("%s_nullsym" % attribute)
        if symbol is not None and str(symbol) != "":
            return str(symbol)
        return "__row_%s" % state["_rid"]


__all__ = [
    "Factor",
    "FactorSample",
    "FactorSampleTables",
    "FactorSamplerError",
    "FactorStep",
    "InvalidFactorization",
    "NoMatchingDonor",
    "PostgresFactorSampler",
    "factors",
]
