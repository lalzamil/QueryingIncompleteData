"""Factor-distribution input for the union-based QE baseline."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from LikeApxRewrittenPostgres import RewrittenLikeApx
from MCDBPostgresNative import NativeMCDB, _clean_name, _temp_name, qident


@dataclasses.dataclass(frozen=True)
class FactorDistribution:
    occurrence_table: str
    symbol_table: str
    distribution_table: str
    seed: int


@dataclasses.dataclass
class QERelation:
    base_table: str
    h: int
    columns: Tuple[str, ...]
    column_types: Dict[str, str]
    missing_attributes: Tuple[str, ...]
    distributions: Dict[str, FactorDistribution]
    factorization_s: float = 0.0
    fallback_occurrences: int = 0

    def has_column(self, column: str) -> bool:
        return column.lower() in self.column_types

    def is_random(self, column: str) -> bool:
        return column.lower() in self.missing_attributes


class FactorDistributionBuilder:
    """Materialize each missing-value distribution from factor queries."""

    def __init__(self, engine: NativeMCDB):
        self.engine = engine
        self.connection = engine.connection

    @staticmethod
    def _context_expression(alias: str, separators: Sequence[str], bins,
                            included: Optional[Iterable[str]] = None) -> str:
        included_set = set(separators if included is None else included)
        values = []
        for separator in separators:
            if separator not in included_set:
                values.append("NULL::text")
            elif separator in bins:
                values.append("%s.bin::text" % bins[separator])
            else:
                values.append("%s.%s::text" % (alias, qident(separator)))
        return "jsonb_build_array(%s)::text" % ", ".join(values)

    @staticmethod
    def _pattern_expression(alias: str, separators: Sequence[str]) -> str:
        terms = [
            "CASE WHEN %s.%s IS NULL THEN 0 ELSE %d END" % (
                alias, qident(separator), 1 << position,
            )
            for position, separator in enumerate(separators)
        ]
        return " + ".join(terms) if terms else "0"

    @staticmethod
    def _bin_joins(source: str, separators: Iterable[str], bin_tables,
                   prefix: str):
        joins = []
        aliases = {}
        for separator in separators:
            if separator not in bin_tables:
                continue
            alias = "%s_%s" % (prefix, _clean_name(separator))
            aliases[separator] = alias
            joins.append(
                "LEFT JOIN %s %s ON %s.value = %s.%s" % (
                    qident(bin_tables[separator]), alias, alias, source,
                    qident(separator),
                )
            )
        return joins, aliases

    def _create_distribution(self, table: str, factor_table: str,
                             columns: Sequence[str], column_types,
                             attribute: str, separators: Sequence[str],
                             bin_tables, prefix: str,
                             seed: int, allow_fallback: bool = False
                             ) -> Tuple[FactorDistribution, int]:
        cursor = self.connection.cursor()
        occurrence_table = _temp_name(prefix, "occurrences", attribute)
        symbol_table = _temp_name(prefix, "symbols", attribute)
        distribution_table = _temp_name(prefix, "distributions", attribute)
        symbol = NativeMCDB._symbol_expression("b", attribute, columns)
        request_joins, request_bins = self._bin_joins(
            "b", separators, bin_tables, "request_bin"
        )
        pattern = self._pattern_expression("b", separators)
        context = self._context_expression(
            "b", separators, request_bins,
        )
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(occurrence_table))
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT b.%s AS rid, %s AS symbol, (%s)::integer AS pattern, "
            "%s AS context_key FROM %s b %s WHERE b.%s IS NULL" % (
                qident(occurrence_table), qident("_rid"), symbol, pattern,
                context, qident(table), " ".join(request_joins),
                qident(attribute),
            )
        )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (rid)" % qident(occurrence_table)
        )
        cursor.execute(
            "CREATE INDEX ON %s (symbol)" % qident(occurrence_table)
        )
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(symbol_table))
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT symbol, array_agg(rid ORDER BY rid) AS occurrence_rids "
            "FROM %s GROUP BY symbol" % (
                qident(symbol_table), qident(occurrence_table),
            )
        )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (symbol)" % qident(symbol_table)
        )
        cursor.execute("SELECT DISTINCT pattern FROM %s ORDER BY pattern" %
                       qident(occurrence_table))
        patterns = [int(row[0]) for row in cursor.fetchall()]
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(distribution_table))
        cursor.execute(
            "CREATE TEMP TABLE %s (pattern integer NOT NULL, "
            "context_key text NOT NULL, donor_values %s[] NOT NULL) "
            "ON COMMIT PRESERVE ROWS" % (
                qident(distribution_table), column_types[attribute],
            )
        )
        for value in patterns:
            included = [
                separator for position, separator in enumerate(separators)
                if value & (1 << position)
            ]
            donor_joins, donor_bins = self._bin_joins(
                "d", included, bin_tables, "donor_bin_%d" % value
            )
            donor_context = self._context_expression(
                "d", separators, donor_bins, included,
            )
            conditions = ["d.%s IS NOT NULL" % qident(attribute)]
            conditions.extend(
                "d.%s IS NOT NULL" % qident(separator)
                for separator in included
            )
            cursor.execute(
                "INSERT INTO %s (pattern, context_key, donor_values) "
                "SELECT %d, %s, array_agg(d.%s ORDER BY d.ctid)::%s[] "
                "FROM %s d %s WHERE %s GROUP BY %s" % (
                    qident(distribution_table), value, donor_context,
                    qident(attribute), column_types[attribute],
                    qident(factor_table), " ".join(donor_joins),
                    " AND ".join(conditions), donor_context,
                )
            )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (pattern, context_key)" %
            qident(distribution_table)
        )
        cursor.execute(
            "SELECT count(*) FROM %s o LEFT JOIN %s d ON "
            "d.pattern=o.pattern AND d.context_key=o.context_key "
            "WHERE d.context_key IS NULL" % (
                qident(occurrence_table), qident(distribution_table),
            )
        )
        unresolved = int(cursor.fetchone()[0])
        if unresolved and not allow_fallback:
            raise ValueError(
                "%d occurrences of %s have no computed factor distribution" %
                (unresolved, attribute)
            )
        if unresolved:
            cursor.execute(
                "WITH global_distribution AS ("
                "SELECT array_agg(%s ORDER BY ctid)::%s[] AS donor_values "
                "FROM %s WHERE %s IS NOT NULL), missing_contexts AS ("
                "SELECT DISTINCT o.pattern, o.context_key FROM %s o "
                "LEFT JOIN %s d ON d.pattern=o.pattern AND "
                "d.context_key=o.context_key WHERE d.context_key IS NULL) "
                "INSERT INTO %s (pattern, context_key, donor_values) "
                "SELECT m.pattern, m.context_key, g.donor_values "
                "FROM missing_contexts m CROSS JOIN global_distribution g" % (
                    qident(attribute), column_types[attribute],
                    qident(factor_table), qident(attribute),
                    qident(occurrence_table), qident(distribution_table),
                    qident(distribution_table),
                )
            )
        for table_name in (occurrence_table, symbol_table, distribution_table):
            cursor.execute("ANALYZE %s" % qident(table_name))
        digest = hashlib.sha1(
            ("%s|%s|%s" % (seed, table, attribute)).encode("utf-8")
        ).digest()
        draw_seed = int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
        return FactorDistribution(
            occurrence_table=occurrence_table,
            symbol_table=symbol_table,
            distribution_table=distribution_table,
            seed=draw_seed,
        ), unresolved

    def create_relation(self, table: str, factor_table: str,
                        missing_attributes: Iterable[str], ordering,
                        h: int, seed: int, prefix: str,
                        n_bins: int = 5,
                        allow_fallback: bool = False) -> QERelation:
        columns, column_types = self.engine._columns(table)
        factor_columns, factor_types = self.engine._columns(factor_table)
        active = self.engine._active_missing(
            table, missing_attributes, columns
        )
        normalized = {
            key.lower(): tuple(
                value.lower() for value in values
                if value.lower() in columns and value.lower() in factor_columns
            )
            for key, values in (ordering or {}).items()
            if key.lower() in columns
        }
        all_separators = [
            separator for attribute in active
            for separator in normalized.get(attribute, tuple())
        ]
        started = time.perf_counter()
        bin_tables = self.engine._create_separator_bins(
            factor_table, all_separators, factor_types, prefix, n_bins
        )
        distributions = {}
        fallback_occurrences = 0
        for attribute in active:
            distribution, unresolved = self._create_distribution(
                table, factor_table, columns, column_types, attribute,
                normalized.get(attribute, tuple()), bin_tables, prefix, seed,
                allow_fallback=allow_fallback,
            )
            distributions[attribute] = distribution
            fallback_occurrences += unresolved
        self.connection.commit()
        return QERelation(
            base_table=table,
            h=h,
            columns=columns,
            column_types=column_types,
            missing_attributes=active,
            distributions=distributions,
            factorization_s=time.perf_counter() - started,
            fallback_occurrences=fallback_occurrences,
        )


class FactorDistributionQE(RewrittenLikeApx):
    """Instantiate each QE repair from materialized factor distributions."""

    @staticmethod
    def create_relation(relation: QERelation,
                        _prefix: Optional[str] = None) -> QERelation:
        return relation

    @staticmethod
    def _draw_index(symbol: str, idx: int, salt: str, seed: int,
                    values: str) -> str:
        key = "(%s) || '|%d|%s'" % (symbol, idx, salt)
        return (
            "1 + ((hashtextextended(%s, %d) & "
            "9223372036854775807::bigint) %% cardinality(%s))::integer"
        ) % (key, seed, values)

    def _repair_sql(self, relation: QERelation, base_alias: str,
                    needed: Sequence[str], idx: int) -> str:
        joins = []
        values = []
        for column in needed:
            if relation.is_random(column):
                source = relation.distributions[column]
                stem = "%s_%s" % (base_alias, _clean_name(column))
                symbol_alias = "sym_%s" % stem
                occurrence_alias = "occ_%s" % stem
                distribution_alias = "dist_%s" % stem
                symbol = NativeMCDB._symbol_expression(
                    base_alias, column, relation.columns
                )
                occurrence_index = self._draw_index(
                    "%s.symbol" % symbol_alias, idx,
                    "occurrence_%s" % column, source.seed,
                    "%s.occurrence_rids" % symbol_alias,
                )
                joins.append(
                    "LEFT JOIN %s %s ON %s.%s IS NULL AND %s.symbol=%s" % (
                        qident(source.symbol_table), symbol_alias, base_alias,
                        qident(column), symbol_alias, symbol,
                    )
                )
                joins.append(
                    "LEFT JOIN %s %s ON %s.rid="
                    "%s.occurrence_rids[%s]" % (
                        qident(source.occurrence_table), occurrence_alias,
                        occurrence_alias, symbol_alias, occurrence_index,
                    )
                )
                joins.append(
                    "LEFT JOIN %s %s ON %s.pattern=%s.pattern AND "
                    "%s.context_key=%s.context_key" % (
                        qident(source.distribution_table), distribution_alias,
                        distribution_alias, occurrence_alias,
                        distribution_alias, occurrence_alias,
                    )
                )
                donor_index = self._draw_index(
                    "%s.symbol" % symbol_alias, idx, column, source.seed,
                    "%s.donor_values" % distribution_alias,
                )
                expression = "COALESCE(%s.%s, %s.donor_values[%s])" % (
                    base_alias, qident(column), distribution_alias, donor_index,
                )
            else:
                expression = "%s.%s" % (base_alias, qident(column))
            values.append("%s AS %s" % (expression, qident(column)))
        suffix = ", " + ", ".join(values) if values else ""
        return (
            "SELECT %d::integer AS idx, %s.%s AS rid%s FROM %s %s %s" % (
                idx, base_alias, qident("_rid"), suffix,
                qident(relation.base_table), base_alias, " ".join(joins),
            )
        )


def relation_mapping(relation: QERelation, *tokens: str):
    result = {relation.base_table: relation}
    for token in tokens:
        result[token] = relation
        result[os.path.basename(token)] = relation
        result[token.lower()] = relation
        result[os.path.basename(token).lower()] = relation
    return result


__all__ = [
    "FactorDistributionBuilder",
    "FactorDistributionQE",
    "QERelation",
    "relation_mapping",
]
