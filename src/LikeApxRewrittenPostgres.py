"""PostgreSQL implementation of the Section 6.3 LikeApx query encoding.

For sampled valuations v_1, ..., v_H, this module constructs one complete
rewritten query for each valuation and combines their answers with UNION ALL.
The sampled values are the same PostgreSQL-generated values used by the MCDB
comparison.  The reported result is the presence probability of each output
tuple, namely COUNT(idx) / H after duplicate tuples within a valuation have
been removed.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

from MCDBPostgresNative import (
    BundleRelation,
    NativeMCDB,
    QueryResult,
    QuerySpec,
    _clean_name,
    literal,
    parse_query,
    qident,
)


@dataclasses.dataclass
class RewrittenLikeApxRelation:
    base_table: str
    sample_tables: Dict[str, str]
    h: int
    columns: Tuple[str, ...]
    column_types: Dict[str, str]
    missing_attributes: Tuple[str, ...]
    encoding_s: float = 0.0

    def has_column(self, column: str) -> bool:
        return column.lower() in self.column_types

    def is_random(self, column: str) -> bool:
        return column.lower() in self.missing_attributes


class RewrittenLikeApx:
    """Compile H valuation-specific copies of a query and union the answers."""

    def __init__(self, engine: NativeMCDB):
        self.engine = engine
        self.connection = engine.connection

    @staticmethod
    def create_relation(bundle: BundleRelation,
                        _prefix: Optional[str] = None) -> RewrittenLikeApxRelation:
        # Each sample table is a normalized Rand_D relation: one row per
        # marked null and one typed array containing its H sampled values.
        return RewrittenLikeApxRelation(
            base_table=bundle.base_table,
            sample_tables=dict(bundle.sample_tables),
            h=bundle.h,
            columns=bundle.columns,
            column_types=dict(bundle.column_types),
            missing_attributes=bundle.missing_attributes,
        )

    @staticmethod
    def _resolve_relation(
        token: str,
        relations: Mapping[str, RewrittenLikeApxRelation],
    ) -> RewrittenLikeApxRelation:
        options = (
            token,
            os.path.basename(token),
            token.lower(),
            os.path.basename(token).lower(),
        )
        for option in options:
            if option in relations:
                return relations[option]
        raise KeyError("No rewritten LikeApx relation supplied for %s" % token)

    def _repair_sql(
        self,
        relation: RewrittenLikeApxRelation,
        base_alias: str,
        needed: Sequence[str],
        idx: int,
    ) -> str:
        joins = []
        values = []
        for column in needed:
            if relation.is_random(column):
                sample_alias = "rand_%s_%s" % (
                    base_alias, _clean_name(column)
                )
                symbol = NativeMCDB._symbol_expression(
                    base_alias, column, relation.columns
                )
                joins.append(
                    "LEFT JOIN %s %s ON %s.%s IS NULL "
                    "AND %s.symbol = %s" % (
                        qident(relation.sample_tables[column]), sample_alias,
                        base_alias, qident(column), sample_alias, symbol,
                    )
                )
                expression = "COALESCE(%s.%s, %s.samples[%d])" % (
                    base_alias, qident(column), sample_alias, idx,
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

    def _world_sql(
        self,
        spec: QuerySpec,
        left: RewrittenLikeApxRelation,
        right: Optional[RewrittenLikeApxRelation],
        needed: Sequence[str],
        idx: int,
    ) -> str:
        left_columns = [column for column in needed if left.has_column(column)]
        if spec.join_column and spec.join_column not in left_columns:
            left_columns.append(spec.join_column)
        left_sql = self._repair_sql(left, "lb", left_columns, idx)

        if not right:
            selections = ", ".join(
                "l.%s AS %s" % (qident(column), qident(column))
                for column in needed
            )
            suffix = ", " + selections if selections else ""
            return "SELECT l.idx%s FROM (%s) l" % (suffix, left_sql)

        if left.h != right.h:
            raise ValueError("LikeApx relations must use the same H")
        right_columns = [column for column in needed if right.has_column(column)]
        if spec.join_column and spec.join_column not in right_columns:
            right_columns.append(spec.join_column)
        right_sql = self._repair_sql(right, "rb", right_columns, idx)

        values = []
        for column in needed:
            source = self.engine._column_source(
                column, left, right, spec.join_column
            )
            values.append("%s.%s AS %s" % (
                source, qident(column), qident(column),
            ))
        suffix = ", " + ", ".join(values) if values else ""
        return (
            "SELECT l.idx%s FROM (%s) l JOIN (%s) r "
            "ON l.idx = r.idx AND l.%s = r.%s" % (
                suffix, left_sql, right_sql,
                qident(spec.join_column or ""),
                qident(spec.join_column or ""),
            )
        )

    def _answer_select(
        self,
        spec: QuerySpec,
        source_sql: str,
    ) -> Tuple[str, Tuple[str, ...]]:
        where_sql = self.engine._predicate_sql(spec.predicates)
        output_columns = tuple(item.alias for item in spec.select_items)
        if not spec.is_aggregation and spec.having is None:
            selections = ", ".join(
                "%s AS %s" % (
                    qident(item.column or item.alias), qident(item.alias)
                )
                for item in spec.select_items
            )
            return (
                "SELECT DISTINCT idx, %s FROM (%s) repaired WHERE %s" % (
                    selections, source_sql, where_sql,
                ),
                output_columns,
            )

        selections = []
        for item in spec.select_items:
            if not item.aggregate and item.column in spec.group_by:
                continue
            selections.append("%s AS %s" % (
                self.engine._aggregate_expression(item), qident(item.alias)
            ))
        group_columns = list(spec.group_by)
        group_sql = ", ".join(
            ["idx"] + [qident(column) for column in group_columns]
        )
        having_sql = ""
        if spec.having:
            function_column, operator, value = spec.having
            function, column = function_column.split(":", 1)
            argument = "*" if column == "*" else qident(column)
            operator = "<>" if operator == "!=" else operator
            having_sql = " HAVING %s(%s) %s %s" % (
                function.upper(), argument, operator, literal(value)
            )
        selection_suffix = ", " + ", ".join(selections) if selections else ""
        result_columns = tuple(group_columns) + tuple(
            item.alias for item in spec.select_items
            if item.aggregate or item.column not in spec.group_by
        )
        return (
            "SELECT %s%s FROM (%s) repaired WHERE %s GROUP BY %s%s" % (
                group_sql, selection_suffix, source_sql, where_sql,
                group_sql, having_sql,
            ),
            result_columns,
        )

    def compile(
        self,
        query: str,
        relations: Mapping[str, RewrittenLikeApxRelation],
    ):
        spec = parse_query(query)
        left = self._resolve_relation(spec.left_token, relations)
        right = self._resolve_relation(spec.right_token, relations) \
            if spec.right_token else None
        needed = self.engine._needed_columns(spec)

        branches = []
        output_columns: Tuple[str, ...] = tuple()
        for idx in range(1, left.h + 1):
            world_sql = self._world_sql(spec, left, right, needed, idx)
            answer_sql, branch_columns = self._answer_select(spec, world_sql)
            if output_columns and branch_columns != output_columns:
                raise AssertionError("LikeApx query copies have different outputs")
            output_columns = branch_columns
            branches.append(answer_sql)

        summary_sql, summary_columns = self.engine._summary_sql(
            spec, output_columns, left.h
        )
        sql = "WITH world_answer AS (\n%s\n) %s" % (
            "\nUNION ALL\n".join(branches), summary_sql,
        )
        return spec, sql, summary_columns

    def evaluate_summary(
        self,
        query: str,
        relations: Mapping[str, RewrittenLikeApxRelation],
    ) -> QueryResult:
        spec, sql, summary_columns = self.compile(query, relations)
        cursor = self.connection.cursor()
        started = time.perf_counter()
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.perf_counter() - started
        return QueryResult(
            query=spec,
            world_rows=[],
            summary_rows=rows,
            world_columns=tuple(),
            summary_columns=summary_columns,
            sql=sql,
            elapsed_s=elapsed,
        )


def relation_mapping(
    relation: RewrittenLikeApxRelation,
    *tokens: str,
) -> Dict[str, RewrittenLikeApxRelation]:
    result = {relation.base_table: relation}
    for token in tokens:
        result[token] = relation
        result[os.path.basename(token)] = relation
        result[token.lower()] = relation
        result[os.path.basename(token).lower()] = relation
    return result


__all__ = [
    "RewrittenLikeApx",
    "RewrittenLikeApxRelation",
    "relation_mapping",
]
