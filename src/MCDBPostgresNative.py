"""PostgreSQL-native sampling and tuple-bundle query evaluation.

Python is used only to generate SQL and collect results. Base tuples, random
draws, tuple bundles, relational evaluation, and Monte Carlo aggregation stay
inside PostgreSQL.

The implementation covers the operators used by the experiment workload:
selection, projection, duplicate removal, equijoin, group-by, HAVING, and the
AVG/SUM/COUNT/MIN/MAX aggregates. Equijoins on incomplete attributes use the
MCDB split construction: each input tuple is grouped by its distinct sampled
join values before the two inputs are joined.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extensions import adapt


CONN = dict(
    host=os.environ.get("PGHOST", "localhost"),
    port=int(os.environ.get("PGPORT", "5433")),
    dbname=os.environ.get("PGDATABASE", "mydb"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
)


def qident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def literal(value: Any) -> str:
    return adapt(value).getquoted().decode("utf-8")


def _clean_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "value"


def _temp_name(*parts: str) -> str:
    raw = "_".join(_clean_name(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return (raw[:50] + "_" + digest)[:63]


def _split_csv(expression: str) -> List[str]:
    result: List[str] = []
    depth = 0
    quote: Optional[str] = None
    start = 0
    for index, char in enumerate(expression):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(expression[start:index].strip())
            start = index + 1
    tail = expression[start:].strip()
    if tail:
        result.append(tail)
    return result


def _parse_value(token: str) -> Any:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    if re.fullmatch(r"[-+]?\d+", token):
        return int(token)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", token):
        return float(token)
    return token


def _column_name(value: str) -> str:
    return value.strip().strip('"').split(".")[-1].strip('"').lower()


@dataclasses.dataclass(frozen=True)
class Predicate:
    column: str
    operator: str
    value: Any


@dataclasses.dataclass(frozen=True)
class SelectItem:
    expression: str
    alias: str
    aggregate: Optional[str] = None
    column: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class QuerySpec:
    raw: str
    left_token: str
    right_token: Optional[str]
    join_column: Optional[str]
    select_items: Tuple[SelectItem, ...]
    predicates: Tuple[Predicate, ...]
    group_by: Tuple[str, ...]
    having: Optional[Tuple[str, str, Any]]

    @property
    def is_aggregation(self) -> bool:
        return any(item.aggregate for item in self.select_items)


def parse_query(query: str) -> QuerySpec:
    compact = " ".join(query.strip().rstrip(";").split())
    select_match = re.match(r"SELECT\s+(.+?)\s+FROM\s+", compact, re.I)
    if not select_match:
        raise ValueError("Unsupported query: missing SELECT/FROM")

    from_match = re.search(
        r"\sFROM\s+(\S+?)(?=\s+JOIN\s+|\s+WHERE\s+|\s+GROUP\s+BY\s+|\s+HAVING\s+|$)",
        compact,
        re.I,
    )
    if not from_match:
        raise ValueError("Unsupported query: missing input relation")
    left_token = from_match.group(1)

    join_match = re.search(
        r"\sJOIN\s+(\S+)\s+USING\s*\(\s*([\w\"]+)\s*\)", compact, re.I
    )
    right_token = join_match.group(1) if join_match else None
    join_column = _column_name(join_match.group(2)) if join_match else None

    items: List[SelectItem] = []
    for position, item_text in enumerate(_split_csv(select_match.group(1))):
        alias_match = re.match(r"(.+?)\s+AS\s+([\w\"]+)$", item_text, re.I)
        expression = alias_match.group(1).strip() if alias_match else item_text.strip()
        explicit_alias = alias_match.group(2).strip('"') if alias_match else None
        aggregate_match = re.match(r"(AVG|SUM|COUNT|MIN|MAX)\s*\((.*?)\)$", expression, re.I)
        if aggregate_match:
            function = aggregate_match.group(1).lower()
            argument = aggregate_match.group(2).strip()
            column = None if argument == "*" else _column_name(argument)
            alias = explicit_alias or "%s_%s" % (function, column or "all")
            items.append(SelectItem(expression, alias.lower(), function, column))
        else:
            column = _column_name(expression)
            alias = explicit_alias or column
            items.append(SelectItem(expression, alias.lower(), None, column))

    where_match = re.search(
        r"\sWHERE\s+(.+?)(?=\s+GROUP\s+BY\s+|\s+HAVING\s+|$)", compact, re.I
    )
    predicates: List[Predicate] = []
    if where_match:
        for clause in re.split(r"\s+AND\s+", where_match.group(1), flags=re.I):
            match = re.match(
                r"([\w\".]+)\s*(!=|<>|>=|<=|=|>|<)\s*(.+?)$", clause.strip()
            )
            if not match:
                raise ValueError("Unsupported WHERE clause: %s" % clause)
            predicates.append(
                Predicate(_column_name(match.group(1)), match.group(2), _parse_value(match.group(3)))
            )

    group_match = re.search(
        r"\sGROUP\s+BY\s+(.+?)(?=\s+HAVING\s+|$)", compact, re.I
    )
    group_by = tuple(
        _column_name(value) for value in _split_csv(group_match.group(1))
    ) if group_match else tuple()

    having_match = re.search(
        r"\sHAVING\s+(AVG|SUM|COUNT|MIN|MAX)\s*\((.*?)\)\s*"
        r"(!=|<>|>=|<=|=|>|<)\s*(.+?)$",
        compact,
        re.I,
    )
    having = None
    if having_match:
        function = having_match.group(1).lower()
        argument = having_match.group(2).strip()
        column = "*" if argument == "*" else _column_name(argument)
        having = ("%s:%s" % (function, column), having_match.group(3),
                  _parse_value(having_match.group(4)))
    elif " HAVING " in compact.upper():
        raise ValueError("Unsupported HAVING clause")

    return QuerySpec(
        raw=compact,
        left_token=left_token,
        right_token=right_token,
        join_column=join_column,
        select_items=tuple(items),
        predicates=tuple(predicates),
        group_by=group_by,
        having=having,
    )


@dataclasses.dataclass
class LazyDistribution:
    table: str
    donor_table: str
    group_table: str
    separators: Tuple[str, ...]
    separator_bins: Dict[str, str]
    seed: int


@dataclasses.dataclass
class BundleRelation:
    base_table: str
    bundle_table: str
    h: int
    columns: Tuple[str, ...]
    column_types: Dict[str, str]
    missing_attributes: Tuple[str, ...]
    sample_tables: Dict[str, str]
    unresolved_draws: Dict[str, int]
    sampling_s: float
    encoding_s: float
    lazy_distributions: Dict[str, LazyDistribution] = dataclasses.field(
        default_factory=dict
    )
    inline_distributions: Tuple[str, ...] = tuple()

    def has_column(self, column: str) -> bool:
        return column.lower() in self.column_types

    def is_random(self, column: str) -> bool:
        return column.lower() in self.missing_attributes

    def sample_column(self, column: str) -> str:
        return "__samples_%s" % column.lower()

    def donor_column(self, column: str) -> str:
        return "__donors_%s" % column.lower()


@dataclasses.dataclass
class QueryResult:
    query: QuerySpec
    world_rows: List[Tuple[Any, ...]]
    summary_rows: List[Tuple[Any, ...]]
    world_columns: Tuple[str, ...]
    summary_columns: Tuple[str, ...]
    sql: str
    elapsed_s: float


class NativeMCDB:
    def __init__(self, connection):
        self.connection = connection
        self.bundles: Dict[str, BundleRelation] = {}
        self._install_repair_functions()

    def _install_repair_functions(self):
        """Install session-local helpers for compact repair sets."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE OR REPLACE FUNCTION pg_temp.mcdb_repair_count(
                repairs int4multirange
            ) RETURNS bigint
            LANGUAGE sql
            IMMUTABLE
            PARALLEL SAFE
            STRICT
            AS $$
                SELECT COALESCE(
                    SUM(upper(repair_range) - lower(repair_range)), 0
                )::bigint
                FROM unnest(repairs) AS repair_ranges(repair_range)
            $$
            """
        )
        self.connection.commit()

    @classmethod
    def connect(cls, **overrides):
        settings = dict(CONN)
        settings.update(overrides)
        return cls(psycopg2.connect(**settings))

    def close(self):
        self.connection.close()

    def prepare_relation(self, source_table: str,
                         working_table: Optional[str] = None) -> str:
        """Create the PostgreSQL relation used by the MCDB operators.

        Experiment relations already stored in PostgreSQL do not have the
        internal row identifier needed to attach non-repeating marked nulls.
        This method adds that identifier inside PostgreSQL. No tuple values
        are transferred to Python.
        """
        source_columns, _ = self._columns(source_table)
        if "_rid" in source_columns and working_table is None:
            return source_table

        working_table = working_table or _temp_name("mcdb_input", source_table)
        cursor = self.connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(working_table))
        if "_rid" in source_columns:
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "SELECT * FROM %s" % (
                    qident(working_table), qident(source_table),
                )
            )
        else:
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "SELECT row_number() OVER (ORDER BY ctid)::bigint AS %s, source.* "
                "FROM %s source" % (
                    qident(working_table), qident("_rid"), qident(source_table),
                )
            )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (%s)" % (
                qident(working_table), qident("_rid"),
            )
        )
        cursor.execute("ANALYZE %s" % qident(working_table))
        self.connection.commit()
        return working_table

    def missing_attributes(self, table: str) -> Tuple[str, ...]:
        """Return columns that contain SQL NULL values in a relation."""
        columns, _ = self._columns(table)
        candidates = [
            column for column in columns
            if column != "_rid" and not column.endswith("_nullsym")
        ]
        return self._active_missing(table, candidates, columns)

    def _columns(self, table: str) -> Tuple[Tuple[str, ...], Dict[str, str]]:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod)
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s
              AND pg_catalog.pg_table_is_visible(c.oid)
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (table,),
        )
        rows = cursor.fetchall()
        if not rows:
            raise ValueError("Relation %s does not exist" % table)
        columns = tuple(row[0].lower() for row in rows)
        types = {row[0].lower(): row[1] for row in rows}
        return columns, types

    def _active_missing(self, table: str, candidates: Iterable[str],
                        columns: Sequence[str]) -> Tuple[str, ...]:
        available = set(columns)
        attrs = [attr.lower() for attr in candidates if attr.lower() in available]
        if not attrs:
            return tuple()
        cursor = self.connection.cursor()
        expressions = [
            "count(*) FILTER (WHERE %s IS NULL)" % qident(attr) for attr in attrs
        ]
        cursor.execute("SELECT %s FROM %s" % (", ".join(expressions), qident(table)))
        counts = cursor.fetchone()
        return tuple(attr for attr, count in zip(attrs, counts) if count > 0)

    @staticmethod
    def _sampling_order(active: Sequence[str],
                        ordering: Mapping[str, Sequence[str]]) -> List[str]:
        active_set = set(active)
        state: Dict[str, int] = {}
        result: List[str] = []

        def visit(attribute: str):
            marker = state.get(attribute, 0)
            if marker == 1:
                raise ValueError("Cycle in sampling dependencies at %s" % attribute)
            if marker == 2:
                return
            state[attribute] = 1
            for dependency in ordering.get(attribute, ()):
                dependency = dependency.lower()
                if dependency in active_set and dependency != attribute:
                    visit(dependency)
            state[attribute] = 2
            result.append(attribute)

        for attr in active:
            visit(attr)
        return result

    @staticmethod
    def _relation_seed(seed: float, table: str) -> float:
        digest = hashlib.sha1(table.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], "big") / float(2 ** 64)
        return ((float(seed) + offset + 1.0) % 2.0) - 1.0

    @staticmethod
    def _symbol_expression(alias: str, attribute: str,
                           columns: Sequence[str]) -> str:
        symbol_column = "%s_nullsym" % attribute
        row_key = literal("__row_") + " || %s.%s::text" % (alias, qident("_rid"))
        if symbol_column in columns:
            return "COALESCE(NULLIF(%s.%s::text, ''), %s)" % (
                alias, qident(symbol_column), row_key
            )
        return row_key

    def _create_sample_table(
        self,
        table: str,
        factor_table: str,
        columns: Sequence[str],
        column_types: Mapping[str, str],
        attribute: str,
        separators: Sequence[str],
        prior_samples: Mapping[str, str],
        h: int,
        prefix: str,
        strict: bool,
        separator_bins: Mapping[str, str],
    ) -> Tuple[str, int]:
        cursor = self.connection.cursor()
        donor_table = _temp_name(prefix, "donors", attribute)
        sample_table = _temp_name(prefix, "samples", attribute)
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(donor_table))
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(sample_table))

        symbol_expr = self._symbol_expression("b", attribute, columns)
        if not separators:
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "SELECT row_number() OVER (ORDER BY d_base.ctid)::integer AS donor_index, "
                "d_base.%s AS donor_value FROM %s d_base "
                "WHERE d_base.%s IS NOT NULL" % (
                    qident(donor_table), qident(attribute),
                    qident(factor_table), qident(attribute),
                )
            )
            cursor.execute("SELECT count(*) FROM %s" % qident(donor_table))
            if cursor.fetchone()[0] == 0:
                raise ValueError(
                    "No observed donors for %s.%s" %
                    (factor_table, attribute)
                )
            cursor.execute(
                "CREATE UNIQUE INDEX ON %s (donor_index)" % qident(donor_table)
            )
            symbol_select = "SELECT %s AS symbol FROM %s b WHERE b.%s IS NULL" % (
                symbol_expr, qident(table), qident(attribute),
            )
            if "%s_nullsym" % attribute in columns:
                symbol_select = "SELECT DISTINCT symbol FROM (%s) symbol_rows" % symbol_select
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "WITH donor_size AS ("
                "SELECT count(*)::integer AS donor_count FROM %s"
                "), symbols AS ("
                "%s"
                "), draws AS MATERIALIZED ("
                "SELECT o.symbol, g.idx, "
                "1 + floor(random() * donor_count)::integer AS donor_index "
                "FROM symbols o CROSS JOIN generate_series(1, %d) AS g(idx) "
                "CROSS JOIN donor_size"
                ") "
                "SELECT draws.symbol, "
                "array_agg(d.donor_value ORDER BY draws.idx)::%s[] AS samples, "
                "false AS used_fallback "
                "FROM draws JOIN %s d USING (donor_index) "
                "GROUP BY draws.symbol" % (
                    qident(sample_table), qident(donor_table), symbol_select,
                    int(h), column_types[attribute], qident(donor_table),
                )
            )
            cursor.execute(
                "CREATE UNIQUE INDEX ON %s (symbol)" % qident(sample_table)
            )
            return sample_table, 0

        donor_separator_values: List[str] = []
        donor_group_values: List[str] = []
        donor_joins: List[str] = []
        for separator in separators:
            if separator in separator_bins:
                alias = "donor_bin_%s" % _clean_name(separator)
                donor_joins.append(
                    "JOIN %s %s ON %s.value = d_base.%s" % (
                        qident(separator_bins[separator]), alias, alias, qident(separator)
                    )
                )
                donor_separator_values.append(
                    "%s.bin AS %s" % (alias, qident(separator))
                )
                donor_group_values.append("%s.bin" % alias)
            else:
                donor_separator_values.append(
                    "d_base.%s AS %s" % (qident(separator), qident(separator))
                )
                donor_group_values.append("d_base.%s" % qident(separator))
        separator_select = ", ".join(donor_separator_values) + ", "
        partition_clause = "PARTITION BY %s " % ", ".join(donor_group_values)
        observed_conditions = ["d_base.%s IS NOT NULL" % qident(attribute)]
        observed_conditions.extend(
            "d_base.%s IS NOT NULL" % qident(value) for value in separators
        )
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT %sd_base.%s AS donor_value, "
            "row_number() OVER (%sORDER BY d_base.ctid)::integer AS donor_index, "
            "count(*) OVER (%s)::integer AS donor_count "
            "FROM %s d_base %s WHERE %s" % (
                qident(donor_table), separator_select, qident(attribute),
                partition_clause, partition_clause,
                qident(factor_table), " ".join(donor_joins),
                " AND ".join(observed_conditions),
            )
        )
        cursor.execute("SELECT count(*) FROM %s" % qident(donor_table))
        if cursor.fetchone()[0] == 0:
            raise ValueError(
                "No observed donors for %s.%s" %
                (factor_table, attribute)
            )
        donor_index_columns = ", ".join(
            [qident(separator) for separator in separators] + ["donor_index"]
        )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (%s)" % (
                qident(donor_table), donor_index_columns,
            )
        )

        global_donor_table = None
        if not strict:
            global_donor_table = _temp_name(prefix, "global_donors", attribute)
            cursor.execute("DROP TABLE IF EXISTS %s" % qident(global_donor_table))
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "SELECT row_number() OVER (ORDER BY d_base.ctid)::integer AS donor_index, "
                "d_base.%s AS donor_value FROM %s d_base "
                "WHERE d_base.%s IS NOT NULL" % (
                    qident(global_donor_table), qident(attribute),
                    qident(factor_table), qident(attribute),
                )
            )
            cursor.execute(
                "CREATE UNIQUE INDEX ON %s (donor_index)" % qident(global_donor_table)
            )

        prior_joins: List[str] = []
        effective_separators: List[str] = []
        for separator in separators:
            if separator in prior_samples:
                alias = "sample_%s" % _clean_name(separator)
                separator_symbol = self._symbol_expression("b", separator, columns)
                prior_joins.append(
                    "LEFT JOIN %s %s ON %s.symbol = %s" % (
                        qident(prior_samples[separator]), alias, alias, separator_symbol
                    )
                )
                effective = "COALESCE(b.%s, %s.samples[o.idx])" % (
                    qident(separator), alias
                )
            else:
                effective = "b.%s" % qident(separator)
            if separator in separator_bins:
                bin_alias = "request_bin_%s" % _clean_name(separator)
                prior_joins.append(
                    "LEFT JOIN %s %s ON %s.value = %s" % (
                        qident(separator_bins[separator]), bin_alias, bin_alias, effective
                    )
                )
                effective = "%s.bin" % bin_alias
            effective_separators.append(
                "%s AS %s" % (effective, qident("effective_%s" % separator))
            )

        request_separator_select = (
            ", " + ", ".join(effective_separators) if effective_separators else ""
        )
        group_conditions = []
        local_donor_conditions = []
        for separator in separators:
            effective_name = qident("effective_%s" % separator)
            group_conditions.append(
                "dg.%s = r.%s" % (
                    qident(separator), qident("effective_%s" % separator)
                )
            )
            local_donor_conditions.append(
                "d.%s = x.%s" % (qident(separator), effective_name)
            )
        group_join = " AND ".join(group_conditions)
        local_donor_join = " AND ".join(local_donor_conditions)
        donor_group_columns = ", ".join(qident(value) for value in separators)

        if strict:
            create_sql = """
                CREATE TEMP TABLE {sample_table} ON COMMIT PRESERVE ROWS AS
                WITH occurrence_groups AS (
                    SELECT {symbol_expr} AS symbol,
                           array_agg(b.{rid} ORDER BY b.{rid}) AS occurrence_rids
                    FROM {base} b
                    WHERE b.{attribute} IS NULL
                    GROUP BY {symbol_expr}
                ),
                selected_occurrences AS (
                    SELECT o.symbol,
                           h.idx,
                           o.occurrence_rids[
                               1 + floor(random() * cardinality(o.occurrence_rids))::integer
                           ] AS rid
                    FROM occurrence_groups o
                    CROSS JOIN generate_series(1, {h}) AS h(idx)
                ),
                requests AS (
                    SELECT o.symbol, o.idx{request_separator_select}
                    FROM selected_occurrences o
                    JOIN {base} b ON b.{rid} = o.rid
                    {prior_joins}
                ),
                donor_groups AS (
                    SELECT {donor_group_columns}, max(donor_count)::integer AS donor_count
                    FROM {donor_table}
                    GROUP BY {donor_group_columns}
                ),
                draw_indices AS MATERIALIZED (
                    SELECT r.*,
                           1 + floor(random() * dg.donor_count)::integer AS donor_index
                    FROM requests r
                    JOIN donor_groups dg ON {group_join}
                ),
                draws AS (
                    SELECT x.symbol,
                           x.idx,
                           d.donor_value AS sampled_value
                    FROM draw_indices x
                    JOIN {donor_table} d
                      ON {local_donor_join}
                     AND d.donor_index = x.donor_index
                )
                SELECT symbol,
                       array_agg(sampled_value ORDER BY idx)::{attribute_type}[] AS samples,
                       false AS used_fallback
                FROM draws
                GROUP BY symbol
            """.format(
                sample_table=qident(sample_table),
                symbol_expr=symbol_expr,
                rid=qident("_rid"),
                base=qident(table),
                attribute=qident(attribute),
                h=int(h),
                request_separator_select=request_separator_select,
                prior_joins=" ".join(prior_joins),
                donor_table=qident(donor_table),
                donor_group_columns=donor_group_columns,
                group_join=group_join,
                local_donor_join=local_donor_join,
                attribute_type=column_types[attribute],
            )
        else:
            create_sql = """
            CREATE TEMP TABLE {sample_table} ON COMMIT PRESERVE ROWS AS
            WITH occurrence_groups AS (
                SELECT {symbol_expr} AS symbol,
                       array_agg(b.{rid} ORDER BY b.{rid}) AS occurrence_rids
                FROM {base} b
                WHERE b.{attribute} IS NULL
                GROUP BY {symbol_expr}
            ),
            selected_occurrences AS (
                SELECT o.symbol,
                       h.idx,
                       o.occurrence_rids[
                           1 + floor(random() * cardinality(o.occurrence_rids))::integer
                       ] AS rid
                FROM occurrence_groups o
                CROSS JOIN generate_series(1, {h}) AS h(idx)
            ),
            requests AS (
                SELECT o.symbol, o.idx{request_separator_select}
                FROM selected_occurrences o
                JOIN {base} b ON b.{rid} = o.rid
                {prior_joins}
            ),
            donor_groups AS (
                SELECT {donor_group_columns}, max(donor_count)::integer AS donor_count
                FROM {donor_table}
                GROUP BY {donor_group_columns}
            ),
            global_size AS (
                SELECT count(*)::integer AS donor_count FROM {global_donor_table}
            ),
            draw_indices AS MATERIALIZED (
                SELECT r.*,
                       dg.donor_count IS NULL AS used_fallback,
                       1 + floor(
                           random() * COALESCE(dg.donor_count, gs.donor_count)
                       )::integer AS donor_index
                FROM requests r
                LEFT JOIN donor_groups dg ON {group_join}
                CROSS JOIN global_size gs
            ),
            draws AS (
                SELECT x.symbol,
                       x.idx,
                       CASE WHEN x.used_fallback
                            THEN gd.donor_value ELSE d.donor_value END AS sampled_value,
                       x.used_fallback
                FROM draw_indices x
                LEFT JOIN {donor_table} d
                  ON NOT x.used_fallback
                 AND {local_donor_join}
                 AND d.donor_index = x.donor_index
                LEFT JOIN {global_donor_table} gd
                  ON x.used_fallback
                 AND gd.donor_index = x.donor_index
            )
            SELECT symbol,
                   array_agg(sampled_value ORDER BY idx)::{attribute_type}[] AS samples,
                   bool_or(used_fallback) AS used_fallback
            FROM draws
            GROUP BY symbol
        """.format(
            sample_table=qident(sample_table),
            symbol_expr=symbol_expr,
            rid=qident("_rid"),
            base=qident(table),
            attribute=qident(attribute),
            h=int(h),
            request_separator_select=request_separator_select,
            prior_joins=" ".join(prior_joins),
            donor_table=qident(donor_table),
            donor_group_columns=donor_group_columns,
            global_donor_table=qident(global_donor_table),
            group_join=group_join,
            local_donor_join=local_donor_join,
            attribute_type=column_types[attribute],
        )
        cursor.execute(create_sql)
        cursor.execute("CREATE UNIQUE INDEX ON %s (symbol)" % qident(sample_table))

        cursor.execute(
            "SELECT count(DISTINCT %s) FROM %s b WHERE b.%s IS NULL" % (
                symbol_expr, qident(table), qident(attribute)
            )
        )
        expected = cursor.fetchone()[0]
        cursor.execute(
            "SELECT count(*) FILTER ("
            "WHERE used_fallback OR cardinality(samples) <> %d"
            "), count(*) FROM %s" % (int(h), qident(sample_table))
        )
        unresolved, produced = cursor.fetchone()
        unresolved += expected - produced
        if strict and unresolved:
            raise ValueError(
                "%d marked-null symbols in %s.%s have no matching observed donor"
                % (unresolved, table, attribute)
            )
        return sample_table, unresolved

    @staticmethod
    def _is_numeric_type(type_name: str) -> bool:
        lowered = type_name.lower()
        return any(token in lowered for token in (
            "smallint", "integer", "bigint", "numeric", "decimal",
            "real", "double precision",
        ))

    def _create_separator_bins(
        self,
        table: str,
        separators: Iterable[str],
        column_types: Mapping[str, str],
        prefix: str,
        n_bins: Optional[int],
    ) -> Dict[str, str]:
        if n_bins is None:
            return {}
        if n_bins <= 0:
            raise ValueError("n_bins must be positive when separator binning is enabled")
        cursor = self.connection.cursor()
        result: Dict[str, str] = {}
        for separator in sorted(set(separators)):
            if separator not in column_types or not self._is_numeric_type(column_types[separator]):
                continue
            cursor.execute(
                "SELECT count(DISTINCT %s) FROM %s WHERE %s IS NOT NULL" % (
                    qident(separator), qident(table), qident(separator)
                )
            )
            distinct_values = cursor.fetchone()[0]
            if distinct_values <= n_bins:
                continue
            bin_table = _temp_name(prefix, "bins", separator)
            cursor.execute("DROP TABLE IF EXISTS %s" % qident(bin_table))
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "SELECT DISTINCT value, bin FROM ("
                "SELECT %s AS value, "
                "LEAST(%d, GREATEST(1, CEIL(cume_dist() OVER "
                "(ORDER BY %s) * %d)::integer)) AS bin "
                "FROM %s WHERE %s IS NOT NULL"
                ") ranked" % (
                    qident(bin_table), qident(separator), n_bins,
                    qident(separator), n_bins, qident(table), qident(separator),
                )
            )
            cursor.execute("CREATE UNIQUE INDEX ON %s (value)" % qident(bin_table))
            result[separator] = bin_table
        return result

    def _has_repeated_symbols(
        self,
        table: str,
        attributes: Iterable[str],
        columns: Sequence[str],
    ) -> bool:
        cursor = self.connection.cursor()
        for attribute in attributes:
            symbol = self._symbol_expression("b", attribute, columns)
            cursor.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM %s b WHERE b.%s IS NULL "
                "GROUP BY %s HAVING count(*) > 1 LIMIT 1"
                ")" % (
                    qident(table), qident(attribute), symbol,
                )
            )
            if cursor.fetchone()[0]:
                return True
        return False

    def _create_lazy_distribution(
        self,
        factor_table: str,
        attribute: str,
        separators: Sequence[str],
        separator_bins: Mapping[str, str],
        prefix: str,
        seed: float,
    ) -> LazyDistribution:
        cursor = self.connection.cursor()
        distribution_table = _temp_name(prefix, "distribution", attribute)
        donor_table = _temp_name(prefix, "lazy_donors", attribute)
        group_table = _temp_name(prefix, "lazy_groups", attribute)
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(distribution_table))
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(donor_table))
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(group_table))

        selected_separators: List[str] = []
        grouped_separators: List[str] = []
        joins: List[str] = []
        for separator in separators:
            if separator in separator_bins:
                alias = "distribution_bin_%s" % _clean_name(separator)
                joins.append(
                    "JOIN %s %s ON %s.value = d.%s" % (
                        qident(separator_bins[separator]), alias, alias,
                        qident(separator),
                    )
                )
                selected_separators.append(
                    "%s.bin AS %s" % (alias, qident(separator))
                )
                grouped_separators.append("%s.bin" % alias)
            else:
                selected_separators.append(
                    "d.%s AS %s" % (
                        qident(separator), qident(separator),
                    )
                )
                grouped_separators.append("d.%s" % qident(separator))

        conditions = ["d.%s IS NOT NULL" % qident(attribute)]
        conditions.extend(
            "d.%s IS NOT NULL" % qident(separator)
            for separator in separators
        )
        selected_prefix = (
            ", ".join(selected_separators) + ", "
            if selected_separators else ""
        )
        group_clause = (
            " GROUP BY " + ", ".join(grouped_separators)
            if grouped_separators else ""
        )
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT %sarray_agg(d.%s ORDER BY d.ctid) AS donor_values "
            "FROM %s d %s WHERE %s%s" % (
                qident(distribution_table), selected_prefix,
                qident(attribute), qident(factor_table), " ".join(joins),
                " AND ".join(conditions), group_clause,
            )
        )
        cursor.execute(
            "SELECT COALESCE(sum(cardinality(donor_values)), 0) FROM %s" %
            qident(distribution_table)
        )
        if cursor.fetchone()[0] == 0:
            raise ValueError(
                "No observed donors for %s.%s" %
                (factor_table, attribute)
            )
        if separators:
            columns_sql = ", ".join(qident(value) for value in separators)
            cursor.execute(
                "CREATE UNIQUE INDEX ON %s (%s)" % (
                    qident(distribution_table), columns_sql,
                )
            )
        partition = (
            "PARTITION BY %s " % ", ".join(grouped_separators)
            if grouped_separators else ""
        )
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT %sd.%s AS donor_value, "
            "row_number() OVER (%sORDER BY d.ctid)::integer AS donor_index "
            "FROM %s d %s WHERE %s" % (
                qident(donor_table), selected_prefix, qident(attribute),
                partition, qident(factor_table), " ".join(joins),
                " AND ".join(conditions),
            )
        )
        separator_columns = ", ".join(
            qident(value) for value in separators
        )
        group_prefix = (
            separator_columns + ", " if separator_columns else ""
        )
        group_clause = (
            " GROUP BY " + separator_columns if separator_columns else ""
        )
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT %smax(donor_index)::integer AS donor_count "
            "FROM %s%s" % (
                qident(group_table), group_prefix, qident(donor_table),
                group_clause,
            )
        )
        donor_index_columns = ", ".join(
            [qident(value) for value in separators] + ["donor_index"]
        )
        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (%s)" % (
                qident(donor_table), donor_index_columns,
            )
        )
        if separator_columns:
            cursor.execute(
                "CREATE UNIQUE INDEX ON %s (%s)" % (
                    qident(group_table), separator_columns,
                )
            )
        cursor.execute("ANALYZE %s" % qident(donor_table))
        cursor.execute("ANALYZE %s" % qident(group_table))
        cursor.execute("ANALYZE %s" % qident(distribution_table))
        digest = hashlib.sha1(
            ("%s|%s|%s" % (seed, factor_table, attribute)).encode("utf-8")
        ).digest()
        draw_seed = int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
        return LazyDistribution(
            table=distribution_table,
            donor_table=donor_table,
            group_table=group_table,
            separators=tuple(separators),
            separator_bins=dict(separator_bins),
            seed=draw_seed,
        )

    @staticmethod
    def _precomputed_draw_index(
        symbol: str,
        index_expression: str,
        salt: str,
        seed: int,
        values: str,
    ) -> str:
        return (
            "1 + ((hashtextextended((%s) || '|' || (%s)::text || '|%s', %d) "
            "& 9223372036854775807::bigint) "
            "%% cardinality(%s))::integer"
        ) % (symbol, index_expression, salt, seed, values)

    def _create_precomputed_sample_table(
        self,
        table: str,
        factor_table: str,
        columns: Sequence[str],
        column_types: Mapping[str, str],
        attribute: str,
        separators: Sequence[str],
        prior_samples: Mapping[str, str],
        h: int,
        prefix: str,
        strict: bool,
        separator_bins: Mapping[str, str],
        seed: float,
    ) -> Tuple[str, int]:
        if not strict:
            return self._create_sample_table(
                table=table,
                factor_table=factor_table,
                columns=columns,
                column_types=column_types,
                attribute=attribute,
                separators=separators,
                prior_samples=prior_samples,
                h=h,
                prefix=prefix,
                strict=strict,
                separator_bins=separator_bins,
            )
        distribution = self._create_lazy_distribution(
            factor_table=factor_table,
            attribute=attribute,
            separators=separators,
            separator_bins=separator_bins,
            prefix=prefix,
            seed=seed,
        )
        cursor = self.connection.cursor()
        sample_table = _temp_name(prefix, "samples", attribute)
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(sample_table))
        symbol = self._symbol_expression("b", attribute, columns)

        if not separators:
            symbol_select = (
                "SELECT %s AS symbol FROM %s b WHERE b.%s IS NULL" % (
                    symbol, qident(table), qident(attribute),
                )
            )
            if "%s_nullsym" % attribute in columns:
                symbol_select = (
                    "SELECT DISTINCT symbol FROM (%s) symbol_rows" %
                    symbol_select
                )
            donor_index = self._precomputed_draw_index(
                "s.symbol", "g.idx", attribute, distribution.seed,
                "d.donor_values",
            )
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "WITH symbols AS (%s), draws AS ("
                "SELECT s.symbol, g.idx, d.donor_values[%s] AS sampled_value "
                "FROM symbols s CROSS JOIN generate_series(1, %d) AS g(idx) "
                "CROSS JOIN %s d"
                ") "
                "SELECT symbol, "
                "array_agg(sampled_value ORDER BY idx)::%s[] AS samples, "
                "false AS used_fallback "
                "FROM draws GROUP BY symbol" % (
                    qident(sample_table), symbol_select, donor_index, h,
                    qident(distribution.table), column_types[attribute],
                )
            )
        else:
            prior_joins: List[str] = []
            effective: Dict[str, str] = {}
            for separator in separators:
                value = "b.%s" % qident(separator)
                if separator in prior_samples:
                    alias = "sample_%s" % _clean_name(separator)
                    separator_symbol = self._symbol_expression(
                        "b", separator, columns
                    )
                    prior_joins.append(
                        "LEFT JOIN %s %s ON %s.symbol = %s" % (
                            qident(prior_samples[separator]), alias, alias,
                            separator_symbol,
                        )
                    )
                    value = "COALESCE(%s, %s.samples[g.idx])" % (
                        value, alias,
                    )
                bin_table = separator_bins.get(separator)
                if bin_table:
                    bin_alias = "sample_bin_%s" % _clean_name(separator)
                    prior_joins.append(
                        "LEFT JOIN %s %s ON %s.value = %s" % (
                            qident(bin_table), bin_alias, bin_alias, value,
                        )
                    )
                    value = "%s.bin" % bin_alias
                effective[separator] = value
            distribution_join = " AND ".join(
                "d.%s IS NOT DISTINCT FROM %s" % (
                    qident(separator), effective[separator],
                )
                for separator in separators
            )
            occurrence_index = self._precomputed_draw_index(
                "og.symbol", "g.idx", "occurrence_%s" % attribute,
                distribution.seed, "og.occurrence_rids",
            )
            donor_index = self._precomputed_draw_index(
                "og.symbol", "g.idx", attribute, distribution.seed,
                "d.donor_values",
            )
            cursor.execute(
                "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
                "WITH occurrence_groups AS MATERIALIZED ("
                "SELECT %s AS symbol, "
                "array_agg(b.%s ORDER BY b.%s) AS occurrence_rids "
                "FROM %s b WHERE b.%s IS NULL GROUP BY %s"
                "), draws AS ("
                "SELECT og.symbol, g.idx, "
                "d.donor_values[%s] AS sampled_value "
                "FROM occurrence_groups og "
                "CROSS JOIN generate_series(1, %d) AS g(idx) "
                "JOIN %s b ON b.%s = og.occurrence_rids[%s] "
                "%s JOIN %s d ON %s"
                ") "
                "SELECT symbol, "
                "array_agg(sampled_value ORDER BY idx)::%s[] AS samples, "
                "false AS used_fallback "
                "FROM draws GROUP BY symbol" % (
                    qident(sample_table), symbol, qident("_rid"),
                    qident("_rid"), qident(table), qident(attribute), symbol,
                    donor_index, h, qident(table), qident("_rid"),
                    occurrence_index, " ".join(prior_joins),
                    qident(distribution.table), distribution_join,
                    column_types[attribute],
                )
            )

        cursor.execute(
            "CREATE UNIQUE INDEX ON %s (symbol)" % qident(sample_table)
        )
        cursor.execute(
            "SELECT count(DISTINCT %s) FROM %s b WHERE b.%s IS NULL" % (
                symbol, qident(table), qident(attribute),
            )
        )
        expected = cursor.fetchone()[0]
        cursor.execute(
            "SELECT count(*) FILTER (WHERE cardinality(samples) <> %d), "
            "count(*) FROM %s" % (h, qident(sample_table))
        )
        unresolved, produced = cursor.fetchone()
        unresolved += expected - produced
        if unresolved:
            raise ValueError(
                "%d marked-null symbols in %s.%s have no matching "
                "precomputed distribution" % (
                    unresolved, table, attribute,
                )
            )
        return sample_table, unresolved

    def create_bundle(
        self,
        table: str,
        missing_attributes: Iterable[str],
        ordering: Optional[Mapping[str, Sequence[str]]],
        h: int,
        seed: float = 0.42,
        prefix: Optional[str] = None,
        strict: bool = True,
        n_bins: Optional[int] = None,
        factor_table: Optional[str] = None,
        lazy_samples: bool = False,
        precomputed_samples: bool = False,
    ) -> BundleRelation:
        """Create an MCDB bundle from factor samples.

        The default is the exact singleton-factor specialization of
        FactorSampler: conditioning values are not discretized and an empty
        conditional relation raises an error. Passing ``strict=False`` permits
        unconditional fallback, and passing ``n_bins`` enables the older
        approximate discretization mode. When ``factor_table`` is supplied,
        its observed rows estimate the factors while ``table`` identifies the
        marked-null symbols to sample and the rows to encode.
        """
        if h <= 0:
            raise ValueError("h must be positive")
        if lazy_samples and precomputed_samples:
            raise ValueError(
                "lazy_samples and precomputed_samples are mutually exclusive"
            )
        columns, column_types = self._columns(table)
        if "_rid" not in columns:
            raise ValueError("Relation %s requires a unique _rid column" % table)
        factor_table = factor_table or table
        factor_columns, factor_column_types = self._columns(factor_table)
        active = self._active_missing(table, missing_attributes, columns)
        normalized_ordering = {
            key.lower(): tuple(value.lower() for value in values if value.lower() in columns)
            for key, values in (ordering or {}).items()
        }
        sample_order = self._sampling_order(active, normalized_ordering)
        prefix = prefix or _temp_name("mcdb", table, str(h))
        cursor = self.connection.cursor()
        cursor.execute("SELECT setseed(%s)", (self._relation_seed(seed, table),))
        sampling_started = time.perf_counter()

        all_separators = [
            separator
            for attribute in sample_order
            for separator in normalized_ordering.get(attribute, tuple())
        ]
        required_factor_columns = set(active) | set(all_separators)
        absent_factor_columns = required_factor_columns - set(factor_columns)
        if absent_factor_columns:
            raise ValueError(
                "Factor relation %s is missing columns %s" %
                (factor_table, sorted(absent_factor_columns))
            )
        separator_bins = self._create_separator_bins(
            factor_table,
            all_separators,
            factor_column_types,
            prefix,
            n_bins,
        )

        sample_tables: Dict[str, str] = {}
        lazy_distributions: Dict[str, LazyDistribution] = {}
        unresolved: Dict[str, int] = {}
        use_lazy = lazy_samples and not self._has_repeated_symbols(
            table, sample_order, columns
        )
        for attribute in sample_order:
            separators = normalized_ordering.get(attribute, tuple())
            if use_lazy:
                lazy_distributions[attribute] = self._create_lazy_distribution(
                    factor_table=factor_table,
                    attribute=attribute,
                    separators=separators,
                    separator_bins=separator_bins,
                    prefix=prefix,
                    seed=seed,
                )
                unresolved[attribute] = 0
            elif precomputed_samples:
                sample_table, unresolved_count = \
                    self._create_precomputed_sample_table(
                        table=table,
                        factor_table=factor_table,
                        columns=columns,
                        column_types=column_types,
                        attribute=attribute,
                        separators=separators,
                        prior_samples=sample_tables,
                        h=h,
                        prefix=prefix,
                        strict=strict,
                        separator_bins=separator_bins,
                        seed=seed,
                    )
                sample_tables[attribute] = sample_table
                unresolved[attribute] = unresolved_count
            else:
                sample_table, unresolved_count = self._create_sample_table(
                    table=table,
                    factor_table=factor_table,
                    columns=columns,
                    column_types=column_types,
                    attribute=attribute,
                    separators=separators,
                    prior_samples=sample_tables,
                    h=h,
                    prefix=prefix,
                    strict=strict,
                    separator_bins=separator_bins,
                )
                sample_tables[attribute] = sample_table
                unresolved[attribute] = unresolved_count
        sampling_s = time.perf_counter() - sampling_started

        encoding_started = time.perf_counter()
        bundle_table = _temp_name(prefix, "bundle")
        cursor.execute("DROP TABLE IF EXISTS %s" % qident(bundle_table))
        sample_selects: List[str] = []
        sample_joins: List[str] = []
        for attribute in active:
            if attribute not in sample_tables:
                continue
            alias = "sample_%s" % _clean_name(attribute)
            symbol = self._symbol_expression("b", attribute, columns)
            sample_selects.append(
                "%s.samples AS %s" % (alias, qident("__samples_%s" % attribute))
            )
            sample_joins.append(
                "LEFT JOIN %s %s ON b.%s IS NULL AND %s.symbol = %s" % (
                    qident(sample_tables[attribute]), alias, qident(attribute), alias, symbol
                )
            )
        extra_select = ", " + ", ".join(sample_selects) if sample_selects else ""
        cursor.execute(
            "CREATE TEMP TABLE %s ON COMMIT PRESERVE ROWS AS "
            "SELECT b.*%s, int4multirange(int4range(1, %d)) AS %s "
            "FROM %s b %s" % (
                qident(bundle_table), extra_select, h + 1, qident("__present"),
                qident(table), " ".join(sample_joins),
            )
        )
        cursor.execute("CREATE UNIQUE INDEX ON %s (%s)" % (
            qident(bundle_table), qident("_rid")
        ))
        inline_distributions: List[str] = []
        self.connection.commit()
        encoding_s = time.perf_counter() - encoding_started

        bundle = BundleRelation(
            base_table=table,
            bundle_table=bundle_table,
            h=h,
            columns=columns,
            column_types=column_types,
            missing_attributes=active,
            sample_tables=sample_tables,
            unresolved_draws=unresolved,
            sampling_s=sampling_s,
            encoding_s=encoding_s,
            lazy_distributions=lazy_distributions,
            inline_distributions=tuple(inline_distributions),
        )
        self.bundles[table] = bundle
        return bundle

    def _value_expression(
        self,
        bundle: BundleRelation,
        alias: str,
        column: str,
        index_expression: str,
        recursion: Optional[frozenset] = None,
    ) -> str:
        column = column.lower()
        if not bundle.is_random(column):
            return "%s.%s" % (alias, qident(column))
        if column in bundle.sample_tables:
            return "COALESCE(%s.%s, %s.%s[%s])" % (
                alias, qident(column), alias, qident(bundle.sample_column(column)),
                index_expression,
            )
        distribution = bundle.lazy_distributions.get(column)
        if distribution is None:
            return "%s.%s" % (alias, qident(column))
        recursion = recursion or frozenset()
        if column in recursion:
            raise ValueError(
                "Cycle in lazy sampling dependencies at %s" % column
            )
        next_recursion = recursion | {column}
        conditions: List[str] = []
        for separator in distribution.separators:
            effective = self._value_expression(
                bundle, alias, separator, index_expression, next_recursion
            )
            bin_table = distribution.separator_bins.get(separator)
            if bin_table:
                effective = (
                    "(SELECT bins.bin FROM %s bins "
                    "WHERE bins.value IS NOT DISTINCT FROM (%s) LIMIT 1)"
                ) % (qident(bin_table), effective)
            conditions.append(
                "d.%s IS NOT DISTINCT FROM (%s)" % (
                    qident(separator), effective,
                )
            )
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        symbol = self._symbol_expression(alias, column, bundle.columns)
        def random_index(values: str) -> str:
            return (
                "1 + ((hashtextextended((%s) || '|' || (%s)::text || '|%s', %d) "
                "& 9223372036854775807::bigint) "
                "%% cardinality(%s))::integer"
            ) % (
                symbol, index_expression, column, distribution.seed, values,
            )
        sampled = (
            "(SELECT d.donor_values[%s] FROM %s d%s LIMIT 1)"
        ) % (
            random_index("d.donor_values"),
            qident(distribution.table), where,
        )
        alternatives: List[str] = []
        if column in bundle.inline_distributions:
            donor_values = "%s.%s" % (
                alias, qident(bundle.donor_column(column)),
            )
            alternatives.append(
                "%s[%s]" % (donor_values, random_index(donor_values))
            )
        alternatives.append(sampled)
        return "COALESCE(%s.%s, %s)" % (
            alias, qident(column), ", ".join(alternatives),
        )

    def _lazy_sampling_joins(
        self,
        bundle: BundleRelation,
        relation_alias: str,
        index_expression: str,
        requested_columns: Iterable[str],
        alias_prefix: str,
    ) -> Tuple[str, Dict[str, str]]:
        required = {
            column.lower() for column in requested_columns
            if bundle.is_random(column)
        }
        pending = list(required)
        while pending:
            attribute = pending.pop()
            distribution = bundle.lazy_distributions.get(attribute)
            if distribution is None:
                continue
            for separator in distribution.separators:
                if bundle.is_random(separator) and separator not in required:
                    required.add(separator)
                    pending.append(separator)

        joins: List[str] = []
        sampled_values: Dict[str, str] = {}
        for attribute, distribution in bundle.lazy_distributions.items():
            if attribute not in required:
                continue
            group_alias = "%s_group_%s" % (
                alias_prefix, _clean_name(attribute)
            )
            donor_alias = "%s_donor_%s" % (
                alias_prefix, _clean_name(attribute)
            )
            group_conditions: List[str] = []
            for separator in distribution.separators:
                effective = sampled_values.get(separator)
                if effective is None:
                    effective = self._value_expression(
                        bundle, relation_alias, separator, index_expression
                    )
                bin_table = distribution.separator_bins.get(separator)
                if bin_table:
                    bin_alias = "%s_bin_%s_%s" % (
                        alias_prefix, _clean_name(attribute),
                        _clean_name(separator),
                    )
                    joins.append(
                        "LEFT JOIN %s %s ON %s.value = (%s)" % (
                            qident(bin_table), bin_alias, bin_alias, effective,
                        )
                    )
                    effective = "%s.bin" % bin_alias
                group_conditions.append(
                    "%s.%s = (%s)" % (
                        group_alias, qident(separator), effective,
                    )
                )
            if group_conditions:
                joins.append(
                    "LEFT JOIN %s %s ON %s" % (
                        qident(distribution.group_table), group_alias,
                        " AND ".join(group_conditions),
                    )
                )
            else:
                joins.append(
                    "CROSS JOIN %s %s" % (
                        qident(distribution.group_table), group_alias,
                    )
                )
            symbol = self._symbol_expression(
                relation_alias, attribute, bundle.columns
            )
            donor_index = (
                "1 + ((hashtextextended((%s) || '|' || (%s)::text || '|%s', %d) "
                "& 9223372036854775807::bigint) "
                "%% %s.donor_count)::integer"
            ) % (
                symbol, index_expression, attribute, distribution.seed,
                group_alias,
            )
            donor_conditions = [
                "%s.%s = %s.%s" % (
                    donor_alias, qident(separator),
                    group_alias, qident(separator),
                )
                for separator in distribution.separators
            ]
            donor_conditions.append(
                "%s.donor_index = %s" % (donor_alias, donor_index)
            )
            joins.append(
                "LEFT JOIN %s %s ON %s" % (
                    qident(distribution.donor_table), donor_alias,
                    " AND ".join(donor_conditions),
                )
            )
            sampled_values[attribute] = "COALESCE(%s.%s, %s.donor_value)" % (
                relation_alias, qident(attribute), donor_alias,
            )
        return " ".join(joins), sampled_values

    @staticmethod
    def _resolve_bundle(token: str,
                        relations: Mapping[str, BundleRelation]) -> BundleRelation:
        options = [token, token.split("/")[-1], token.lower(), token.split("/")[-1].lower()]
        for option in options:
            if option in relations:
                return relations[option]
        raise KeyError("No bundle supplied for query relation %s" % token)

    @staticmethod
    def _column_source(column: str, left: BundleRelation,
                       right: Optional[BundleRelation], join_column: Optional[str]) -> str:
        in_left = left.has_column(column)
        in_right = bool(right and right.has_column(column))
        if in_left and in_right and column != join_column:
            raise ValueError("Ambiguous unqualified column %s" % column)
        if in_left:
            return "l"
        if in_right:
            return "r"
        raise ValueError("Unknown query column %s" % column)

    def _needed_columns(self, spec: QuerySpec) -> List[str]:
        result: List[str] = []
        for item in spec.select_items:
            if item.column and item.column not in result:
                result.append(item.column)
        for predicate in spec.predicates:
            if predicate.column not in result:
                result.append(predicate.column)
        for column in spec.group_by:
            if column not in result:
                result.append(column)
        if spec.having:
            argument = spec.having[0].split(":", 1)[1]
            if argument != "*" and argument not in result:
                result.append(argument)
        if spec.join_column and spec.join_column not in result:
            result.append(spec.join_column)
        return result

    def _single_world_rows(self, left: BundleRelation,
                           needed: Sequence[str]) -> Tuple[List[str], str]:
        lazy_joins, lazy_values = self._lazy_sampling_joins(
            left, "l", "g.idx", needed, "single_world"
        )
        select_values = []
        for column in needed:
            value = lazy_values.get(column)
            if value is None:
                value = self._value_expression(
                    left, "l", column, "g.idx"
                )
            select_values.append(
                "%s AS %s" % (
                    value, qident(column)
                )
            )
        select_suffix = ", " + ", ".join(select_values) if select_values else ""
        ctes = [
            "world_rows AS (SELECT g.idx%s FROM %s l "
            "CROSS JOIN generate_series(1, %d) AS g(idx) %s)" % (
                select_suffix, qident(left.bundle_table), left.h, lazy_joins
            )
        ]
        return ctes, "world_rows"

    def _split_cte(self, bundle: BundleRelation, alias: str,
                   join_column: str, cte_name: str) -> str:
        lazy_joins, lazy_values = self._lazy_sampling_joins(
            bundle, alias, "g.idx", [join_column],
            "%s_split" % cte_name,
        )
        value = lazy_values.get(join_column)
        if value is None:
            value = self._value_expression(
                bundle, alias, join_column, "g.idx"
            )
        return (
            "%s AS ("
            "SELECT %s.%s AS rid, v.join_value, array_agg(g.idx ORDER BY g.idx) AS repairs "
            "FROM %s %s CROSS JOIN generate_series(1, %d) AS g(idx) "
            "%s "
            "CROSS JOIN LATERAL (SELECT %s AS join_value) v "
            "WHERE v.join_value IS NOT NULL "
            "GROUP BY %s.%s, v.join_value)"
        ) % (
            cte_name, alias, qident("_rid"), qident(bundle.bundle_table), alias,
            bundle.h, lazy_joins, value, alias, qident("_rid"),
        )

    def _join_world_rows(self, left: BundleRelation, right: BundleRelation,
                         join_column: str,
                         needed: Sequence[str]) -> Tuple[List[str], str]:
        if left.h != right.h:
            raise ValueError("Joined bundles must use the same H")
        h = left.h
        left_random = left.is_random(join_column)
        right_random = right.is_random(join_column)
        ctes: List[str] = []
        left_needed = [
            column for column in needed
            if self._column_source(column, left, right, join_column) == "l"
        ]
        right_needed = [
            column for column in needed
            if self._column_source(column, left, right, join_column) == "r"
        ]
        left_lazy_joins, left_lazy_values = self._lazy_sampling_joins(
            left, "l", "g.idx", left_needed, "join_world_left"
        )
        right_lazy_joins, right_lazy_values = self._lazy_sampling_joins(
            right, "r", "g.idx", right_needed, "join_world_right"
        )
        lazy_joins = " ".join(
            value for value in (left_lazy_joins, right_lazy_joins) if value
        )

        def selected_values(index_expression: str) -> str:
            values = []
            for column in needed:
                source = self._column_source(column, left, right, join_column)
                bundle = left if source == "l" else right
                lazy_values = (
                    left_lazy_values if source == "l" else right_lazy_values
                )
                value = lazy_values.get(column)
                if value is None:
                    value = self._value_expression(
                        bundle, source, column, index_expression
                    )
                values.append(
                    "%s AS %s" % (
                        value, qident(column),
                    )
                )
            return ", " + ", ".join(values) if values else ""

        if not left_random and not right_random:
            values = selected_values("g.idx")
            ctes.append(
                "world_rows AS (SELECT g.idx%s FROM %s l JOIN %s r "
                "ON l.%s = r.%s CROSS JOIN generate_series(1, %d) AS g(idx) "
                "%s)" % (
                    values, qident(left.bundle_table), qident(right.bundle_table),
                    qident(join_column), qident(join_column), h, lazy_joins,
                )
            )
            return ctes, "world_rows"

        if left_random:
            ctes.append(self._split_cte(left, "l", join_column, "left_split"))
        if right_random:
            ctes.append(self._split_cte(right, "r", join_column, "right_split"))

        if left_random and right_random:
            ctes.append(
                "joined_split AS ("
                "SELECT ls.rid AS left_rid, rs.rid AS right_rid, "
                "ARRAY(SELECT x FROM unnest(ls.repairs) x "
                "INTERSECT SELECT y FROM unnest(rs.repairs) y) AS repairs "
                "FROM left_split ls JOIN right_split rs "
                "ON ls.join_value = rs.join_value)"
            )
        elif left_random:
            ctes.append(
                "joined_split AS (SELECT ls.rid AS left_rid, r.%s AS right_rid, "
                "ls.repairs FROM left_split ls JOIN %s r "
                "ON ls.join_value = r.%s)" % (
                    qident("_rid"), qident(right.bundle_table), qident(join_column)
                )
            )
        else:
            ctes.append(
                "joined_split AS (SELECT l.%s AS left_rid, rs.rid AS right_rid, "
                "rs.repairs FROM %s l JOIN right_split rs "
                "ON l.%s = rs.join_value)" % (
                    qident("_rid"), qident(left.bundle_table), qident(join_column)
                )
            )

        values = selected_values("g.idx")
        ctes.append(
            "world_rows AS (SELECT g.idx%s FROM joined_split j "
            "JOIN %s l ON l.%s = j.left_rid "
            "JOIN %s r ON r.%s = j.right_rid "
            "CROSS JOIN LATERAL unnest(j.repairs) AS g(idx) "
            "%s "
            "WHERE cardinality(j.repairs) > 0)" % (
                values, qident(left.bundle_table), qident("_rid"),
                qident(right.bundle_table), qident("_rid"),
                lazy_joins,
            )
        )
        return ctes, "world_rows"

    @staticmethod
    def _predicate_from_values(predicates: Sequence[Predicate], value_for) -> str:
        if not predicates:
            return "TRUE"
        parts = []
        for predicate in predicates:
            operator = "<>" if predicate.operator == "!=" else predicate.operator
            parts.append("%s %s %s" % (
                value_for(predicate.column), operator, literal(predicate.value)
            ))
        return " AND ".join(parts)

    @staticmethod
    def _compact_output_items(spec: QuerySpec) -> List[SelectItem]:
        return [item for item in spec.select_items if not item.aggregate]

    @staticmethod
    def _compact_summary_tail(output_items: Sequence[SelectItem], h: int) -> str:
        output_columns = [qident(item.alias) for item in output_items]
        groups = ", ".join(output_columns)
        return (
            "merged_answers AS ("
            "SELECT %s, range_agg(repair_set) AS repair_set "
            "FROM contributions GROUP BY %s) "
            "SELECT %s, pg_temp.mcdb_repair_count(repair_set)::double precision "
            "/ %d AS probability "
            "FROM merged_answers WHERE NOT isempty(repair_set) "
            "ORDER BY %s"
        ) % (groups, groups, groups, h, groups)

    def _compact_single_summary(self, spec: QuerySpec,
                                relation: BundleRelation) -> Tuple[str, Tuple[str, ...]]:
        output_items = self._compact_output_items(spec)
        used_columns = []
        for item in output_items:
            if item.column and item.column not in used_columns:
                used_columns.append(item.column)
        for predicate in spec.predicates:
            if predicate.column not in used_columns:
                used_columns.append(predicate.column)
        random_columns = [
            column for column in used_columns if relation.is_random(column)
        ]

        fixed_output = ", ".join(
            "l.%s AS %s" % (qident(item.column or item.alias), qident(item.alias))
            for item in output_items
        )
        fixed_predicate = self._predicate_from_values(
            spec.predicates, lambda column: "l.%s" % qident(column)
        )
        if random_columns:
            fixed_rows = " AND ".join(
                "l.%s IS NOT NULL" % qident(column) for column in random_columns
            )
            random_rows = " OR ".join(
                "l.%s IS NULL" % qident(column) for column in random_columns
            )
        else:
            fixed_rows = "TRUE"
            random_rows = "FALSE"

        ctes = [
            "fixed_contributions AS ("
            "SELECT %s, l.%s AS repair_set "
            "FROM %s l WHERE %s AND %s AND NOT isempty(l.%s))" % (
                fixed_output,
                qident("__present"),
                qident(relation.bundle_table),
                fixed_rows,
                fixed_predicate,
                qident("__present"),
            )
        ]

        if random_columns:
            if (
                len(output_items) == 1
                and self._fixed_single_output_covers_donors(
                    relation,
                    output_items[0].column or output_items[0].alias,
                    fixed_rows,
                    fixed_predicate,
                )
            ):
                ctes.append(
                    "contributions AS (SELECT * FROM fixed_contributions)"
                )
                tail = self._compact_summary_tail(
                    output_items, relation.h
                )
                sql = "WITH " + ",\n".join(ctes) + ",\n" + tail
                columns = tuple(
                    item.alias for item in output_items
                ) + ("probability",)
                return sql, columns
            deterministic_predicates = [
                predicate for predicate in spec.predicates
                if not relation.is_random(predicate.column)
            ]
            deterministic_predicate = self._predicate_from_values(
                deterministic_predicates,
                lambda column: "l.%s" % qident(column),
            )
            lazy_joins, lazy_values = self._lazy_sampling_joins(
                relation, "l", "g.idx", used_columns, "compact_single"
            )

            def indexed_value(column: str) -> str:
                return lazy_values.get(column) or self._value_expression(
                    relation, "l", column, "g.idx"
                )

            indexed_output = ", ".join(
                "%s AS %s" % (
                    indexed_value(item.column or item.alias),
                    qident(item.alias),
                )
                for item in output_items
            )
            indexed_predicate = self._predicate_from_values(
                spec.predicates, indexed_value,
            )
            group_columns = ", ".join(qident(item.alias) for item in output_items)
            ctes.extend((
                "random_input AS MATERIALIZED ("
                "SELECT l.* FROM %s l "
                "WHERE (%s) AND %s AND NOT isempty(l.%s))" % (
                    qident(relation.bundle_table), random_rows,
                    deterministic_predicate, qident("__present"),
                ),
                "random_rows AS ("
                "SELECT l.%s AS source_rid, g.idx, %s "
                "FROM random_input l "
                "CROSS JOIN generate_series(1, %d) AS g(idx) "
                "%s "
                "WHERE g.idx <@ l.%s AND %s)" % (
                    qident("_rid"),
                    indexed_output,
                    relation.h,
                    lazy_joins,
                    qident("__present"),
                    indexed_predicate,
                ),
                "random_contributions AS ("
                "SELECT %s, range_agg("
                "int4range(idx::integer, idx::integer + 1)) AS repair_set "
                "FROM random_rows GROUP BY source_rid, %s)" % (
                    group_columns, group_columns
                ),
                "contributions AS ("
                "SELECT * FROM fixed_contributions "
                "UNION ALL SELECT * FROM random_contributions)",
            ))
        else:
            ctes.append("contributions AS (SELECT * FROM fixed_contributions)")

        tail = self._compact_summary_tail(output_items, relation.h)
        sql = "WITH " + ",\n".join(ctes) + ",\n" + tail
        columns = tuple(item.alias for item in output_items) + ("probability",)
        return sql, columns

    def _fixed_single_output_covers_donors(
        self,
        relation: BundleRelation,
        column: str,
        fixed_rows: str,
        fixed_predicate: str,
    ) -> bool:
        distribution = relation.lazy_distributions.get(column.lower())
        if distribution is None:
            return False
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT NOT EXISTS ("
            "SELECT 1 FROM ("
            "SELECT donor_value FROM %s "
            "EXCEPT "
            "SELECT l.%s FROM %s l "
            "WHERE %s AND %s AND NOT isempty(l.%s)"
            ") uncovered)"
            % (
                qident(distribution.donor_table),
                qident(column),
                qident(relation.bundle_table),
                fixed_rows,
                fixed_predicate,
                qident("__present"),
            )
        )
        return bool(cursor.fetchone()[0])

    def _key_fragments_cte(self, relation: BundleRelation, relation_alias: str,
                           join_column: str, cte_name: str) -> str:
        observed = (
            "SELECT %s.%s AS rid, %s.%s AS join_value, "
            "%s.%s AS repair_set FROM %s %s "
            "WHERE %s.%s IS NOT NULL AND NOT isempty(%s.%s)"
        ) % (
            relation_alias, qident("_rid"),
            relation_alias, qident(join_column),
            relation_alias, qident("__present"),
            qident(relation.bundle_table), relation_alias,
            relation_alias, qident(join_column),
            relation_alias, qident("__present"),
        )
        if not relation.is_random(join_column):
            return "%s AS (%s)" % (cte_name, observed)

        if join_column in relation.lazy_distributions:
            lazy_joins, lazy_values = self._lazy_sampling_joins(
                relation, relation_alias, "g.idx", [join_column],
                "%s_lazy_key" % cte_name,
            )
            join_value = lazy_values[join_column]
            sampled = (
                "SELECT %s.%s AS rid, sampled.join_value, "
                "range_agg(int4range(g.idx::integer, "
                "g.idx::integer + 1)) AS repair_set "
                "FROM %s %s CROSS JOIN generate_series(1, %d) AS g(idx) "
                "%s CROSS JOIN LATERAL (SELECT %s AS join_value) sampled "
                "WHERE %s.%s IS NULL AND sampled.join_value IS NOT NULL "
                "AND g.idx::integer <@ %s.%s "
                "GROUP BY %s.%s, sampled.join_value"
            ) % (
                relation_alias, qident("_rid"),
                qident(relation.bundle_table), relation_alias, relation.h,
                lazy_joins, join_value,
                relation_alias, qident(join_column),
                relation_alias, qident("__present"),
                relation_alias, qident("_rid"),
            )
        else:
            sampled = (
                "SELECT %s.%s AS rid, sampled.join_value, "
                "range_agg(int4range(sampled.idx::integer, "
                "sampled.idx::integer + 1)) AS repair_set "
                "FROM %s %s CROSS JOIN LATERAL "
                "unnest(%s.%s) WITH ORDINALITY AS sampled(join_value, idx) "
                "WHERE %s.%s IS NULL AND sampled.join_value IS NOT NULL "
                "AND sampled.idx::integer <@ %s.%s "
                "GROUP BY %s.%s, sampled.join_value"
            ) % (
                relation_alias, qident("_rid"),
                qident(relation.bundle_table), relation_alias,
                relation_alias, qident(relation.sample_column(join_column)),
                relation_alias, qident(join_column),
                relation_alias, qident("__present"),
                relation_alias, qident("_rid"),
            )
        return "%s AS (%s UNION ALL %s)" % (cte_name, observed, sampled)

    def _compact_join_summary(self, spec: QuerySpec, left: BundleRelation,
                              right: BundleRelation) -> Tuple[str, Tuple[str, ...]]:
        if left.h != right.h:
            raise ValueError("Joined bundles must use the same H")
        join_column = spec.join_column or ""
        output_items = self._compact_output_items(spec)

        def source_for(column: str):
            if column == join_column:
                return None, None
            source = self._column_source(column, left, right, join_column)
            return source, left if source == "l" else right

        used_columns = []
        for item in output_items:
            column = item.column or item.alias
            if column != join_column and column not in used_columns:
                used_columns.append(column)
        for predicate in spec.predicates:
            if predicate.column != join_column and predicate.column not in used_columns:
                used_columns.append(predicate.column)

        random_columns = []
        for column in used_columns:
            source, relation = source_for(column)
            if relation and relation.is_random(column):
                random_columns.append((source, relation, column))

        def fixed_value(column: str) -> str:
            if column == join_column:
                return "j.join_value"
            source, _relation = source_for(column)
            return "%s.%s" % (source, qident(column))

        left_used = [
            column for column in used_columns
            if source_for(column)[0] == "l"
        ]
        right_used = [
            column for column in used_columns
            if source_for(column)[0] == "r"
        ]
        left_lazy_joins, left_lazy_values = self._lazy_sampling_joins(
            left, "l", "g.idx", left_used, "compact_join_left"
        )
        right_lazy_joins, right_lazy_values = self._lazy_sampling_joins(
            right, "r", "g.idx", right_used, "compact_join_right"
        )
        lazy_joins = " ".join(
            value for value in (left_lazy_joins, right_lazy_joins) if value
        )

        def indexed_value(column: str) -> str:
            if column == join_column:
                return "j.join_value"
            source, relation = source_for(column)
            lazy_values = (
                left_lazy_values if source == "l" else right_lazy_values
            )
            if column in lazy_values:
                return lazy_values[column]
            return self._value_expression(relation, source, column, "g.idx")

        fixed_output = ", ".join(
            "%s AS %s" % (fixed_value(item.column or item.alias), qident(item.alias))
            for item in output_items
        )
        fixed_predicate = self._predicate_from_values(spec.predicates, fixed_value)
        if random_columns:
            fixed_rows = " AND ".join(
                "%s.%s IS NOT NULL" % (source, qident(column))
                for source, _relation, column in random_columns
            )
            random_rows = " OR ".join(
                "%s.%s IS NULL" % (source, qident(column))
                for source, _relation, column in random_columns
            )
        else:
            fixed_rows = "TRUE"
            random_rows = "FALSE"

        ctes = [
            self._key_fragments_cte(left, "l", join_column, "left_keys"),
            self._key_fragments_cte(right, "r", join_column, "right_keys"),
            "joined_bundles AS ("
            "SELECT lk.rid AS left_rid, rk.rid AS right_rid, lk.join_value, "
            "lk.repair_set * rk.repair_set AS repair_set "
            "FROM left_keys lk JOIN right_keys rk "
            "ON lk.join_value = rk.join_value "
            "WHERE NOT isempty(lk.repair_set * rk.repair_set))",
            "fixed_contributions AS ("
            "SELECT %s, j.repair_set "
            "FROM joined_bundles j "
            "JOIN %s l ON l.%s = j.left_rid "
            "JOIN %s r ON r.%s = j.right_rid "
            "WHERE %s AND %s)" % (
                fixed_output,
                qident(left.bundle_table), qident("_rid"),
                qident(right.bundle_table), qident("_rid"),
                fixed_rows, fixed_predicate,
            ),
        ]

        if random_columns:
            if (
                len(output_items) == 1
                and self._fixed_join_output_covers_donors(
                    spec,
                    left,
                    right,
                    output_items[0].column or output_items[0].alias,
                    random_columns,
                )
            ):
                ctes.append(
                    "contributions AS (SELECT * FROM fixed_contributions)"
                )
                tail = self._compact_summary_tail(
                    output_items, left.h
                )
                sql = "WITH " + ",\n".join(ctes) + ",\n" + tail
                columns = tuple(
                    item.alias for item in output_items
                ) + ("probability",)
                return sql, columns
            indexed_output = ", ".join(
                "%s AS %s" % (
                    indexed_value(item.column or item.alias), qident(item.alias)
                )
                for item in output_items
            )
            indexed_predicate = self._predicate_from_values(
                spec.predicates, indexed_value
            )
            group_columns = ", ".join(qident(item.alias) for item in output_items)
            ctes.extend((
                "random_rows AS ("
                "SELECT j.left_rid, j.right_rid, g.idx, %s "
                "FROM joined_bundles j "
                "JOIN %s l ON l.%s = j.left_rid "
                "JOIN %s r ON r.%s = j.right_rid "
                "CROSS JOIN generate_series(1, %d) AS g(idx) "
                "%s "
                "WHERE (%s) AND g.idx <@ j.repair_set AND %s)" % (
                    indexed_output,
                    qident(left.bundle_table), qident("_rid"),
                    qident(right.bundle_table), qident("_rid"),
                    left.h, lazy_joins, random_rows, indexed_predicate,
                ),
                "random_contributions AS ("
                "SELECT %s, range_agg("
                "int4range(idx::integer, idx::integer + 1)) AS repair_set "
                "FROM random_rows GROUP BY left_rid, right_rid, %s)" % (
                    group_columns, group_columns
                ),
                "contributions AS ("
                "SELECT * FROM fixed_contributions "
                "UNION ALL SELECT * FROM random_contributions)",
            ))
        else:
            ctes.append("contributions AS (SELECT * FROM fixed_contributions)")

        tail = self._compact_summary_tail(output_items, left.h)
        sql = "WITH " + ",\n".join(ctes) + ",\n" + tail
        columns = tuple(item.alias for item in output_items) + ("probability",)
        return sql, columns

    def _fixed_join_output_covers_donors(
        self,
        spec: QuerySpec,
        left: BundleRelation,
        right: BundleRelation,
        column: str,
        random_columns,
    ) -> bool:
        join_column = spec.join_column or ""
        if (
            column == join_column
            or left.is_random(join_column)
            or right.is_random(join_column)
        ):
            return False
        source = self._column_source(
            column, left, right, join_column
        )
        relation = left if source == "l" else right
        distribution = relation.lazy_distributions.get(column.lower())
        if distribution is None:
            return False

        def direct_value(value: str) -> str:
            if value == join_column:
                return "l.%s" % qident(join_column)
            value_source = self._column_source(
                value, left, right, join_column
            )
            return "%s.%s" % (value_source, qident(value))

        fixed_rows = " AND ".join(
            "%s.%s IS NOT NULL" % (value_source, qident(value))
            for value_source, _value_relation, value in random_columns
        )
        fixed_predicate = self._predicate_from_values(
            spec.predicates, direct_value
        )
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT NOT EXISTS ("
            "SELECT 1 FROM ("
            "SELECT donor_value FROM %s "
            "EXCEPT "
            "SELECT %s FROM %s l JOIN %s r "
            "ON l.%s = r.%s "
            "WHERE %s AND %s"
            ") uncovered)"
            % (
                qident(distribution.donor_table),
                direct_value(column),
                qident(left.bundle_table),
                qident(right.bundle_table),
                qident(join_column),
                qident(join_column),
                fixed_rows,
                fixed_predicate,
            )
        )
        return bool(cursor.fetchone()[0])

    def compile_compact_summary(self, query: str,
                                relations: Mapping[str, BundleRelation]):
        spec = parse_query(query)
        left = self._resolve_bundle(spec.left_token, relations)
        right = self._resolve_bundle(spec.right_token, relations) \
            if spec.right_token else None
        if spec.is_aggregation or spec.having is not None:
            needed = self._needed_columns(spec)
            if right:
                has_random_value = any(
                    (
                        left if self._column_source(
                            column, left, right, spec.join_column
                        ) == "l" else right
                    ).is_random(column)
                    for column in needed
                )
            else:
                has_random_value = any(
                    left.is_random(column) for column in needed
                )
            if not has_random_value:
                summary_sql, summary_columns = \
                    self._deterministic_aggregate_summary(
                        spec, left, right, needed
                    )
                return spec, summary_sql, summary_columns
            if (
                right is None
                and spec.having is None
                and all(
                    item.aggregate in (None, "avg")
                    for item in spec.select_items
                )
            ):
                summary_sql, summary_columns = \
                    self._compact_single_average_summary(
                        spec, left, needed
                )
                return spec, summary_sql, summary_columns
            if (
                right is not None
                and not spec.group_by
                and spec.having is None
                and not left.is_random(spec.join_column or "")
                and not right.is_random(spec.join_column or "")
                and all(
                    item.aggregate in (None, "avg")
                    for item in spec.select_items
                )
            ):
                summary_sql, summary_columns = \
                    self._compact_join_average_summary(
                        spec, left, right, needed
                    )
                return spec, summary_sql, summary_columns
            _spec, _world_sql, _world_columns, summary_sql, summary_columns = \
                self.compile(query, relations)
            return spec, summary_sql, summary_columns
        if right:
            summary_sql, summary_columns = self._compact_join_summary(
                spec, left, right
            )
        else:
            summary_sql, summary_columns = self._compact_single_summary(spec, left)
        return spec, summary_sql, summary_columns

    def _deterministic_aggregate_summary(
        self,
        spec: QuerySpec,
        left: BundleRelation,
        right: Optional[BundleRelation],
        needed: Sequence[str],
    ) -> Tuple[str, Tuple[str, ...]]:
        selected = []
        for column in needed:
            if right:
                source = self._column_source(
                    column, left, right, spec.join_column
                )
            else:
                source = "l"
            selected.append(
                "%s.%s AS %s" % (
                    source, qident(column), qident(column)
                )
            )
        suffix = ", " + ", ".join(selected) if selected else ""
        if right:
            source_sql = (
                "deterministic_rows AS ("
                "SELECT 1 AS idx%s FROM %s l JOIN %s r "
                "ON l.%s = r.%s)"
            ) % (
                suffix,
                qident(left.bundle_table),
                qident(right.bundle_table),
                qident(spec.join_column or ""),
                qident(spec.join_column or ""),
            )
        else:
            source_sql = (
                "deterministic_rows AS ("
                "SELECT 1 AS idx%s FROM %s l)"
            ) % (suffix, qident(left.bundle_table))
        answer_cte, _output_columns = self._answer_cte(
            spec, "deterministic_rows"
        )
        select_parts = [qident(column) for column in spec.group_by]
        columns: List[str] = list(spec.group_by)
        for item in spec.select_items:
            if not item.aggregate:
                continue
            expected = "expected_%s" % item.alias
            sample_stddev = "sample_stddev_%s" % item.alias
            select_parts.append(
                "%s::double precision AS %s" % (
                    qident(item.alias), qident(expected)
                )
            )
            select_parts.append(
                "CASE WHEN %s IS NULL THEN NULL::double precision "
                "ELSE 0::double precision END AS %s" % (
                    qident(item.alias), qident(sample_stddev)
                )
            )
            columns.extend((expected, sample_stddev))
        if spec.group_by:
            select_parts.append(
                "1::double precision AS group_probability"
            )
            columns.append("group_probability")
        sql = "WITH %s,\n%s SELECT %s FROM world_answer" % (
            source_sql, answer_cte, ", ".join(select_parts)
        )
        if spec.group_by:
            sql += " ORDER BY " + ", ".join(
                qident(column) for column in spec.group_by
            )
        return sql, tuple(columns)

    def _compact_single_average_summary(
        self,
        spec: QuerySpec,
        relation: BundleRelation,
        needed: Sequence[str],
    ) -> Tuple[str, Tuple[str, ...]]:
        random_columns = [
            column for column in needed if relation.is_random(column)
        ]
        fixed_rows = " AND ".join(
            "l.%s IS NOT NULL" % qident(column)
            for column in random_columns
        )
        random_rows = " OR ".join(
            "l.%s IS NULL" % qident(column)
            for column in random_columns
        )
        fixed_predicate = self._predicate_from_values(
            spec.predicates,
            lambda column: "l.%s" % qident(column),
        )
        deterministic_predicates = [
            predicate for predicate in spec.predicates
            if not relation.is_random(predicate.column)
        ]
        deterministic_predicate = self._predicate_from_values(
            deterministic_predicates,
            lambda column: "l.%s" % qident(column),
        )
        group_columns = list(spec.group_by)
        group_select = ", ".join(
            "l.%s AS %s" % (qident(column), qident(column))
            for column in group_columns
        )
        group_prefix = group_select + ", " if group_select else ""
        group_sql = ", ".join(qident(column) for column in group_columns)
        group_clause = " GROUP BY " + group_sql if group_sql else ""
        aggregate_items = [
            item for item in spec.select_items if item.aggregate == "avg"
        ]
        partials = []
        partial_aliases = []
        for item in aggregate_items:
            sum_alias = "__sum_%s" % item.alias
            count_alias = "__count_%s" % item.alias
            partials.extend((
                "sum(l.%s) AS %s" % (
                    qident(item.column or ""), qident(sum_alias)
                ),
                "count(l.%s)::bigint AS %s" % (
                    qident(item.column or ""), qident(count_alias)
                ),
            ))
            partial_aliases.extend((sum_alias, count_alias))
        ctes = [
            "fixed_stats AS ("
            "SELECT %s%s FROM %s l "
            "WHERE %s AND %s%s)" % (
                group_prefix,
                ", ".join(partials),
                qident(relation.bundle_table),
                fixed_rows,
                fixed_predicate,
                group_clause,
            )
        ]
        lazy_joins, lazy_values = self._lazy_sampling_joins(
            relation, "l", "g.idx", needed, "compact_average"
        )

        def indexed_value(column: str) -> str:
            return lazy_values.get(column) or self._value_expression(
                relation, "l", column, "g.idx"
            )

        indexed_values = ", ".join(
            "%s AS %s" % (indexed_value(column), qident(column))
            for column in needed
        )
        indexed_predicate = self._predicate_from_values(
            spec.predicates, indexed_value
        )
        random_group_select = ", ".join(
            qident(column) for column in group_columns
        )
        random_group_prefix = (
            random_group_select + ", " if random_group_select else ""
        )
        random_group_by = ", ".join(
            ["idx"] + [qident(column) for column in group_columns]
        )
        random_partials = []
        for item in aggregate_items:
            random_partials.extend((
                "sum(%s) AS %s" % (
                    qident(item.column or ""),
                    qident("__sum_%s" % item.alias),
                ),
                "count(%s)::bigint AS %s" % (
                    qident(item.column or ""),
                    qident("__count_%s" % item.alias),
                ),
            ))
        ctes.extend((
            "random_input AS MATERIALIZED ("
            "SELECT l.* FROM %s l WHERE (%s) AND %s "
            "AND NOT isempty(l.%s))" % (
                qident(relation.bundle_table),
                random_rows,
                deterministic_predicate,
                qident("__present"),
            ),
            "random_rows AS ("
            "SELECT g.idx, %s FROM random_input l "
            "CROSS JOIN generate_series(1, %d) AS g(idx) %s "
            "WHERE g.idx <@ l.%s AND %s)" % (
                indexed_values,
                relation.h,
                lazy_joins,
                qident("__present"),
                indexed_predicate,
            ),
            "random_stats AS ("
            "SELECT idx, %s%s FROM random_rows GROUP BY %s)" % (
                random_group_prefix,
                ", ".join(random_partials),
                random_group_by,
            ),
        ))
        contribution_columns = [
            qident(column) for column in group_columns
        ] + [qident(column) for column in partial_aliases]
        contribution_sql = ", ".join(contribution_columns)
        ctes.append(
            "contributions AS ("
            "SELECT g.idx, %s FROM fixed_stats f "
            "CROSS JOIN generate_series(1, %d) AS g(idx) "
            "UNION ALL SELECT idx, %s FROM random_stats)"
            % (contribution_sql, relation.h, contribution_sql)
        )
        aggregate_values = []
        output_columns = list(group_columns)
        for item in aggregate_items:
            aggregate_values.append(
                "sum(%s) / NULLIF(sum(%s), 0)::numeric AS %s" % (
                    qident("__sum_%s" % item.alias),
                    qident("__count_%s" % item.alias),
                    qident(item.alias),
                )
            )
            output_columns.append(item.alias)
        world_group_by = ", ".join(
            ["idx"] + [qident(column) for column in group_columns]
        )
        world_prefix = (
            ", ".join(qident(column) for column in group_columns) + ", "
            if group_columns else ""
        )
        ctes.append(
            "world_answer AS (SELECT idx, %s%s FROM contributions "
            "GROUP BY %s)" % (
                world_prefix,
                ", ".join(aggregate_values),
                world_group_by,
            )
        )
        summary, summary_columns = self._summary_sql(
            spec, output_columns, relation.h
        )
        return "WITH " + ",\n".join(ctes) + " " + summary, summary_columns

    def _compact_join_average_summary(
        self,
        spec: QuerySpec,
        left: BundleRelation,
        right: BundleRelation,
        needed: Sequence[str],
    ) -> Tuple[str, Tuple[str, ...]]:
        join_column = spec.join_column or ""

        def source_for(column: str):
            if column == join_column:
                return None, None
            source = self._column_source(
                column, left, right, join_column
            )
            return source, left if source == "l" else right

        def fixed_value(column: str) -> str:
            if column == join_column:
                return "j.join_value"
            source, _relation = source_for(column)
            return "%s.%s" % (source, qident(column))

        used_columns = [
            column for column in needed if column != join_column
        ]
        random_columns = []
        for column in used_columns:
            source, relation = source_for(column)
            if relation and relation.is_random(column):
                random_columns.append((source, relation, column))
        fixed_rows = " AND ".join(
            "%s.%s IS NOT NULL" % (source, qident(column))
            for source, _relation, column in random_columns
        )
        random_rows = " OR ".join(
            "%s.%s IS NULL" % (source, qident(column))
            for source, _relation, column in random_columns
        )
        fixed_predicate = self._predicate_from_values(
            spec.predicates, fixed_value
        )
        deterministic_predicates = []
        for predicate in spec.predicates:
            if predicate.column == join_column:
                deterministic_predicates.append(predicate)
                continue
            _source, relation = source_for(predicate.column)
            if relation and not relation.is_random(predicate.column):
                deterministic_predicates.append(predicate)
        deterministic_predicate = self._predicate_from_values(
            deterministic_predicates, fixed_value
        )
        aggregate_items = [
            item for item in spec.select_items if item.aggregate == "avg"
        ]
        fixed_partials = []
        partial_aliases = []
        for item in aggregate_items:
            sum_alias = "__sum_%s" % item.alias
            count_alias = "__count_%s" % item.alias
            value = fixed_value(item.column or "")
            fixed_partials.extend((
                "sum(%s) AS %s" % (value, qident(sum_alias)),
                "count(%s)::bigint AS %s" % (
                    value, qident(count_alias)
                ),
            ))
            partial_aliases.extend((sum_alias, count_alias))
        ctes = [
            self._key_fragments_cte(
                left, "l", join_column, "left_average_keys"
            ),
            self._key_fragments_cte(
                right, "r", join_column, "right_average_keys"
            ),
            "average_joined_bundles AS ("
            "SELECT lk.rid AS left_rid, rk.rid AS right_rid, "
            "lk.join_value, lk.repair_set * rk.repair_set AS repair_set "
            "FROM left_average_keys lk JOIN right_average_keys rk "
            "ON lk.join_value = rk.join_value "
            "WHERE NOT isempty(lk.repair_set * rk.repair_set))",
            "fixed_stats AS ("
            "SELECT %s FROM average_joined_bundles j "
            "JOIN %s l ON l.%s = j.left_rid "
            "JOIN %s r ON r.%s = j.right_rid "
            "WHERE %s AND %s)" % (
                ", ".join(fixed_partials),
                qident(left.bundle_table), qident("_rid"),
                qident(right.bundle_table), qident("_rid"),
                fixed_rows, fixed_predicate,
            ),
        ]
        left_used = [
            column for column in used_columns
            if source_for(column)[0] == "l"
        ]
        right_used = [
            column for column in used_columns
            if source_for(column)[0] == "r"
        ]
        left_lazy_joins, left_lazy_values = self._lazy_sampling_joins(
            left, "l", "g.idx", left_used, "compact_average_join_left"
        )
        right_lazy_joins, right_lazy_values = self._lazy_sampling_joins(
            right, "r", "g.idx", right_used, "compact_average_join_right"
        )
        lazy_joins = " ".join(
            value for value in (
                left_lazy_joins, right_lazy_joins
            ) if value
        )

        def indexed_value(column: str) -> str:
            if column == join_column:
                return "j.join_value"
            source, relation = source_for(column)
            lazy_values = (
                left_lazy_values if source == "l"
                else right_lazy_values
            )
            return lazy_values.get(column) or self._value_expression(
                relation, source, column, "g.idx"
            )

        indexed_values = ", ".join(
            "%s AS %s" % (indexed_value(column), qident(column))
            for column in needed
        )
        indexed_predicate = self._predicate_from_values(
            spec.predicates, indexed_value
        )
        random_partials = []
        for item in aggregate_items:
            random_partials.extend((
                "sum(%s) AS %s" % (
                    qident(item.column or ""),
                    qident("__sum_%s" % item.alias),
                ),
                "count(%s)::bigint AS %s" % (
                    qident(item.column or ""),
                    qident("__count_%s" % item.alias),
                ),
            ))
        ctes.extend((
            "random_join_input AS MATERIALIZED ("
            "SELECT j.left_rid, j.right_rid, j.join_value, "
            "j.repair_set FROM average_joined_bundles j "
            "JOIN %s l ON l.%s = j.left_rid "
            "JOIN %s r ON r.%s = j.right_rid "
            "WHERE (%s) AND %s)" % (
                qident(left.bundle_table), qident("_rid"),
                qident(right.bundle_table), qident("_rid"),
                random_rows, deterministic_predicate,
            ),
            "random_rows AS ("
            "SELECT g.idx, %s FROM random_join_input j "
            "JOIN %s l ON l.%s = j.left_rid "
            "JOIN %s r ON r.%s = j.right_rid "
            "CROSS JOIN generate_series(1, %d) AS g(idx) %s "
            "WHERE g.idx <@ j.repair_set AND %s)" % (
                indexed_values,
                qident(left.bundle_table), qident("_rid"),
                qident(right.bundle_table), qident("_rid"),
                left.h,
                lazy_joins,
                indexed_predicate,
            ),
            "random_stats AS ("
            "SELECT idx, %s FROM random_rows GROUP BY idx)" % (
                ", ".join(random_partials)
            ),
        ))
        contribution_columns = ", ".join(
            qident(column) for column in partial_aliases
        )
        ctes.append(
            "contributions AS ("
            "SELECT g.idx, %s FROM fixed_stats f "
            "CROSS JOIN generate_series(1, %d) AS g(idx) "
            "UNION ALL SELECT idx, %s FROM random_stats)"
            % (contribution_columns, left.h, contribution_columns)
        )
        aggregate_values = []
        output_columns = []
        for item in aggregate_items:
            aggregate_values.append(
                "sum(%s) / NULLIF(sum(%s), 0)::numeric AS %s" % (
                    qident("__sum_%s" % item.alias),
                    qident("__count_%s" % item.alias),
                    qident(item.alias),
                )
            )
            output_columns.append(item.alias)
        ctes.append(
            "world_answer AS (SELECT idx, %s FROM contributions "
            "GROUP BY idx)" % ", ".join(aggregate_values)
        )
        summary, summary_columns = self._summary_sql(
            spec, output_columns, left.h
        )
        return "WITH " + ",\n".join(ctes) + " " + summary, summary_columns

    @staticmethod
    def _predicate_sql(predicates: Sequence[Predicate]) -> str:
        if not predicates:
            return "TRUE"
        parts = []
        for predicate in predicates:
            operator = "<>" if predicate.operator == "!=" else predicate.operator
            parts.append("%s %s %s" % (
                qident(predicate.column), operator, literal(predicate.value)
            ))
        return " AND ".join(parts)

    @staticmethod
    def _aggregate_expression(item: SelectItem) -> str:
        if not item.aggregate:
            return qident(item.column or item.alias)
        argument = "*" if item.column is None else qident(item.column)
        return "%s(%s)" % (item.aggregate.upper(), argument)

    def _answer_cte(self, spec: QuerySpec, source_name: str) -> Tuple[str, Tuple[str, ...]]:
        where_sql = self._predicate_sql(spec.predicates)
        output_columns = tuple(item.alias for item in spec.select_items)
        if not spec.is_aggregation and spec.having is None:
            selections = ", ".join(
                "%s AS %s" % (qident(item.column or item.alias), qident(item.alias))
                for item in spec.select_items
            )
            return (
                "world_answer AS (SELECT DISTINCT idx, %s FROM %s WHERE %s)" % (
                    selections, source_name, where_sql
                ),
                output_columns,
            )

        selections = []
        for item in spec.select_items:
            if not item.aggregate and item.column in spec.group_by:
                continue
            selections.append("%s AS %s" % (
                self._aggregate_expression(item), qident(item.alias)
            ))
        group_columns = list(spec.group_by)
        group_sql = ", ".join(["idx"] + [qident(column) for column in group_columns])
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
            "world_answer AS (SELECT %s%s FROM %s WHERE %s GROUP BY %s%s)" % (
                group_sql,
                selection_suffix,
                source_name,
                where_sql,
                group_sql,
                having_sql,
            ),
            result_columns,
        )

    def _summary_sql(self, spec: QuerySpec, output_columns: Sequence[str], h: int) -> Tuple[str, Tuple[str, ...]]:
        if not spec.is_aggregation:
            groups = ", ".join(qident(column) for column in output_columns)
            return (
                "SELECT %s, count(*)::double precision / %d AS probability "
                "FROM world_answer GROUP BY %s ORDER BY %s" % (groups, h, groups, groups),
                tuple(output_columns) + ("probability",),
            )

        group_aliases = list(spec.group_by)
        aggregate_aliases = [
            item.alias for item in spec.select_items if item.aggregate
        ]
        select_parts = [qident(value) for value in group_aliases]
        summary_columns: List[str] = list(group_aliases)
        for value in aggregate_aliases:
            select_parts.append("avg(%s)::double precision AS %s" % (
                qident(value), qident("expected_%s" % value)
            ))
            select_parts.append("stddev_samp(%s)::double precision AS %s" % (
                qident(value), qident("sample_stddev_%s" % value)
            ))
            summary_columns.extend(["expected_%s" % value, "sample_stddev_%s" % value])
        if group_aliases:
            select_parts.append("count(DISTINCT idx)::double precision / %d AS group_probability" % h)
            summary_columns.append("group_probability")
            group_sql = ", ".join(qident(value) for value in group_aliases)
            return (
                "SELECT %s FROM world_answer GROUP BY %s ORDER BY %s" % (
                    ", ".join(select_parts), group_sql, group_sql
                ),
                tuple(summary_columns),
            )
        return "SELECT %s FROM world_answer" % ", ".join(select_parts), tuple(summary_columns)

    def compile(self, query: str,
                relations: Mapping[str, BundleRelation]) -> Tuple[QuerySpec, str, Tuple[str, ...], str, Tuple[str, ...]]:
        spec = parse_query(query)
        left = self._resolve_bundle(spec.left_token, relations)
        right = self._resolve_bundle(spec.right_token, relations) if spec.right_token else None
        needed = self._needed_columns(spec)
        if right:
            ctes, source = self._join_world_rows(left, right, spec.join_column or "", needed)
        else:
            ctes, source = self._single_world_rows(left, needed)
        answer_cte, output_columns = self._answer_cte(spec, source)
        ctes.append(answer_cte)
        summary_sql, summary_columns = self._summary_sql(spec, output_columns, left.h)
        with_sql = "WITH " + ",\n".join(ctes)
        world_sql = with_sql + " SELECT * FROM world_answer ORDER BY 1"
        final_summary_sql = with_sql + " " + summary_sql
        return spec, world_sql, ("idx",) + output_columns, final_summary_sql, summary_columns

    def compile_reference(self, query: str,
                          relations: Mapping[str, BundleRelation]) -> Tuple[str, Tuple[str, ...]]:
        """Compile explicit repair-by-repair relational evaluation.

        This intentionally expands each input relation before every operator.
        It is used as a correctness oracle for the bundle and Split plans.
        """
        spec = parse_query(query)
        left = self._resolve_bundle(spec.left_token, relations)
        right = self._resolve_bundle(spec.right_token, relations) if spec.right_token else None
        needed = self._needed_columns(spec)
        ctes: List[str] = []

        def expanded_cte(name: str, bundle: BundleRelation, alias: str,
                         relation_columns: Sequence[str]) -> str:
            values = [
                "%s AS %s" % (
                    self._value_expression(bundle, alias, column, "g.idx"),
                    qident(column),
                )
                for column in relation_columns
            ]
            suffix = ", " + ", ".join(values) if values else ""
            return "%s AS (SELECT g.idx, %s.%s AS rid%s FROM %s %s " \
                   "CROSS JOIN generate_series(1, %d) AS g(idx))" % (
                       name, alias, qident("_rid"), suffix,
                       qident(bundle.bundle_table), alias, bundle.h,
                   )

        left_columns = [column for column in needed if left.has_column(column)]
        if spec.join_column and spec.join_column not in left_columns:
            left_columns.append(spec.join_column)
        ctes.append(expanded_cte("reference_left", left, "lb", left_columns))

        if right:
            right_columns = [column for column in needed if right.has_column(column)]
            if spec.join_column and spec.join_column not in right_columns:
                right_columns.append(spec.join_column)
            ctes.append(expanded_cte("reference_right", right, "rb", right_columns))
            values = []
            for column in needed:
                source = self._column_source(column, left, right, spec.join_column)
                values.append("%s.%s AS %s" % (
                    source, qident(column), qident(column)
                ))
            suffix = ", " + ", ".join(values) if values else ""
            ctes.append(
                "world_rows AS (SELECT l.idx%s FROM reference_left l "
                "JOIN reference_right r ON l.idx = r.idx AND l.%s = r.%s)" % (
                    suffix, qident(spec.join_column or ""),
                    qident(spec.join_column or ""),
                )
            )
        else:
            values = ", ".join(
                "l.%s AS %s" % (qident(column), qident(column)) for column in needed
            )
            suffix = ", " + values if values else ""
            ctes.append("world_rows AS (SELECT l.idx%s FROM reference_left l)" % suffix)

        answer_cte, output_columns = self._answer_cte(spec, "world_rows")
        ctes.append(answer_cte)
        sql = "WITH " + ",\n".join(ctes) + " SELECT * FROM world_answer ORDER BY 1"
        return sql, ("idx",) + output_columns

    def evaluate(self, query: str,
                 relations: Mapping[str, BundleRelation]) -> QueryResult:
        spec, world_sql, world_columns, _summary_sql, _summary_columns = self.compile(
            query, relations
        )
        _compact_spec, summary_sql, summary_columns = self.compile_compact_summary(
            query, relations
        )
        cursor = self.connection.cursor()
        started = time.perf_counter()
        cursor.execute(world_sql)
        world_rows = cursor.fetchall()
        cursor.execute(summary_sql)
        summary_rows = cursor.fetchall()
        elapsed = time.perf_counter() - started
        return QueryResult(
            query=spec,
            world_rows=world_rows,
            summary_rows=summary_rows,
            world_columns=world_columns,
            summary_columns=summary_columns,
            sql=summary_sql,
            elapsed_s=elapsed,
        )

    def evaluate_summary(self, query: str,
                         relations: Mapping[str, BundleRelation]) -> QueryResult:
        """Evaluate a query once and return only its reported estimates."""
        spec, summary_sql, summary_columns = self.compile_compact_summary(
            query, relations
        )
        world_columns = ("idx",) + tuple(
            item.alias for item in spec.select_items
        )
        cursor = self.connection.cursor()
        started = time.perf_counter()
        cursor.execute(summary_sql)
        summary_rows = cursor.fetchall()
        elapsed = time.perf_counter() - started
        return QueryResult(
            query=spec,
            world_rows=[],
            summary_rows=summary_rows,
            world_columns=world_columns,
            summary_columns=summary_columns,
            sql=summary_sql,
            elapsed_s=elapsed,
        )


def relation_mapping(bundle: BundleRelation, *tokens: str) -> Dict[str, BundleRelation]:
    mapping = {bundle.base_table: bundle}
    for token in tokens:
        mapping[token] = bundle
        mapping[token.split("/")[-1]] = bundle
        mapping[token.lower()] = bundle
        mapping[token.split("/")[-1].lower()] = bundle
    return mapping


__all__ = [
    "BundleRelation",
    "CONN",
    "NativeMCDB",
    "Predicate",
    "QueryResult",
    "QuerySpec",
    "SelectItem",
    "parse_query",
    "relation_mapping",
]
