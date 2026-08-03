"""Correctness tests for MCDBPostgresNative.

Every optimized tuple-bundle result is compared with an explicit relation in
which each tuple is expanded once per repair before query evaluation.
"""

from collections import Counter
from decimal import Decimal

from LikeApxPostgresNative import NativeLikeApx, relation_mapping as like_mapping
from LikeApxRewrittenPostgres import (
    RewrittenLikeApx,
    relation_mapping as rewritten_like_mapping,
)
from MCDBPostgresNative import CONN, NativeMCDB, relation_mapping


def normalized(rows):
    def value(item):
        if isinstance(item, Decimal):
            return float(item)
        return item
    return Counter(tuple(value(item) for item in row) for row in rows)


def assert_matches(engine, query, relations):
    optimized = engine.evaluate(query, relations)
    reference_sql, reference_columns = engine.compile_reference(query, relations)
    cursor = engine.connection.cursor()
    cursor.execute(reference_sql)
    reference = cursor.fetchall()
    assert optimized.world_columns == reference_columns
    assert normalized(optimized.world_rows) == normalized(reference), (
        query,
        normalized(optimized.world_rows) - normalized(reference),
        normalized(reference) - normalized(optimized.world_rows),
    )
    _spec, _world_sql, _world_columns, summary_sql, summary_columns = \
        engine.compile(query, relations)
    cursor.execute(summary_sql)
    expanded_summary = cursor.fetchall()
    assert optimized.summary_columns == summary_columns
    assert normalized(optimized.summary_rows) == normalized(expanded_summary), (
        query,
        normalized(optimized.summary_rows) - normalized(expanded_summary),
        normalized(expanded_summary) - normalized(optimized.summary_rows),
    )
    print("PASS", query)


def assert_like_matches(engine, likeapx, query, mcdb_relations, like_relations,
                        label="LikeApx"):
    mcdb = engine.evaluate_summary(query, mcdb_relations)
    encoded = likeapx.evaluate_summary(query, like_relations)
    assert mcdb.summary_columns == encoded.summary_columns
    assert normalized(mcdb.summary_rows) == normalized(encoded.summary_rows), query
    print("PASS %s" % label, query)


def main():
    engine = NativeMCDB.connect()
    cursor = engine.connection.cursor()
    cursor.execute("""
        CREATE TEMP TABLE native_single (
            _rid BIGINT PRIMARY KEY,
            g TEXT,
            y INTEGER,
            y_nullsym TEXT,
            z INTEGER,
            z_nullsym TEXT,
            x INTEGER
        ) ON COMMIT PRESERVE ROWS
    """)
    cursor.execute("""
        INSERT INTO native_single VALUES
            (1, 'a', NULL, 'shared_y', NULL, 'z_1', 1),
            (2, 'b', NULL, 'shared_y', 10, NULL, 2),
            (3, 'a', NULL, 'single_y', NULL, 'z_3', 3),
            (4, 'a', 10, NULL, 20, NULL, 4),
            (5, 'a', 20, NULL, NULL, 'z_5', 5),
            (6, 'b', 30, NULL, 30, NULL, 6),
            (7, 'b', 40, NULL, 40, NULL, 7)
    """)
    engine.connection.commit()

    single = engine.create_bundle(
        "native_single", ["y", "z"], {"y": ["g"], "z": ["g"]}, h=25,
        seed=0.19, prefix="test_single", strict=True,
    )
    cursor.execute(
        "SELECT a.__samples_y = b.__samples_y "
        "FROM %s a JOIN %s b ON a._rid = 1 AND b._rid = 2" %
        ('"%s"' % single.bundle_table, '"%s"' % single.bundle_table)
    )
    assert cursor.fetchone()[0] is True

    single_relations = relation_mapping(single, "native_single")
    likeapx = NativeLikeApx(engine)
    like_single = likeapx.create_relation(single, "test_like_single")
    like_single_relations = like_mapping(like_single, "native_single")
    rewritten = RewrittenLikeApx(engine)
    rewritten_single = rewritten.create_relation(single, "test_rewritten_single")
    rewritten_single_relations = rewritten_like_mapping(
        rewritten_single, "native_single"
    )
    single_queries = [
        "SELECT g FROM native_single WHERE y > 15",
        "SELECT y FROM native_single WHERE x >= 1",
        "SELECT y FROM native_single WHERE y > 15",
        "SELECT y FROM native_single WHERE z > 15",
        "SELECT y, g FROM native_single GROUP BY y, g",
        "SELECT g, COUNT(*) AS cnt FROM native_single "
        "WHERE y > 15 GROUP BY g HAVING COUNT(*) > 1",
        "SELECT AVG(y) FROM native_single WHERE x > 0",
        "SELECT AVG(y) FROM native_single GROUP BY g",
    ]
    for query in single_queries:
        _spec, rewritten_sql, _columns = rewritten.compile(
            query, rewritten_single_relations
        )
        assert rewritten_sql.count("\nUNION ALL\n") == single.h - 1
        assert_matches(engine, query, single_relations)
        assert_like_matches(
            engine, likeapx, query, single_relations, like_single_relations
        )
        assert_like_matches(
            engine, rewritten, query, single_relations,
            rewritten_single_relations, "LikeApx rewritten",
        )

    cursor.execute("""
        CREATE TEMP TABLE native_left (
            _rid BIGINT PRIMARY KEY,
            k INTEGER,
            k_nullsym TEXT,
            lv TEXT,
            score INTEGER,
            measure INTEGER,
            measure_nullsym TEXT
        ) ON COMMIT PRESERVE ROWS
    """)
    cursor.execute("""
        INSERT INTO native_left VALUES
            (1, NULL, 'left_k_1', 'l1', 5, NULL, 'measure_1'),
            (2, NULL, 'left_k_2', 'l2', 8, 80, NULL),
            (3, 1, NULL, 'l3', 4, NULL, 'measure_3'),
            (4, 2, NULL, 'l4', -1, 40, NULL),
            (5, 3, NULL, 'l5', 9, 90, NULL)
    """)
    cursor.execute("""
        CREATE TEMP TABLE native_right (
            _rid BIGINT PRIMARY KEY,
            k INTEGER,
            k_nullsym TEXT,
            rv TEXT
        ) ON COMMIT PRESERVE ROWS
    """)
    cursor.execute("""
        INSERT INTO native_right VALUES
            (1, NULL, 'right_k_1', 'r1'),
            (2, 1, NULL, 'r2'),
            (3, 2, NULL, 'r3'),
            (4, 3, NULL, 'r4')
    """)
    engine.connection.commit()

    left = engine.create_bundle(
        "native_left", ["k", "measure"], {"k": [], "measure": []}, h=25,
        seed=0.31, prefix="test_left", strict=True,
    )
    cursor.execute(
        "SELECT count(DISTINCT __samples_k) FROM %s WHERE k IS NULL" %
        ('"%s"' % left.bundle_table)
    )
    assert cursor.fetchone()[0] > 1
    right = engine.create_bundle(
        "native_right", ["k"], {"k": []}, h=25,
        seed=0.31, prefix="test_right", strict=True,
    )
    join_relations = {}
    join_relations.update(relation_mapping(left, "native_left"))
    join_relations.update(relation_mapping(right, "native_right"))
    like_left = likeapx.create_relation(left, "test_like_left")
    like_right = likeapx.create_relation(right, "test_like_right")
    like_join_relations = {}
    like_join_relations.update(like_mapping(like_left, "native_left"))
    like_join_relations.update(like_mapping(like_right, "native_right"))
    rewritten_left = rewritten.create_relation(left, "test_rewritten_left")
    rewritten_right = rewritten.create_relation(right, "test_rewritten_right")
    rewritten_join_relations = {}
    rewritten_join_relations.update(
        rewritten_like_mapping(rewritten_left, "native_left")
    )
    rewritten_join_relations.update(
        rewritten_like_mapping(rewritten_right, "native_right")
    )
    join_queries = [
        "SELECT lv FROM native_left JOIN native_right USING (k) WHERE score > 0",
        "SELECT k FROM native_left JOIN native_right USING (k) WHERE score > 0",
        "SELECT measure FROM native_left JOIN native_right USING (k) "
        "WHERE score > 0",
        "SELECT AVG(score) FROM native_left JOIN native_right USING (k) "
        "WHERE score > 0",
    ]
    for query in join_queries:
        _spec, rewritten_sql, _columns = rewritten.compile(
            query, rewritten_join_relations
        )
        assert rewritten_sql.count("\nUNION ALL\n") == left.h - 1
        assert_matches(engine, query, join_relations)
        assert_like_matches(
            engine, likeapx, query, join_relations, like_join_relations
        )
        assert_like_matches(
            engine, rewritten, query, join_relations,
            rewritten_join_relations, "LikeApx rewritten",
        )

    print("All PostgreSQL-native MCDB tests passed")
    engine.close()


if __name__ == "__main__":
    main()
