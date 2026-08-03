"""Correctness tests for the exact PostgreSQL FactorSampler."""

from collections import Counter

from FactorSamplerPostgres import (
    InvalidFactorization,
    NoMatchingDonor,
    PostgresFactorSampler,
    factors,
)
from LikeApxPostgresNative import NativeLikeApx, relation_mapping as like_mapping
from MCDBPostgresNative import NativeMCDB, qident, relation_mapping


def execute(cursor, statement):
    cursor.execute(statement)


def test_empirical_marginal(sampler, cursor):
    execute(cursor, """
        CREATE TEMP TABLE fs_marginal (
            _rid bigint PRIMARY KEY,
            x integer,
            x_nullsym text
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_marginal VALUES
            (1, 1, NULL),
            (2, 1, NULL),
            (3, 2, NULL),
            (4, NULL, 'x_target')
    """)
    sampler.connection.commit()
    x_factors = factors((("x",), tuple()))
    values = sampler.sample_marked_null(
        "fs_marginal", "x", "x_target", x_factors, h=400, seed=0.11
    )
    counts = Counter(values)
    assert set(counts) == {1, 2}
    proportion = counts[1] / len(values)
    assert 0.57 < proportion < 0.76, counts
    print("PASS empirical marginal preserves donor multiplicities")


def test_reverse_factorization(sampler, cursor):
    execute(cursor, """
        CREATE TEMP TABLE fs_chain (
            _rid bigint PRIMARY KEY,
            y integer,
            y_nullsym text,
            x integer,
            x_nullsym text
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_chain VALUES
            (1, 10, NULL, 1, NULL),
            (2, 10, NULL, 1, NULL),
            (3, 20, NULL, 2, NULL),
            (4, 20, NULL, 2, NULL),
            (5, NULL, 'y_target', NULL, 'x_target')
    """)
    sampler.connection.commit()
    y_factors = factors(
        (("y",), ("x",)),
        (("x",), tuple()),
    )
    x_factors = factors((("x",), tuple()))
    sampler.set_seed(0.21)
    observed = Counter()
    for _ in range(120):
        result = sampler.sample_from_tuple(
            "fs_chain", 5, "y", y_factors, {"x": x_factors}
        )
        assert (result.state["x"], result.value) in {(1, 10), (2, 20)}
        assert tuple(step.factor_index for step in result.steps) == (2, 1)
        observed[result.value] += 1
    assert set(observed) == {10, 20}
    print("PASS factors are sampled from last to first")


def test_observed_z_condition(sampler, cursor):
    execute(cursor, """
        CREATE TEMP TABLE fs_observed_z (
            _rid bigint PRIMARY KEY,
            y integer,
            y_nullsym text,
            c text,
            x integer
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_observed_z VALUES
            (1, 10, NULL, 'a', 1),
            (2, 11, NULL, 'b', 1),
            (3, NULL, 'y_target', 'b', 1)
    """)
    sampler.connection.commit()
    y_factors = factors(
        (("y", "c"), ("x",)),
        (("x",), tuple()),
    )
    sampler.set_seed(0.31)
    for _ in range(20):
        result = sampler.sample_from_tuple(
            "fs_observed_z", 3, "y", y_factors
        )
        assert result.value == 11
        assert result.steps[-1].conditioning_values == (("x", 1), ("c", "b"))
    print("PASS donor query conditions on X and observed Z attributes")


def test_repeated_target_and_bundle(engine, sampler, cursor):
    execute(cursor, """
        CREATE TEMP TABLE fs_repeated_target (
            _rid bigint PRIMARY KEY,
            y integer,
            y_nullsym text,
            x integer
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_repeated_target VALUES
            (1, 10, NULL, 1),
            (2, 20, NULL, 2),
            (3, NULL, 'shared_y', 1),
            (4, NULL, 'shared_y', 2)
    """)
    sampler.connection.commit()
    y_factors = factors(
        (("y",), ("x",)),
        (("x",), tuple()),
    )
    values = sampler.sample_marked_null(
        "fs_repeated_target", "y", "shared_y", y_factors,
        h=400, seed=0.41,
    )
    counts = Counter(values)
    assert set(counts) == {10, 20}
    assert 0.38 < counts[10] / len(values) < 0.62, counts

    bundle = sampler.create_bundle(
        "fs_repeated_target", {"y": y_factors}, h=30,
        prefix="test_exact_factor_bundle", seed=0.43,
    )
    sample_table = bundle.sample_tables["y"]
    cursor.execute(
        "SELECT cardinality(samples), used_fallback FROM %s "
        "WHERE symbol = 'shared_y'" % qident(sample_table)
    )
    assert cursor.fetchone() == (30, False)
    cursor.execute(
        "SELECT a.__samples_y = b.__samples_y FROM %s a JOIN %s b "
        "ON a._rid = 3 AND b._rid = 4" % (
            qident(bundle.bundle_table), qident(bundle.bundle_table)
        )
    )
    assert cursor.fetchone()[0] is True
    assert bundle.unresolved_draws == {"y": 0}

    query = "SELECT y FROM fs_repeated_target WHERE y > 0"
    mcdb_relations = relation_mapping(bundle, "fs_repeated_target")
    mcdb_result = engine.evaluate_summary(query, mcdb_relations)
    likeapx = NativeLikeApx(engine)
    like_relation = likeapx.create_relation(bundle, "test_factor_likeapx")
    like_result = likeapx.evaluate_summary(
        query, like_mapping(like_relation, "fs_repeated_target")
    )
    assert mcdb_result.summary_columns == like_result.summary_columns
    assert sorted(mcdb_result.summary_rows) == sorted(like_result.summary_rows)
    print("PASS repeated null uses occurrence frequencies and one shared array")


def test_repeated_separator_recursion(sampler, cursor):
    execute(cursor, """
        CREATE TEMP TABLE fs_recursive (
            _rid bigint PRIMARY KEY,
            y integer,
            y_nullsym text,
            x integer,
            x_nullsym text
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_recursive VALUES
            (1, 10, NULL, 1, NULL),
            (2, 20, NULL, 2, NULL),
            (3, NULL, 'y_target', NULL, 'shared_x'),
            (4, 30, NULL, NULL, 'shared_x')
    """)
    sampler.connection.commit()
    y_factors = factors(
        (("y",), ("x",)),
        (("x",), tuple()),
    )
    x_factors = factors((("x",), tuple()))
    sampler.set_seed(0.51)
    observed = Counter()
    for _ in range(120):
        result = sampler.sample_from_tuple(
            "fs_recursive", 3, "y", y_factors, {"x": x_factors}
        )
        assert (result.state["x"], result.value) in {(1, 10), (2, 20)}
        assert result.steps[0].target == "x"
        assert result.steps[-1].target == "y"
        observed[result.value] += 1
    assert set(observed) == {10, 20}
    print("PASS repeated separating-set null is sampled recursively")


def test_no_fallback_and_validation(sampler, cursor):
    execute(cursor, """
        CREATE TEMP TABLE fs_no_donor (
            _rid bigint PRIMARY KEY,
            y integer,
            y_nullsym text,
            x integer
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_no_donor VALUES
            (1, 10, NULL, 1),
            (2, NULL, 'y_target', 3)
    """)
    sampler.connection.commit()
    y_factors = factors(
        (("y",), ("x",)),
        (("x",), tuple()),
    )
    try:
        sampler.sample_from_tuple("fs_no_donor", 2, "y", y_factors)
    except NoMatchingDonor:
        pass
    else:
        raise AssertionError("FactorSampler silently used a fallback donor")

    invalid = factors((("y",), ("x",)))
    try:
        sampler.sample_from_tuple("fs_no_donor", 2, "y", invalid)
    except InvalidFactorization:
        pass
    else:
        raise AssertionError("FactorSampler accepted an invalid factor order")
    print("PASS empty donor groups fail and invalid orders are rejected")


def test_native_bulk_factor_sampler(engine, cursor):
    execute(cursor, """
        CREATE TEMP TABLE fs_native_chain (
            _rid bigint PRIMARY KEY,
            y integer,
            y_nullsym text,
            x integer,
            x_nullsym text
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_native_chain VALUES
            (1, 10, NULL, 1, NULL),
            (2, 10, NULL, 1, NULL),
            (3, 20, NULL, 2, NULL),
            (4, 20, NULL, 2, NULL),
            (5, NULL, 'y_target', NULL, 'x_target')
    """)
    engine.connection.commit()
    bundle = engine.create_bundle(
        "fs_native_chain",
        ("y", "x"),
        {"y": ("x",), "x": tuple()},
        h=160,
        seed=0.61,
        prefix="test_native_exact_factor",
    )
    cursor.execute(
        "SELECT __samples_x, __samples_y FROM %s WHERE _rid = 5" %
        qident(bundle.bundle_table)
    )
    x_samples, y_samples = cursor.fetchone()
    assert len(x_samples) == 160
    assert len(y_samples) == 160
    assert set(zip(x_samples, y_samples)) <= {(1, 10), (2, 20)}
    assert set(x_samples) == {1, 2}
    assert bundle.unresolved_draws == {"x": 0, "y": 0}

    execute(cursor, """
        CREATE TEMP TABLE fs_native_no_donor (
            _rid bigint PRIMARY KEY,
            y integer,
            y_nullsym text,
            x integer
        ) ON COMMIT PRESERVE ROWS
    """)
    execute(cursor, """
        INSERT INTO fs_native_no_donor VALUES
            (1, 10, NULL, 1),
            (2, NULL, 'y_target', 3)
    """)
    engine.connection.commit()
    try:
        engine.create_bundle(
            "fs_native_no_donor",
            ("y",),
            {"y": ("x",)},
            h=20,
            seed=0.63,
            prefix="test_native_exact_no_donor",
        )
    except ValueError as error:
        assert "no matching observed donor" in str(error)
        engine.connection.rollback()
    else:
        raise AssertionError("Native FactorSampler silently used a fallback donor")
    print("PASS native FactorSampler uses exact reverse-order groups and no fallback")


def main():
    engine = NativeMCDB.connect()
    sampler = PostgresFactorSampler(engine.connection)
    cursor = engine.connection.cursor()
    try:
        test_empirical_marginal(sampler, cursor)
        test_reverse_factorization(sampler, cursor)
        test_observed_z_condition(sampler, cursor)
        test_repeated_target_and_bundle(engine, sampler, cursor)
        test_repeated_separator_recursion(sampler, cursor)
        test_no_fallback_and_validation(sampler, cursor)
        test_native_bulk_factor_sampler(engine, cursor)
        print("All exact FactorSampler tests passed")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
