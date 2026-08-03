from collections import Counter

from MCDBPostgresNative import NativeMCDB, qident, relation_mapping


def main():
    engine = NativeMCDB.connect()
    cursor = engine.connection.cursor()
    try:
        cursor.execute(
            """
            CREATE TEMP TABLE lazy_chain (
                _rid bigint PRIMARY KEY,
                y integer,
                y_nullsym text,
                x integer,
                x_nullsym text,
                kind text
            ) ON COMMIT PRESERVE ROWS
            """
        )
        cursor.execute(
            """
            INSERT INTO lazy_chain VALUES
                (1, 10, NULL, 1, NULL, 'donor'),
                (2, 10, NULL, 1, NULL, 'donor'),
                (3, 20, NULL, 2, NULL, 'donor'),
                (4, 20, NULL, 2, NULL, 'donor'),
                (5, NULL, 'y_target', NULL, 'x_target', 'target'),
                (6, NULL, 'y_second', NULL, 'x_second', 'other')
            """
        )
        engine.connection.commit()
        bundle = engine.create_bundle(
            "lazy_chain",
            ("y", "x"),
            {"y": ("x",), "x": tuple()},
            h=400,
            seed=0.61,
            prefix="test_lazy_factor",
            lazy_samples=True,
        )
        assert set(bundle.lazy_distributions) == {"x", "y"}
        assert bundle.sample_tables == {}
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND column_name LIKE '__samples_%%'",
            (bundle.bundle_table,),
        )
        assert cursor.fetchall() == []

        result = engine.evaluate_summary(
            "SELECT x, y FROM lazy_chain WHERE y > 0",
            relation_mapping(bundle, "lazy_chain"),
        )
        probabilities = {
            (row[0], row[1]): float(row[2])
            for row in result.summary_rows
        }
        assert set(probabilities) == {(1, 10), (2, 20)}
        assert all(0.0 < value <= 1.0 for value in probabilities.values())

        target_result = engine.evaluate_summary(
            "SELECT x, y FROM lazy_chain WHERE kind = 'target'",
            relation_mapping(bundle, "lazy_chain"),
        )
        target_probabilities = {
            (row[0], row[1]): float(row[2])
            for row in target_result.summary_rows
        }
        assert set(target_probabilities) == {(1, 10), (2, 20)}
        assert abs(sum(target_probabilities.values()) - 1.0) < 1e-9
        assert all(
            abs(value - 0.5) < 0.1
            for value in target_probabilities.values()
        )

        precomputed = engine.create_bundle(
            "lazy_chain",
            ("y", "x"),
            {"y": ("x",), "x": tuple()},
            h=400,
            seed=0.61,
            prefix="test_precomputed_factor",
            precomputed_samples=True,
        )
        cursor.execute(
            "SELECT __samples_x, __samples_y FROM %s WHERE _rid = 5" %
            qident(precomputed.bundle_table)
        )
        x_samples, y_samples = cursor.fetchone()
        assert len(x_samples) == 400
        assert len(y_samples) == 400
        assert set(zip(x_samples, y_samples)) <= {(1, 10), (2, 20)}

        cursor.execute(
            """
            CREATE TEMP TABLE lazy_repeated (
                _rid bigint PRIMARY KEY,
                y integer,
                y_nullsym text
            ) ON COMMIT PRESERVE ROWS
            """
        )
        cursor.execute(
            """
            INSERT INTO lazy_repeated VALUES
                (1, 10, NULL),
                (2, 20, NULL),
                (3, NULL, 'shared'),
                (4, NULL, 'shared')
            """
        )
        engine.connection.commit()
        repeated = engine.create_bundle(
            "lazy_repeated",
            ("y",),
            {"y": tuple()},
            h=40,
            seed=0.41,
            prefix="test_lazy_repeated",
            lazy_samples=True,
        )
        assert repeated.lazy_distributions == {}
        assert set(repeated.sample_tables) == {"y"}
        cursor.execute(
            "SELECT a.__samples_y = b.__samples_y "
            "FROM %s a JOIN %s b ON a._rid = 3 AND b._rid = 4" % (
                qident(repeated.bundle_table),
                qident(repeated.bundle_table),
            )
        )
        assert cursor.fetchone()[0] is True
        repeated_precomputed = engine.create_bundle(
            "lazy_repeated",
            ("y",),
            {"y": tuple()},
            h=40,
            seed=0.41,
            prefix="test_precomputed_repeated",
            precomputed_samples=True,
        )
        cursor.execute(
            "SELECT a.__samples_y = b.__samples_y "
            "FROM %s a JOIN %s b ON a._rid = 3 AND b._rid = 4" % (
                qident(repeated_precomputed.bundle_table),
                qident(repeated_precomputed.bundle_table),
            )
        )
        assert cursor.fetchone()[0] is True
        print("All lazy CAMC tests passed")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
