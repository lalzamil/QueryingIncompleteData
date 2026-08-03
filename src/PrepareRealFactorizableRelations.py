#!/usr/bin/env python3
"""Create meaningful relations for the factorizable real datasets."""

from pathlib import Path

import pandas as pd


ROOT = Path("data")
OUTPUT = ROOT / "MI" / "real_factorizable_relations"

DATASETS = {
    "student": {
        "source": ROOT / "MI" / "student_admission_record_dirty.csv",
        "key": "id",
        "relations": {
            "student_applicant_profile": ["id", "age", "gender"],
            "student_academic_record": [
                "id",
                "admission_test_score",
                "high_school_percentage",
            ],
            "student_admission_record": [
                "id",
                "name",
                "city",
                "admission_status",
            ],
        },
    },
    "aircraft": {
        "source": ROOT / "MI" / "Aiplane_BlueBook.csv",
        "key": "id",
        "relations": {
            "aircraft_description": [
                "id",
                "model",
                "company",
                "engine_type",
                "fuel_gal_lbs",
                "gross_weight_lbs",
                "empty_weight_lbs",
                "length_ft_in",
                "height_ft_in",
                "wing_span_ft_in",
            ],
            "aircraft_flight_performance": [
                "id",
                "max_speed_knots",
                "rcmnd_cruise_knots",
                "stall_knots_dirty",
                "range_n_m",
            ],
            "aircraft_climb_ceiling": [
                "id",
                "all_eng_service_ceiling",
                "eng_out_service_ceiling",
                "all_eng_rate_of_climb",
                "eng_out_rate_of_climb",
            ],
            "aircraft_takeoff_landing": [
                "id",
                "takeoff_over_50ft",
                "takeoff_ground_run",
                "landing_over_50ft",
                "landing_ground_roll",
            ],
        },
    },
    "medical": {
        "source": ROOT / "MI" / "medical_conditions_dataset.csv",
        "key": "id",
        "relations": {
            "medical_patient": [
                "id",
                "full_name",
                "age",
                "gender",
                "smoking_status",
            ],
            "medical_measurements": [
                "id",
                "bmi",
                "blood_pressure",
                "glucose_levels",
            ],
            "medical_diagnosis": ["id", "condition"],
        },
    },
}


def with_key(frame, key):
    result = frame.copy()
    if key not in result.columns:
        result.insert(0, key, range(1, len(result) + 1))
    if result[key].isna().any() or result[key].duplicated().any():
        raise ValueError(f"{key} must be complete and unique")
    return result


def reconstruct(paths, key):
    frames = [pd.read_csv(path) for path in paths]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=key, how="inner", validate="one_to_one")
    return result


def same_values(expected, actual):
    expected = expected.sort_values("id").reset_index(drop=True)
    actual = actual[expected.columns].sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset, specification in DATASETS.items():
        source = with_key(pd.read_csv(specification["source"]), specification["key"])
        paths = []
        covered = {specification["key"]}
        for relation, columns in specification["relations"].items():
            overlap = covered.intersection(set(columns) - {specification["key"]})
            if overlap:
                raise ValueError(f"{dataset} repeats non-key columns: {sorted(overlap)}")
            covered.update(columns)
            relation_path = OUTPUT / f"{relation}.csv"
            source[columns].to_csv(relation_path, index=False)
            paths.append(relation_path)
        if covered != set(source.columns):
            missing = sorted(set(source.columns) - covered)
            extra = sorted(covered - set(source.columns))
            raise ValueError(f"{dataset} column mismatch; missing={missing}, extra={extra}")
        restored = reconstruct(paths, specification["key"])
        same_values(source, restored)
        print(
            f"{dataset}: {len(source)} rows, {len(paths)} relations, "
            "join reproduces the original relation"
        )


if __name__ == "__main__":
    main()
