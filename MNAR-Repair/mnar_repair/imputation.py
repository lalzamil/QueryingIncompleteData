"""Imputation functions for attributes selected by MNAR-Repair."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def _observed_fill_value(series: pd.Series):
    observed = series.dropna()
    if observed.empty:
        raise ValueError(f"Attribute '{series.name}' has no observed value.")
    if pd.api.types.is_numeric_dtype(series):
        return observed.median()
    modes = observed.mode(dropna=True)
    if modes.empty:
        raise ValueError(f"Attribute '{series.name}' has no observed mode.")
    return modes.iloc[0]


def simple_impute_selected(
    relation: pd.DataFrame,
    attributes: Iterable[str],
    mgraph=None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Impute selected numerical attributes by median and categorical attributes by mode."""
    del mgraph, random_state
    repaired = relation.copy(deep=True)
    for attribute in attributes:
        if attribute not in repaired:
            raise ValueError(f"Repair attribute '{attribute}' is absent from the relation.")
        repaired[attribute] = repaired[attribute].fillna(
            _observed_fill_value(repaired[attribute])
        )
    return repaired


def markov_blanket(
    attribute: str,
    mgraph: Mapping,
    available_attributes: Iterable[str],
) -> list[str]:
    """Return the available m-graph parents, children, and spouses of an attribute."""
    available = set(available_attributes)
    blanket = set(mgraph[attribute].get("parents", []))

    for child, information in mgraph.items():
        parents = set(information.get("parents", []))
        if attribute in parents:
            blanket.add(child)
            blanket.update(parents - {attribute})

    return sorted((blanket & available) - {attribute})


def _encode_frame(frame: pd.DataFrame):
    encoded = frame.copy()
    categories = {}
    for column in encoded:
        if not pd.api.types.is_numeric_dtype(encoded[column]):
            categorical = encoded[column].astype("category")
            categories[column] = list(categorical.cat.categories)
            encoded[column] = categorical.cat.codes.replace(-1, np.nan).astype(float)
        else:
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").astype(float)
    return encoded, categories


def _fill_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()
    for column in filled:
        observed = filled[column].dropna()
        value = float(observed.median()) if not observed.empty else 0.0
        if not math.isfinite(value):
            value = 0.0
        filled[column] = filled[column].fillna(value)
    return filled


def _decode_codes(series: pd.Series, categories: list):
    codes = series.round().astype("Int64")
    codes = codes.where((codes >= 0) & (codes < len(categories)), pd.NA)
    return codes.map(lambda value: categories[int(value)] if pd.notna(value) else np.nan)


def mbi_impute_selected(
    relation: pd.DataFrame,
    attributes: Iterable[str],
    mgraph: Mapping,
    random_state: int = 42,
    n_estimators: int = 80,
    max_fit_rows: int = 50_000,
) -> pd.DataFrame:
    """Impute selected attributes with random forests over their Markov blankets."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as error:
        raise ImportError(
            "Markov Blanket Imputation requires scikit-learn. "
            "Install MNAR-Repair/requirements.txt."
        ) from error

    repaired = relation.copy(deep=True)
    for attribute in attributes:
        if attribute not in repaired:
            raise ValueError(f"Repair attribute '{attribute}' is absent from the relation.")
        missing = repaired[attribute].isna()
        if not missing.any():
            continue

        predictors = markov_blanket(attribute, mgraph, repaired.columns)
        if not predictors:
            repaired[attribute] = repaired[attribute].fillna(
                _observed_fill_value(repaired[attribute])
            )
            continue

        encoded, categories = _encode_frame(repaired[[attribute] + predictors])
        target = encoded[attribute]
        training_indices = np.flatnonzero(target.notna().to_numpy())
        prediction_indices = np.flatnonzero(target.isna().to_numpy())
        if training_indices.size < 5:
            repaired[attribute] = repaired[attribute].fillna(
                _observed_fill_value(repaired[attribute])
            )
            continue

        if training_indices.size > max_fit_rows:
            generator = np.random.default_rng(random_state)
            training_indices = generator.choice(
                training_indices,
                size=max_fit_rows,
                replace=False,
            )

        predictors_frame = _fill_predictors(encoded[predictors])
        if attribute in categories:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(
                predictors_frame.iloc[training_indices],
                target.iloc[training_indices].round().astype(int),
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(
                predictors_frame.iloc[training_indices],
                target.iloc[training_indices],
            )

        predictions = model.predict(predictors_frame.iloc[prediction_indices])
        encoded.iloc[prediction_indices, encoded.columns.get_loc(attribute)] = predictions
        if attribute in categories:
            repaired[attribute] = _decode_codes(encoded[attribute], categories[attribute])
        else:
            repaired[attribute] = encoded[attribute]

    return repaired
