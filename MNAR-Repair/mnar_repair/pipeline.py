"""End-to-end MNAR-Repair pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from .graph import mnar_edges, update_mgraph, validate_mgraph
from .imputation import mbi_impute_selected, simple_impute_selected
from .selection import greedy_repair_set

Imputer = Callable[[pd.DataFrame, list[str], Mapping[str, Any], int], pd.DataFrame]


@dataclass(frozen=True)
class RepairResult:
    """The selected attributes, repaired relation, and updated m-graph."""

    repair_set: tuple[str, ...]
    relation: pd.DataFrame
    mgraph: dict[str, dict[str, Any]]


def repair_relation(
    relation: pd.DataFrame,
    mgraph: Mapping[str, Mapping[str, Any]],
    costs: Mapping[str, float] | None = None,
    imputer: str | Imputer = "mbi",
    random_state: int = 42,
) -> RepairResult:
    """Select and impute an MNAR-Repair without modifying either input object."""
    validate_mgraph(mgraph, relation.columns)
    original = relation.copy(deep=True)
    current_mgraph = update_mgraph(mgraph, original)
    missingness_rates = {
        attribute: float(original[attribute].isna().mean())
        for attribute, information in current_mgraph.items()
        if information["mechanism"] != "FullyObserved"
    }
    selected = greedy_repair_set(current_mgraph, costs, missingness_rates)

    if isinstance(imputer, str):
        implementations = {
            "simple": simple_impute_selected,
            "mbi": mbi_impute_selected,
        }
        if imputer not in implementations:
            choices = ", ".join(sorted(implementations))
            raise ValueError(f"Unknown imputer '{imputer}'. Expected one of: {choices}.")
        imputation_function = implementations[imputer]
    elif callable(imputer):
        imputation_function = imputer
    else:
        raise TypeError("The imputer must be a supported name or a callable.")

    repaired = imputation_function(
        original.copy(deep=True),
        selected,
        current_mgraph,
        random_state,
    )
    if not isinstance(repaired, pd.DataFrame):
        raise TypeError("The imputation function must return a pandas DataFrame.")
    if list(repaired.columns) != list(original.columns) or not repaired.index.equals(original.index):
        raise ValueError("The imputation function changed the relation schema or tuple index.")

    selected_attributes = set(selected)
    for attribute in original:
        observed = original[attribute].notna()
        if not repaired.loc[observed, attribute].equals(original.loc[observed, attribute]):
            raise RuntimeError(
                f"The imputation function changed an observed value in '{attribute}'."
            )
        if attribute not in selected_attributes and not repaired[attribute].equals(original[attribute]):
            raise RuntimeError(
                f"The imputation function changed non-selected attribute '{attribute}'."
            )

    for attribute in selected:
        if repaired[attribute].isna().any():
            count = int(repaired[attribute].isna().sum())
            raise RuntimeError(
                f"The imputation function left {count} missing cells in '{attribute}'."
            )

    updated = update_mgraph(current_mgraph, repaired)
    remaining = mnar_edges(updated)
    if remaining:
        display = ", ".join(
            f"{parent}->R_{child}" for parent, child in sorted(remaining)
        )
        raise RuntimeError(f"The repaired relation still contains MNAR edges: {display}.")

    return RepairResult(tuple(selected), repaired, updated)
