"""Validation and update operations for m-graph metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd

SUPPORTED_MECHANISMS = {"FullyObserved", "MCAR", "MAR", "MNAR"}


def validate_mgraph(mgraph: Mapping[str, Mapping[str, Any]], columns=None) -> None:
    """Validate the compact m-graph representation accepted by MNAR-Repair."""
    if not isinstance(mgraph, Mapping) or not mgraph:
        raise ValueError("The m-graph must be a nonempty JSON object.")

    graph_attributes = set(mgraph)
    if columns is not None:
        relation_attributes = set(columns)
        absent = graph_attributes - relation_attributes
        if absent:
            names = ", ".join(sorted(absent))
            raise ValueError(f"M-graph attributes absent from the relation: {names}.")

    for attribute, information in mgraph.items():
        if not isinstance(information, Mapping):
            raise ValueError(f"The m-graph entry for '{attribute}' must be an object.")
        mechanism = information.get("mechanism")
        if mechanism not in SUPPORTED_MECHANISMS:
            labels = ", ".join(sorted(SUPPORTED_MECHANISMS))
            raise ValueError(
                f"Unsupported mechanism '{mechanism}' for '{attribute}'. "
                f"Expected one of: {labels}."
            )
        parents = information.get("parents", [])
        if not isinstance(parents, list) or not all(isinstance(parent, str) for parent in parents):
            raise ValueError(f"The parents of '{attribute}' must be a list of attribute names.")
        absent_parents = set(parents) - graph_attributes
        if absent_parents:
            names = ", ".join(sorted(absent_parents))
            raise ValueError(f"Parents of '{attribute}' absent from the m-graph: {names}.")


def incomplete_attributes(mgraph: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Return the attributes represented as incomplete in the m-graph."""
    validate_mgraph(mgraph)
    return {
        attribute
        for attribute, information in mgraph.items()
        if information["mechanism"] != "FullyObserved"
    }


def mnar_edges(mgraph: Mapping[str, Mapping[str, Any]]) -> set[tuple[str, str]]:
    """Return pairs (A, B) representing MNAR edges A -> R_B."""
    incomplete = incomplete_attributes(mgraph)
    return {
        (parent, child)
        for child, information in mgraph.items()
        if child in incomplete
        for parent in information.get("parents", [])
        if parent in incomplete
    }


def update_mgraph(
    mgraph: Mapping[str, Mapping[str, Any]],
    relation: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Return the m-graph after accounting for the relation's complete attributes."""
    validate_mgraph(mgraph, relation.columns)
    updated = deepcopy(dict(mgraph))
    is_incomplete = {
        attribute: bool(relation[attribute].isna().any())
        for attribute in updated
    }

    for attribute, missing in is_incomplete.items():
        if not missing:
            updated[attribute]["mechanism"] = "FullyObserved"
            updated[attribute]["parents"] = []

    for attribute, missing in is_incomplete.items():
        if not missing:
            continue
        parents = updated[attribute].get("parents", [])
        if not parents:
            updated[attribute]["mechanism"] = "MCAR"
        elif any(is_incomplete[parent] for parent in parents):
            updated[attribute]["mechanism"] = "MNAR"
        else:
            updated[attribute]["mechanism"] = "MAR"

    return updated
