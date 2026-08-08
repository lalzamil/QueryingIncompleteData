"""Greedy selection of attributes for MNAR-Repair."""

from __future__ import annotations

from typing import Mapping

from .graph import incomplete_attributes, mnar_edges


def greedy_repair_set(
    mgraph,
    costs: Mapping[str, float] | None = None,
    missingness_rates: Mapping[str, float] | None = None,
) -> list[str]:
    """Return the greedy weighted-Set-Cover repair set.

    For an MNAR edge A -> R_B, either endpoint covers the edge. The gain of an
    attribute is therefore the number of uncovered edges incident to it.
    """
    costs = dict(costs or {})
    missingness_rates = dict(missingness_rates or {})
    incomplete = incomplete_attributes(mgraph)

    unknown_costs = set(costs) - incomplete
    if unknown_costs:
        names = ", ".join(sorted(unknown_costs))
        raise ValueError(f"Repair costs were supplied for non-incomplete attributes: {names}.")

    for attribute in incomplete:
        cost = float(costs.get(attribute, 1.0))
        if cost <= 0:
            raise ValueError(f"The repair cost of '{attribute}' must be positive.")
        costs[attribute] = cost

        rate = float(missingness_rates.get(attribute, 1.0))
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"The missingness rate of '{attribute}' must be in [0, 1].")
        missingness_rates[attribute] = rate

    uncovered = mnar_edges(mgraph)
    repair_set: list[str] = []

    while uncovered:
        candidates = []
        for attribute in sorted(incomplete - set(repair_set)):
            gain = sum(
                1
                for parent, child in uncovered
                if parent == attribute or child == attribute
            )
            if gain:
                candidates.append(
                    (
                        -(gain / costs[attribute]),
                        missingness_rates[attribute],
                        attribute,
                    )
                )

        if not candidates:
            remaining = ", ".join(
                f"{parent}->R_{child}" for parent, child in sorted(uncovered)
            )
            raise RuntimeError(f"No attribute covers the remaining MNAR edges: {remaining}.")

        _, _, selected = min(candidates)
        repair_set.append(selected)
        uncovered = {
            edge for edge in uncovered if selected not in edge
        }

    return repair_set
