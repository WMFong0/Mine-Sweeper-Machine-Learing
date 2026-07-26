from __future__ import annotations

import math
from dataclasses import dataclass

from minesweeper_ml.constraints import Constraint, infer_mine_probabilities
from minesweeper_ml.game import Coordinate


_CERTAINTY_EPSILON = 1e-12


@dataclass(frozen=True)
class LookaheadEvaluation:
    expected_forced_cells: float
    expected_entropy_reduction: float
    zero_region_probability: float
    search_nodes: int
    budget_exhausted: bool
    valid: bool


def evaluate_safe_click(
    candidate: Coordinate,
    *,
    hidden_neighbors: frozenset[Coordinate],
    has_known_mine_neighbor: bool,
    constraints: list[Constraint],
    hidden_cells: set[Coordinate],
    model_mine_probabilities: dict[Coordinate, float],
    remaining_mines: int | None,
    prior_strength: float,
    max_constraint_nodes: int,
    max_total_search_nodes: int,
    min_outcome_probability: float,
) -> LookaheadEvaluation:
    if max_total_search_nodes <= 0:
        raise ValueError("max_total_search_nodes must be greater than zero.")
    if not 0.0 <= min_outcome_probability <= 1.0:
        raise ValueError(
            "min_outcome_probability must be between zero and one."
        )

    safe_constraints = [
        *constraints,
        (frozenset({candidate}), 0),
    ]
    conditioned = infer_mine_probabilities(
        safe_constraints,
        hidden_cells=hidden_cells,
        model_mine_probabilities=model_mine_probabilities,
        remaining_mines=remaining_mines,
        prior_strength=prior_strength,
        max_search_nodes=max_constraint_nodes,
        max_total_search_nodes=max_total_search_nodes,
        query_cells=hidden_neighbors,
    )
    search_nodes = conditioned.search_nodes
    if (
        conditioned.budget_exhausted
        or conditioned.overflowed_boxes
        or not conditioned.consistent
    ):
        return _invalid_evaluation(
            search_nodes,
            budget_exhausted=conditioned.budget_exhausted,
        )

    conditioned_probabilities = _complete_probabilities(
        hidden_cells,
        model_mine_probabilities,
        conditioned.mine_probabilities,
    )
    outcome_distribution = (
        conditioned.query_mine_count_distribution
    )
    if outcome_distribution is None:
        return _invalid_evaluation(
            search_nodes,
            budget_exhausted=False,
        )
    scored_cells = hidden_cells - {candidate}
    baseline_forced = _forced_cells(
        scored_cells,
        conditioned_probabilities,
    )
    baseline_entropy = _total_entropy(
        scored_cells,
        conditioned_probabilities,
    )

    valid_outcome_mass = 0.0
    weighted_forced_cells = 0.0
    weighted_entropy_reduction = 0.0
    zero_region_mass = 0.0

    for mine_count, outcome_probability in sorted(
        outcome_distribution.items()
    ):
        if (
            outcome_probability <= 0.0
            or outcome_probability < min_outcome_probability
        ):
            continue

        remaining_node_budget = max_total_search_nodes - search_nodes
        if remaining_node_budget <= 0:
            return _invalid_evaluation(
                search_nodes,
                budget_exhausted=True,
            )
        outcome = infer_mine_probabilities(
            [
                *safe_constraints,
                (hidden_neighbors, mine_count),
            ],
            hidden_cells=hidden_cells,
            model_mine_probabilities=model_mine_probabilities,
            remaining_mines=remaining_mines,
            prior_strength=prior_strength,
            max_search_nodes=max_constraint_nodes,
            max_total_search_nodes=remaining_node_budget,
        )
        search_nodes += outcome.search_nodes
        if outcome.budget_exhausted:
            return _invalid_evaluation(
                search_nodes,
                budget_exhausted=True,
            )
        if outcome.overflowed_boxes:
            return _invalid_evaluation(
                search_nodes,
                budget_exhausted=False,
            )
        if not outcome.consistent:
            continue

        outcome_probabilities = _complete_probabilities(
            hidden_cells,
            model_mine_probabilities,
            outcome.mine_probabilities,
        )
        outcome_forced = _forced_cells(
            scored_cells,
            outcome_probabilities,
        )
        newly_forced_count = len(outcome_forced - baseline_forced)
        entropy_reduction = baseline_entropy - _total_entropy(
            scored_cells,
            outcome_probabilities,
        )
        valid_outcome_mass += outcome_probability
        weighted_forced_cells += outcome_probability * newly_forced_count
        weighted_entropy_reduction += (
            outcome_probability * entropy_reduction
        )
        if mine_count == 0 and not has_known_mine_neighbor:
            zero_region_mass += outcome_probability

    if valid_outcome_mass <= 0.0:
        return _invalid_evaluation(
            search_nodes,
            budget_exhausted=False,
        )

    return LookaheadEvaluation(
        expected_forced_cells=weighted_forced_cells / valid_outcome_mass,
        expected_entropy_reduction=(
            weighted_entropy_reduction / valid_outcome_mass
        ),
        zero_region_probability=zero_region_mass / valid_outcome_mass,
        search_nodes=search_nodes,
        budget_exhausted=False,
        valid=True,
    )


def _complete_probabilities(
    hidden_cells: set[Coordinate],
    model_mine_probabilities: dict[Coordinate, float],
    inferred_probabilities: dict[Coordinate, float],
) -> dict[Coordinate, float]:
    return {
        cell: inferred_probabilities.get(
            cell,
            model_mine_probabilities[cell],
        )
        for cell in hidden_cells
    }


def _forced_cells(
    cells: set[Coordinate],
    probabilities: dict[Coordinate, float],
) -> set[Coordinate]:
    return {
        cell
        for cell in cells
        if probabilities[cell] <= _CERTAINTY_EPSILON
        or probabilities[cell] >= 1.0 - _CERTAINTY_EPSILON
    }


def _total_entropy(
    cells: set[Coordinate],
    probabilities: dict[Coordinate, float],
) -> float:
    return sum(_binary_entropy(probabilities[cell]) for cell in cells)


def _binary_entropy(probability: float) -> float:
    if (
        probability <= _CERTAINTY_EPSILON
        or probability >= 1.0 - _CERTAINTY_EPSILON
    ):
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def _invalid_evaluation(
    search_nodes: int,
    *,
    budget_exhausted: bool,
) -> LookaheadEvaluation:
    return LookaheadEvaluation(
        expected_forced_cells=0.0,
        expected_entropy_reduction=0.0,
        zero_region_probability=0.0,
        search_nodes=search_nodes,
        budget_exhausted=budget_exhausted,
        valid=False,
    )
