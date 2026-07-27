from __future__ import annotations

import math
from dataclasses import dataclass

from minesweeper_ml.constraints import (
    Constraint,
    ConstraintInferenceContext,
    build_constraint_context,
    infer_safe_click_outcomes,
    infer_mine_probabilities,
)
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


def poisson_binomial_distribution(
    probabilities: list[float],
) -> dict[int, float]:
    distribution = {0: 1.0}
    for probability in probabilities:
        updated: dict[int, float] = {}
        for mine_count, weight in distribution.items():
            updated[mine_count] = (
                updated.get(mine_count, 0.0)
                + weight * (1.0 - probability)
            )
            updated[mine_count + 1] = (
                updated.get(mine_count + 1, 0.0)
                + weight * probability
            )
        distribution = updated
    return distribution


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
    inference_context: ConstraintInferenceContext | None = None,
    exact_outcomes: bool = True,
) -> LookaheadEvaluation:
    if max_total_search_nodes <= 0:
        raise ValueError("max_total_search_nodes must be greater than zero.")
    if not 0.0 <= min_outcome_probability <= 1.0:
        raise ValueError(
            "min_outcome_probability must be between zero and one."
        )
    if not exact_outcomes:
        return _evaluate_independent_safe_click(
            candidate,
            hidden_neighbors=hidden_neighbors,
            has_known_mine_neighbor=has_known_mine_neighbor,
            constraints=constraints,
            hidden_cells=hidden_cells,
            model_mine_probabilities=model_mine_probabilities,
            remaining_mines=remaining_mines,
            prior_strength=prior_strength,
            max_constraint_nodes=max_constraint_nodes,
            max_total_search_nodes=max_total_search_nodes,
            min_outcome_probability=min_outcome_probability,
        )

    reused_context = inference_context is not None
    context = inference_context or build_constraint_context(
        constraints,
        hidden_cells=hidden_cells,
        model_mine_probabilities=model_mine_probabilities,
        remaining_mines=remaining_mines,
        prior_strength=prior_strength,
        max_search_nodes=max_constraint_nodes,
        max_total_search_nodes=max_total_search_nodes,
    )
    search_nodes = 0 if reused_context else context.search_nodes
    if (
        context.budget_exhausted
        or context.overflowed_boxes
        or not context.consistent
        or not context.complete
    ):
        return _invalid_evaluation(
            search_nodes,
            budget_exhausted=context.budget_exhausted,
        )

    conditioned = infer_safe_click_outcomes(
        context,
        candidate,
        hidden_neighbors,
    )
    if not conditioned.consistent:
        return _invalid_evaluation(search_nodes, budget_exhausted=False)
    conditioned_probabilities = conditioned.mine_probabilities
    outcome_distribution = conditioned.outcome_probabilities
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

        outcome_probabilities = (
            conditioned.mine_probabilities_by_outcome[mine_count]
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


def _evaluate_independent_safe_click(
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
    safe_constraints = [*constraints, (frozenset({candidate}), 0)]
    conditioned = infer_mine_probabilities(
        safe_constraints,
        hidden_cells=hidden_cells,
        model_mine_probabilities=model_mine_probabilities,
        remaining_mines=remaining_mines,
        prior_strength=prior_strength,
        max_search_nodes=max_constraint_nodes,
        max_total_search_nodes=max_total_search_nodes,
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
    outcome_distribution = poisson_binomial_distribution(
        [
            conditioned_probabilities[neighbor]
            for neighbor in sorted(
                hidden_neighbors,
                key=_coordinate_sort_key,
            )
        ]
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
    valid_mass = forced_mass = entropy_mass = zero_mass = 0.0
    for mine_count, probability in sorted(outcome_distribution.items()):
        if probability <= 0.0 or probability < min_outcome_probability:
            continue
        remaining_nodes = max_total_search_nodes - search_nodes
        if remaining_nodes <= 0:
            return _invalid_evaluation(
                search_nodes,
                budget_exhausted=True,
            )
        outcome = infer_mine_probabilities(
            [*safe_constraints, (hidden_neighbors, mine_count)],
            hidden_cells=hidden_cells,
            model_mine_probabilities=model_mine_probabilities,
            remaining_mines=remaining_mines,
            prior_strength=prior_strength,
            max_search_nodes=max_constraint_nodes,
            max_total_search_nodes=remaining_nodes,
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
        probabilities = _complete_probabilities(
            hidden_cells,
            model_mine_probabilities,
            outcome.mine_probabilities,
        )
        forced = _forced_cells(scored_cells, probabilities)
        valid_mass += probability
        forced_mass += probability * len(forced - baseline_forced)
        entropy_mass += probability * (
            baseline_entropy - _total_entropy(scored_cells, probabilities)
        )
        if mine_count == 0 and not has_known_mine_neighbor:
            zero_mass += probability
    if valid_mass <= 0.0:
        return _invalid_evaluation(
            search_nodes,
            budget_exhausted=False,
        )
    return LookaheadEvaluation(
        expected_forced_cells=forced_mass / valid_mass,
        expected_entropy_reduction=entropy_mass / valid_mass,
        zero_region_probability=zero_mass / valid_mass,
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


def _coordinate_sort_key(
    coordinate: Coordinate,
) -> tuple[int, int]:
    x, y = coordinate
    return y, x
