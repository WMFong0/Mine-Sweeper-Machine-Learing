from __future__ import annotations

import math
from dataclasses import dataclass

from minesweeper_ml.game import Coordinate


Constraint = tuple[frozenset[Coordinate], int]


@dataclass(frozen=True)
class ConstraintBox:
    cells: frozenset[Coordinate]
    constraints: tuple[Constraint, ...]


@dataclass(frozen=True)
class WeightedAssignment:
    mine_mask: int
    mine_count: int
    weight: float


@dataclass(frozen=True)
class BoxEnumeration:
    assignment_count: int
    search_nodes: int
    overflowed: bool
    variables: tuple[Coordinate, ...]
    assignments: tuple[WeightedAssignment, ...]
    weight_by_mine_count: dict[int, float]
    mine_weight_by_cell_and_count: dict[Coordinate, dict[int, float]]
    mine_probabilities: dict[Coordinate, float]


@dataclass(frozen=True)
class MineProbabilityInference:
    mine_probabilities: dict[Coordinate, float]
    consistent: bool
    globally_coupled: bool
    overflowed_boxes: int
    search_nodes: int
    budget_exhausted: bool = False


@dataclass(frozen=True)
class ConstraintInferenceContext:
    mine_probabilities: dict[Coordinate, float]
    consistent: bool
    globally_coupled: bool
    overflowed_boxes: int
    search_nodes: int
    budget_exhausted: bool
    complete: bool
    enumerations: tuple[BoxEnumeration, ...]
    hidden_cells: frozenset[Coordinate]
    constrained_cells: frozenset[Coordinate]
    model_mine_probabilities: dict[Coordinate, float]
    remaining_mines: int | None
    prior_strength: float


@dataclass(frozen=True)
class SafeClickOutcomeInference:
    mine_probabilities: dict[Coordinate, float]
    outcome_probabilities: dict[int, float]
    mine_probabilities_by_outcome: dict[int, dict[Coordinate, float]]
    consistent: bool


def build_constraint_boxes(constraints: list[Constraint]) -> list[ConstraintBox]:
    pending = [
        (set(cells), (cells, mine_count))
        for cells, mine_count in constraints
        if cells
    ]
    boxes = []

    while pending:
        cells, first_constraint = pending.pop(0)
        box_constraints = [first_constraint]
        changed = True
        while changed:
            changed = False
            remaining = []
            for candidate_cells, candidate_constraint in pending:
                if cells & candidate_cells:
                    cells.update(candidate_cells)
                    box_constraints.append(candidate_constraint)
                    changed = True
                else:
                    remaining.append((candidate_cells, candidate_constraint))
            pending = remaining

        boxes.append(
            ConstraintBox(
                cells=frozenset(cells),
                constraints=tuple(box_constraints),
            )
        )

    return sorted(boxes, key=lambda box: min((y, x) for x, y in box.cells))


def enumerate_constraint_box(
    box: ConstraintBox,
    mine_probabilities: dict[Coordinate, float],
    *,
    prior_strength: float,
    max_search_nodes: int,
) -> BoxEnumeration:
    degree = {cell: 0 for cell in box.cells}
    for cells, _ in box.constraints:
        for cell in cells:
            degree[cell] += 1
    variables = tuple(
        sorted(
            box.cells,
            key=lambda cell: (-degree[cell], cell[1], cell[0]),
        )
    )

    constraint_indexes = {cell: [] for cell in variables}
    targets = []
    assigned_mines = []
    unassigned_cells = []
    for index, (cells, mine_count) in enumerate(box.constraints):
        targets.append(mine_count)
        assigned_mines.append(0)
        unassigned_cells.append(len(cells))
        for cell in cells:
            constraint_indexes[cell].append(index)

    assignments = [0] * len(variables)
    search_nodes = 0
    overflowed = False
    weighted_masks: list[tuple[int, int, float]] = []

    def visit(position: int) -> None:
        nonlocal search_nodes, overflowed
        if search_nodes >= max_search_nodes:
            overflowed = True
            return
        search_nodes += 1

        if position == len(variables):
            mine_count = sum(assignments)
            weighted_masks.append(
                (
                    sum(
                        int(is_mine) << index
                        for index, is_mine in enumerate(assignments)
                    ),
                    mine_count,
                    prior_strength
                    * sum(
                        math.log(
                            mine_probabilities[cell]
                            if is_mine
                            else 1.0 - mine_probabilities[cell]
                        )
                        for cell, is_mine in zip(variables, assignments)
                    ),
                )
            )
            return

        cell = variables[position]
        touched_constraints = constraint_indexes[cell]
        for is_mine in (0, 1):
            assignments[position] = is_mine
            valid = True
            for constraint_index in touched_constraints:
                assigned_mines[constraint_index] += is_mine
                unassigned_cells[constraint_index] -= 1
                target = targets[constraint_index]
                if (
                    assigned_mines[constraint_index] > target
                    or assigned_mines[constraint_index]
                    + unassigned_cells[constraint_index]
                    < target
                ):
                    valid = False

            if valid:
                visit(position + 1)

            for constraint_index in touched_constraints:
                assigned_mines[constraint_index] -= is_mine
                unassigned_cells[constraint_index] += 1
            if overflowed:
                return

    visit(0)
    maximum_log_weight = max(
        (log_weight for _, _, log_weight in weighted_masks),
        default=0.0,
    )
    log_scale = (
        0.0
        if math.exp(maximum_log_weight) > 0.0
        else maximum_log_weight
    )
    stored_assignments = tuple(
        WeightedAssignment(
            mine_mask,
            mine_count,
            math.exp(log_weight - log_scale),
        )
        for mine_mask, mine_count, log_weight in weighted_masks
    )
    weight_by_mine_count: dict[int, float] = {}
    mine_weight_by_cell_and_count = {cell: {} for cell in variables}
    for assignment in stored_assignments:
        weight_by_mine_count[assignment.mine_count] = (
            weight_by_mine_count.get(assignment.mine_count, 0.0)
            + assignment.weight
        )
        mine_mask = assignment.mine_mask
        while mine_mask:
            bit = mine_mask & -mine_mask
            cell = variables[bit.bit_length() - 1]
            weights_by_count = mine_weight_by_cell_and_count[cell]
            weights_by_count[assignment.mine_count] = (
                weights_by_count.get(assignment.mine_count, 0.0)
                + assignment.weight
            )
            mine_mask ^= bit
    total_weight = sum(weight_by_mine_count.values())
    normalized_probabilities = (
        {
            cell: sum(weights.values()) / total_weight
            for cell, weights in mine_weight_by_cell_and_count.items()
        }
        if total_weight
        else {}
    )
    return BoxEnumeration(
        assignment_count=len(stored_assignments),
        search_nodes=search_nodes,
        overflowed=overflowed,
        variables=variables,
        assignments=stored_assignments,
        weight_by_mine_count=weight_by_mine_count,
        mine_weight_by_cell_and_count=mine_weight_by_cell_and_count,
        mine_probabilities=normalized_probabilities,
    )


def infer_mine_probabilities(
    constraints: list[Constraint],
    *,
    hidden_cells: set[Coordinate],
    model_mine_probabilities: dict[Coordinate, float],
    remaining_mines: int | None,
    prior_strength: float,
    max_search_nodes: int,
    max_total_search_nodes: int | None = None,
) -> MineProbabilityInference:
    context = build_constraint_context(
        constraints,
        hidden_cells=hidden_cells,
        model_mine_probabilities=model_mine_probabilities,
        remaining_mines=remaining_mines,
        prior_strength=prior_strength,
        max_search_nodes=max_search_nodes,
        max_total_search_nodes=max_total_search_nodes,
    )
    return MineProbabilityInference(
        mine_probabilities=context.mine_probabilities,
        consistent=context.consistent,
        globally_coupled=context.globally_coupled,
        overflowed_boxes=context.overflowed_boxes,
        search_nodes=context.search_nodes,
        budget_exhausted=context.budget_exhausted,
    )


def build_constraint_context(
    constraints: list[Constraint],
    *,
    hidden_cells: set[Coordinate],
    model_mine_probabilities: dict[Coordinate, float],
    remaining_mines: int | None,
    prior_strength: float,
    max_search_nodes: int,
    max_total_search_nodes: int | None = None,
) -> ConstraintInferenceContext:
    if max_total_search_nodes is not None and max_total_search_nodes <= 0:
        raise ValueError("max_total_search_nodes must be greater than zero.")

    if any(not cells and mine_count != 0 for cells, mine_count in constraints):
        return ConstraintInferenceContext(
            mine_probabilities={},
            consistent=False,
            globally_coupled=False,
            overflowed_boxes=0,
            search_nodes=0,
            budget_exhausted=False,
            complete=False,
            enumerations=(),
            hidden_cells=frozenset(hidden_cells),
            constrained_cells=frozenset(),
            model_mine_probabilities=dict(model_mine_probabilities),
            remaining_mines=remaining_mines,
            prior_strength=prior_strength,
        )

    boxes = build_constraint_boxes(constraints)
    enumerations = []
    local_probabilities: dict[Coordinate, float] = {}
    overflowed_boxes = 0
    search_nodes = 0
    budget_exhausted = False

    for box in boxes:
        remaining_total_nodes = (
            None
            if max_total_search_nodes is None
            else max_total_search_nodes - search_nodes
        )
        if remaining_total_nodes is not None and remaining_total_nodes <= 0:
            budget_exhausted = True
            break
        enumeration_node_limit = (
            max_search_nodes
            if remaining_total_nodes is None
            else min(max_search_nodes, remaining_total_nodes)
        )
        limited_by_total_budget = (
            remaining_total_nodes is not None
            and remaining_total_nodes <= max_search_nodes
        )
        enumeration = enumerate_constraint_box(
            box,
            model_mine_probabilities,
            prior_strength=prior_strength,
            max_search_nodes=enumeration_node_limit,
        )
        search_nodes += enumeration.search_nodes
        if enumeration.overflowed:
            if limited_by_total_budget:
                budget_exhausted = True
                break
            overflowed_boxes += 1
            continue
        if enumeration.assignment_count == 0:
            return ConstraintInferenceContext(
                mine_probabilities={},
                consistent=False,
                globally_coupled=False,
                overflowed_boxes=overflowed_boxes,
                search_nodes=search_nodes,
                budget_exhausted=False,
                complete=False,
                enumerations=tuple(enumerations),
                hidden_cells=frozenset(hidden_cells),
                constrained_cells=frozenset(
                    cell
                    for box in boxes
                    for cell in box.cells
                ),
                model_mine_probabilities=dict(model_mine_probabilities),
                remaining_mines=remaining_mines,
                prior_strength=prior_strength,
            )
        enumerations.append(enumeration)
        local_probabilities.update(enumeration.mine_probabilities)

    constrained_cells = frozenset(
        cell
        for box in boxes
        for cell in box.cells
    )
    context_options = {
        "enumerations": tuple(enumerations),
        "hidden_cells": frozenset(hidden_cells),
        "constrained_cells": constrained_cells,
        "model_mine_probabilities": dict(model_mine_probabilities),
        "remaining_mines": remaining_mines,
        "prior_strength": prior_strength,
    }

    if budget_exhausted:
        return ConstraintInferenceContext(
            mine_probabilities=local_probabilities,
            consistent=True,
            globally_coupled=False,
            overflowed_boxes=overflowed_boxes,
            search_nodes=search_nodes,
            budget_exhausted=True,
            complete=False,
            **context_options,
        )

    if overflowed_boxes:
        return ConstraintInferenceContext(
            mine_probabilities=local_probabilities,
            consistent=True,
            globally_coupled=False,
            overflowed_boxes=overflowed_boxes,
            search_nodes=search_nodes,
            budget_exhausted=False,
            complete=False,
            **context_options,
        )

    if remaining_mines is not None and (
        remaining_mines < 0 or remaining_mines > len(hidden_cells)
    ):
        return ConstraintInferenceContext(
            mine_probabilities={},
            consistent=False,
            globally_coupled=False,
            overflowed_boxes=0,
            search_nodes=search_nodes,
            budget_exhausted=False,
            complete=False,
            **context_options,
        )

    provisional = ConstraintInferenceContext(
        mine_probabilities={},
        consistent=True,
        globally_coupled=remaining_mines is not None,
        overflowed_boxes=0,
        search_nodes=search_nodes,
        budget_exhausted=False,
        complete=True,
        **context_options,
    )
    conditioned = _infer_conditioned(
        provisional,
        candidate=None,
        query_cells=frozenset(),
    )
    if not conditioned.consistent:
        return ConstraintInferenceContext(
            mine_probabilities={},
            consistent=False,
            globally_coupled=False,
            overflowed_boxes=0,
            search_nodes=search_nodes,
            budget_exhausted=False,
            complete=False,
            **context_options,
        )
    return ConstraintInferenceContext(
        mine_probabilities=(
            conditioned.mine_probabilities
            if remaining_mines is not None
            else local_probabilities
        ),
        consistent=True,
        globally_coupled=remaining_mines is not None,
        overflowed_boxes=0,
        search_nodes=search_nodes,
        budget_exhausted=False,
        complete=True,
        **context_options,
    )


def infer_safe_click_outcomes(
    context: ConstraintInferenceContext,
    candidate: Coordinate,
    query_cells: frozenset[Coordinate],
) -> SafeClickOutcomeInference:
    if (
        not context.consistent
        or not context.complete
        or candidate not in context.hidden_cells
    ):
        return SafeClickOutcomeInference({}, {}, {}, False)
    return _infer_conditioned(
        context,
        candidate=candidate,
        query_cells=query_cells & context.hidden_cells,
    )


def _infer_conditioned(
    context: ConstraintInferenceContext,
    *,
    candidate: Coordinate | None,
    query_cells: frozenset[Coordinate],
) -> SafeClickOutcomeInference:
    components = [
        _box_component(enumeration, query_cells, candidate)
        for enumeration in context.enumerations
    ]
    components.extend(
        _cell_component(
            cell,
            (
                0.5
                if context.remaining_mines is not None
                else context.model_mine_probabilities.get(cell, 0.5)
            ),
            context.prior_strength,
            cell in query_cells,
            cell == candidate,
        )
        for cell in sorted(
            context.hidden_cells - context.constrained_cells,
            key=lambda cell: (cell[1], cell[0]),
        )
    )
    prefixes = [{(0, 0): 1.0}]
    for distribution, _ in components:
        prefixes.append(
            _convolve_joint_distributions(prefixes[-1], distribution)
        )
    suffixes = [None] * (len(components) + 1)
    suffixes[-1] = {(0, 0): 1.0}
    for index in range(len(components) - 1, -1, -1):
        suffixes[index] = _convolve_joint_distributions(
            components[index][0],
            suffixes[index + 1],
        )

    outcome_weights: dict[int, float] = {}
    for (total_mines, query_mines), weight in prefixes[-1].items():
        if (
            context.remaining_mines is None
            or total_mines == context.remaining_mines
        ):
            outcome_weights[query_mines] = (
                outcome_weights.get(query_mines, 0.0) + weight
            )
    total_weight = sum(outcome_weights.values())
    if total_weight <= 0.0:
        return SafeClickOutcomeInference({}, {}, {}, False)

    mine_numerators = {
        outcome: {cell: 0.0 for cell in context.hidden_cells}
        for outcome in outcome_weights
    }
    for index, (_, local_mine_distributions) in enumerate(components):
        outside = _convolve_joint_distributions(
            prefixes[index],
            suffixes[index + 1],
        )
        for cell, local_mine_distribution in local_mine_distributions.items():
            for (
                total_mines,
                query_mines,
            ), weight in _convolve_joint_distributions(
                local_mine_distribution,
                outside,
            ).items():
                if (
                    query_mines in mine_numerators
                    and (
                        context.remaining_mines is None
                        or total_mines == context.remaining_mines
                    )
                ):
                    mine_numerators[query_mines][cell] += weight

    probabilities_by_outcome = {
        outcome: {
            cell: numerator / outcome_weights[outcome]
            for cell, numerator in numerators.items()
        }
        for outcome, numerators in mine_numerators.items()
    }
    return SafeClickOutcomeInference(
        mine_probabilities={
            cell: sum(
                mine_numerators[outcome][cell]
                for outcome in outcome_weights
            )
            / total_weight
            for cell in context.hidden_cells
        },
        outcome_probabilities={
            outcome: weight / total_weight
            for outcome, weight in outcome_weights.items()
            if weight > 0.0
        },
        mine_probabilities_by_outcome=probabilities_by_outcome,
        consistent=True,
    )


def _box_component(
    enumeration: BoxEnumeration,
    query_cells: frozenset[Coordinate],
    candidate: Coordinate | None,
) -> tuple[
    dict[tuple[int, int], float],
    dict[Coordinate, dict[tuple[int, int], float]],
]:
    variable_indexes = {
        cell: index
        for index, cell in enumerate(enumeration.variables)
    }
    query_mask = sum(
        1 << variable_indexes[cell]
        for cell in query_cells & variable_indexes.keys()
    )
    candidate_mask = (
        1 << variable_indexes[candidate]
        if candidate in variable_indexes
        else 0
    )
    distribution: dict[tuple[int, int], float] = {}
    mine_distributions = {
        cell: {}
        for cell in enumeration.variables
    }
    for assignment in enumeration.assignments:
        if assignment.mine_mask & candidate_mask:
            continue
        counts = (
            assignment.mine_count,
            (assignment.mine_mask & query_mask).bit_count(),
        )
        distribution[counts] = (
            distribution.get(counts, 0.0) + assignment.weight
        )
        mine_mask = assignment.mine_mask
        while mine_mask:
            bit = mine_mask & -mine_mask
            weights = mine_distributions[
                enumeration.variables[bit.bit_length() - 1]
            ]
            weights[counts] = weights.get(counts, 0.0) + assignment.weight
            mine_mask ^= bit
    return distribution, mine_distributions


def _cell_component(
    cell: Coordinate,
    mine_probability: float,
    prior_strength: float,
    queried: bool,
    forced_safe: bool,
) -> tuple[
    dict[tuple[int, int], float],
    dict[Coordinate, dict[tuple[int, int], float]],
]:
    safe_weight = (1.0 - mine_probability) ** prior_strength
    if forced_safe:
        return {(0, 0): safe_weight}, {cell: {}}
    mine_weight = mine_probability**prior_strength
    mine_counts = (1, int(queried))
    return (
        {
            (0, 0): safe_weight,
            mine_counts: mine_weight,
        },
        {cell: {mine_counts: mine_weight}},
    )


def _convolve_joint_distributions(
    first: dict[tuple[int, int], float],
    second: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    combined: dict[tuple[int, int], float] = {}
    for (
        first_mines,
        first_query_mines,
    ), first_weight in first.items():
        for (
            second_mines,
            second_query_mines,
        ), second_weight in second.items():
            counts = (
                first_mines + second_mines,
                first_query_mines + second_query_mines,
            )
            combined[counts] = (
                combined.get(counts, 0.0)
                + first_weight * second_weight
            )
    return combined
