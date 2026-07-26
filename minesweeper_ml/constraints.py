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
class BoxEnumeration:
    assignment_count: int
    search_nodes: int
    overflowed: bool
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
    variables = sorted(
        box.cells,
        key=lambda cell: (-degree[cell], cell[1], cell[0]),
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
    assignment_count = 0
    search_nodes = 0
    overflowed = False
    log_weight_scale = 0.0
    has_raw_weight = False
    weight_by_mine_count: dict[int, float] = {}
    mine_weight_by_cell = {cell: 0.0 for cell in variables}
    mine_weight_by_cell_and_count = {cell: {} for cell in variables}

    def visit(position: int) -> None:
        nonlocal assignment_count, search_nodes, overflowed
        nonlocal log_weight_scale, has_raw_weight
        if search_nodes >= max_search_nodes:
            overflowed = True
            return
        search_nodes += 1

        if position == len(variables):
            assignment_count += 1
            mine_count = sum(assignments)
            log_weight = 0.0
            for cell, is_mine in zip(variables, assignments):
                probability = mine_probabilities[cell]
                log_weight += math.log(probability if is_mine else 1.0 - probability)
            scaled_log_weight = prior_strength * log_weight
            raw_weight = math.exp(scaled_log_weight)
            if raw_weight > 0.0:
                if not has_raw_weight and log_weight_scale != 0.0:
                    scale_factor = math.exp(log_weight_scale)
                    for count in weight_by_mine_count:
                        weight_by_mine_count[count] *= scale_factor
                    for weighted_cell in mine_weight_by_cell:
                        mine_weight_by_cell[weighted_cell] *= scale_factor
                        for count in mine_weight_by_cell_and_count[weighted_cell]:
                            mine_weight_by_cell_and_count[weighted_cell][count] *= (
                                scale_factor
                            )
                    log_weight_scale = 0.0
                has_raw_weight = True
                weight = raw_weight
            elif has_raw_weight:
                weight = 0.0
            elif log_weight_scale == 0.0:
                log_weight_scale = scaled_log_weight
                weight = 1.0
            elif scaled_log_weight > log_weight_scale:
                scale_factor = math.exp(log_weight_scale - scaled_log_weight)
                for count in weight_by_mine_count:
                    weight_by_mine_count[count] *= scale_factor
                for weighted_cell in mine_weight_by_cell:
                    mine_weight_by_cell[weighted_cell] *= scale_factor
                    for count in mine_weight_by_cell_and_count[weighted_cell]:
                        mine_weight_by_cell_and_count[weighted_cell][count] *= (
                            scale_factor
                        )
                log_weight_scale = scaled_log_weight
                weight = 1.0
            else:
                weight = math.exp(scaled_log_weight - log_weight_scale)
            weight_by_mine_count[mine_count] = (
                weight_by_mine_count.get(mine_count, 0.0) + weight
            )
            for cell, is_mine in zip(variables, assignments):
                if is_mine:
                    mine_weight_by_cell[cell] += weight
                    weights_by_count = mine_weight_by_cell_and_count[cell]
                    weights_by_count[mine_count] = (
                        weights_by_count.get(mine_count, 0.0) + weight
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
    total_weight = sum(weight_by_mine_count.values())
    normalized_probabilities = (
        {
            cell: mine_weight / total_weight
            for cell, mine_weight in mine_weight_by_cell.items()
        }
        if total_weight
        else {}
    )
    return BoxEnumeration(
        assignment_count=assignment_count,
        search_nodes=search_nodes,
        overflowed=overflowed,
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
    if max_total_search_nodes is not None and max_total_search_nodes <= 0:
        raise ValueError("max_total_search_nodes must be greater than zero.")

    if any(not cells and mine_count != 0 for cells, mine_count in constraints):
        return MineProbabilityInference(
            mine_probabilities={},
            consistent=False,
            globally_coupled=False,
            overflowed_boxes=0,
            search_nodes=0,
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
            return MineProbabilityInference(
                mine_probabilities={},
                consistent=False,
                globally_coupled=False,
                overflowed_boxes=overflowed_boxes,
                search_nodes=search_nodes,
            )
        enumerations.append(enumeration)
        local_probabilities.update(enumeration.mine_probabilities)

    if budget_exhausted:
        return MineProbabilityInference(
            mine_probabilities=local_probabilities,
            consistent=True,
            globally_coupled=False,
            overflowed_boxes=overflowed_boxes,
            search_nodes=search_nodes,
            budget_exhausted=True,
        )

    if overflowed_boxes or remaining_mines is None:
        return MineProbabilityInference(
            mine_probabilities=local_probabilities,
            consistent=True,
            globally_coupled=False,
            overflowed_boxes=overflowed_boxes,
            search_nodes=search_nodes,
        )

    if remaining_mines < 0 or remaining_mines > len(hidden_cells):
        return MineProbabilityInference(
            mine_probabilities={},
            consistent=False,
            globally_coupled=False,
            overflowed_boxes=0,
            search_nodes=search_nodes,
        )

    constrained_cells = {
        cell
        for box in boxes
        for cell in box.cells
    }
    unconstrained_cells = hidden_cells - constrained_cells
    unconstrained_distribution = {
        mine_count: float(math.comb(len(unconstrained_cells), mine_count))
        for mine_count in range(len(unconstrained_cells) + 1)
    }

    all_box_distribution = {0: 1.0}
    for enumeration in enumerations:
        all_box_distribution = _convolve_distributions(
            all_box_distribution,
            enumeration.weight_by_mine_count,
        )
    global_distribution = _convolve_distributions(
        all_box_distribution,
        unconstrained_distribution,
    )
    total_weight = global_distribution.get(remaining_mines, 0.0)
    if total_weight <= 0.0:
        return MineProbabilityInference(
            mine_probabilities={},
            consistent=False,
            globally_coupled=False,
            overflowed_boxes=0,
            search_nodes=search_nodes,
        )

    probabilities = {}
    for box_index, enumeration in enumerate(enumerations):
        outside_distribution = unconstrained_distribution
        for other_index, other_enumeration in enumerate(enumerations):
            if other_index == box_index:
                continue
            outside_distribution = _convolve_distributions(
                outside_distribution,
                other_enumeration.weight_by_mine_count,
            )

        for cell, mine_weights in enumeration.mine_weight_by_cell_and_count.items():
            numerator = sum(
                mine_weight
                * outside_distribution.get(remaining_mines - box_mines, 0.0)
                for box_mines, mine_weight in mine_weights.items()
            )
            probabilities[cell] = numerator / total_weight

    if unconstrained_cells:
        unconstrained_numerator = sum(
            (unconstrained_mines / len(unconstrained_cells))
            * unconstrained_weight
            * all_box_distribution.get(
                remaining_mines - unconstrained_mines,
                0.0,
            )
            for (
                unconstrained_mines,
                unconstrained_weight,
            ) in unconstrained_distribution.items()
        )
        unconstrained_probability = unconstrained_numerator / total_weight
        probabilities.update(
            {
                cell: unconstrained_probability
                for cell in unconstrained_cells
            }
        )

    return MineProbabilityInference(
        mine_probabilities=probabilities,
        consistent=True,
        globally_coupled=True,
        overflowed_boxes=0,
        search_nodes=search_nodes,
    )


def _convolve_distributions(
    first: dict[int, float],
    second: dict[int, float],
) -> dict[int, float]:
    combined: dict[int, float] = {}
    for first_count, first_weight in first.items():
        for second_count, second_weight in second.items():
            total_count = first_count + second_count
            combined[total_count] = (
                combined.get(total_count, 0.0)
                + first_weight * second_weight
            )
    return combined
