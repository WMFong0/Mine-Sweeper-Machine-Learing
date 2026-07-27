from __future__ import annotations

from dataclasses import dataclass

from minesweeper_ml.constraints import (
    Constraint,
    ConstraintInferenceContext,
    build_constraint_context,
    count_legal_worlds,
    infer_safe_click_outcomes,
)
from minesweeper_ml.game import Coordinate


_CERTAINTY_EPSILON = 1e-12


@dataclass(frozen=True)
class EndgameEvaluation:
    move: Coordinate | None
    win_probability: float
    legal_worlds: int
    search_nodes: int
    evaluated_states: int
    budget_exhausted: bool
    valid: bool


class _BudgetExhausted(RuntimeError):
    pass


class _InvalidState(RuntimeError):
    pass


def solve_endgame(
    constraints: list[Constraint],
    *,
    hidden_cells: set[Coordinate],
    remaining_mines: int | None,
    width: int,
    height: int,
    max_worlds: int = 256,
    max_search_nodes: int = 100_000,
    max_constraint_nodes: int = 250_000,
    inference_context: ConstraintInferenceContext | None = None,
) -> EndgameEvaluation:
    if max_worlds <= 0:
        raise ValueError("max_worlds must be greater than zero.")
    if max_search_nodes <= 0:
        raise ValueError("max_search_nodes must be greater than zero.")
    if max_constraint_nodes <= 0:
        raise ValueError("max_constraint_nodes must be greater than zero.")
    if remaining_mines is None:
        return EndgameEvaluation(
            None, 0.0, 0, 0, 0, False, False
        )

    nodes = states = 0
    cache: dict[
        tuple[
            tuple[Coordinate, ...],
            int,
            tuple[Constraint, ...],
        ],
        tuple[float, Coordinate | None],
    ] = {}

    def consume(search_nodes: int, *, state: bool = True) -> None:
        nonlocal nodes, states
        states += int(state)
        nodes += max(1, search_nodes) if state else search_nodes
        if nodes > max_search_nodes:
            raise _BudgetExhausted

    def make_context(
        state_constraints: tuple[Constraint, ...],
        state_hidden: frozenset[Coordinate],
    ) -> ConstraintInferenceContext:
        remaining_budget = max_search_nodes - nodes
        if remaining_budget <= 0:
            raise _BudgetExhausted
        context = build_constraint_context(
            list(state_constraints),
            hidden_cells=set(state_hidden),
            model_mine_probabilities={
                cell: 0.5 for cell in state_hidden
            },
            remaining_mines=remaining_mines,
            prior_strength=0.0,
            max_search_nodes=max_constraint_nodes,
            max_total_search_nodes=remaining_budget,
        )
        consume(context.search_nodes)
        if context.budget_exhausted:
            raise _BudgetExhausted
        if (
            not context.consistent
            or not context.complete
            or context.overflowed_boxes
        ):
            raise _InvalidState
        return context

    def solve_state(
        state_constraints: tuple[Constraint, ...],
        state_hidden: frozenset[Coordinate],
        context: ConstraintInferenceContext | None = None,
    ) -> tuple[float, Coordinate | None]:
        key = (
            tuple(sorted(state_hidden, key=_coordinate_key)),
            remaining_mines,
            state_constraints,
        )
        if key in cache:
            return cache[key]
        current = context or make_context(state_constraints, state_hidden)
        worlds = count_legal_worlds(current)
        if not worlds or worlds > max_worlds:
            raise _InvalidState
        probabilities = current.mine_probabilities
        if (
            remaining_mines == 0
            or len(state_hidden) == remaining_mines
            or all(
                probability <= _CERTAINTY_EPSILON
                or probability >= 1.0 - _CERTAINTY_EPSILON
                for probability in probabilities.values()
            )
        ):
            cache[key] = (1.0, None)
            return cache[key]

        ordered_candidates = sorted(
            (
                candidate
                for candidate in state_hidden
                if probabilities[candidate]
                < 1.0 - _CERTAINTY_EPSILON
            ),
            key=_coordinate_key,
        )
        certain_safe = [
            candidate
            for candidate in ordered_candidates
            if probabilities[candidate] <= _CERTAINTY_EPSILON
        ]
        best_probability, best_move = -1.0, None
        for candidate in (
            certain_safe[:1] if certain_safe else ordered_candidates
        ):
            mine_probability = probabilities[candidate]
            hidden_neighbors = frozenset(
                neighbor
                for neighbor in _neighbors(
                    candidate,
                    width=width,
                    height=height,
                )
                if neighbor in state_hidden
            )
            remaining_work = max_search_nodes - nodes
            if remaining_work <= 0:
                raise _BudgetExhausted
            outcomes = infer_safe_click_outcomes(
                current,
                candidate,
                hidden_neighbors,
                max_work_units=remaining_work,
            )
            consume(outcomes.work_units, state=False)
            if outcomes.budget_exhausted:
                raise _BudgetExhausted
            if not outcomes.consistent:
                raise _InvalidState
            continuation = 0.0
            for clue_mines, outcome_probability in sorted(
                outcomes.outcome_probabilities.items()
            ):
                child_hidden = state_hidden - {candidate}
                child_constraints = _condition_safe_click(
                    state_constraints,
                    candidate,
                    hidden_neighbors,
                    clue_mines,
                    child_hidden,
                )
                child_probability, _ = solve_state(
                    child_constraints,
                    child_hidden,
                )
                continuation += outcome_probability * child_probability
            win_probability = (
                1.0 - mine_probability
            ) * continuation
            if win_probability > best_probability + _CERTAINTY_EPSILON:
                best_probability, best_move = win_probability, candidate

        if best_move is None:
            raise _InvalidState
        cache[key] = (best_probability, best_move)
        return cache[key]

    root_constraints = _normalize_constraints(
        constraints,
        frozenset(hidden_cells),
    )
    root_context = (
        inference_context
        if (
            inference_context is not None
            and inference_context.prior_strength == 0.0
        )
        else None
    )
    if root_context is None:
        try:
            root_context = make_context(
                root_constraints,
                frozenset(hidden_cells),
            )
        except (_BudgetExhausted, RecursionError):
            return EndgameEvaluation(
                None, 0.0, 0, nodes, states, True, False
            )
        except _InvalidState:
            return EndgameEvaluation(
                None, 0.0, 0, nodes, states, False, False
            )
    else:
        consume(0)

    legal_worlds = count_legal_worlds(root_context)
    if not legal_worlds or legal_worlds > max_worlds:
        return EndgameEvaluation(
            None,
            0.0,
            legal_worlds,
            nodes,
            states,
            False,
            False,
        )
    try:
        win_probability, move = solve_state(
            root_constraints,
            frozenset(hidden_cells),
            root_context,
        )
    except (_BudgetExhausted, RecursionError):
        return EndgameEvaluation(
            None,
            0.0,
            legal_worlds,
            nodes,
            states,
            True,
            False,
        )
    except _InvalidState:
        return EndgameEvaluation(
            None,
            0.0,
            legal_worlds,
            nodes,
            states,
            False,
            False,
        )
    return EndgameEvaluation(
        move,
        win_probability,
        legal_worlds,
        nodes,
        states,
        False,
        True,
    )


def _condition_safe_click(
    constraints: tuple[Constraint, ...],
    candidate: Coordinate,
    hidden_neighbors: frozenset[Coordinate],
    clue_mines: int,
    hidden_cells: frozenset[Coordinate],
) -> tuple[Constraint, ...]:
    return _normalize_constraints(
        [
            *((cells - {candidate}, target) for cells, target in constraints),
            (hidden_neighbors - {candidate}, clue_mines),
        ],
        hidden_cells,
    )


def _normalize_constraints(
    constraints: list[Constraint] | tuple[Constraint, ...],
    hidden_cells: frozenset[Coordinate],
) -> tuple[Constraint, ...]:
    return tuple(
        sorted(
            {
                (frozenset(cells & hidden_cells), int(target))
                for cells, target in constraints
                if cells & hidden_cells or target
            },
            key=lambda constraint: (
                tuple(sorted(constraint[0], key=_coordinate_key)),
                constraint[1],
            ),
        )
    )


def _neighbors(
    coordinate: Coordinate,
    *,
    width: int,
    height: int,
) -> tuple[Coordinate, ...]:
    x, y = coordinate
    return tuple(
        (neighbor_x, neighbor_y)
        for neighbor_y in range(max(0, y - 1), min(height, y + 2))
        for neighbor_x in range(max(0, x - 1), min(width, x + 2))
        if (neighbor_x, neighbor_y) != coordinate
    )


def _coordinate_key(coordinate: Coordinate) -> tuple[int, int]:
    return coordinate[1], coordinate[0]
