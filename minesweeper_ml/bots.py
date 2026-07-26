from __future__ import annotations

from collections import deque
from typing import Protocol

from minesweeper_ml.constraints import Constraint, infer_mine_probabilities
from minesweeper_ml.data import encode_board_features, encode_board_state
from minesweeper_ml.game import Cell, Coordinate
from minesweeper_ml.lookahead import evaluate_safe_click


class PredictionModel(Protocol):
    def predict(self, board, verbose: int = 0):
        ...


class RuleBasedMinesweeperBot:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.deduced_safe_moves: deque[Coordinate] = deque()
        self.deduced_mines: set[Coordinate] = set()

    def get_next_move(self, visible_map: list[list[Cell]]) -> Coordinate | None:
        rule_move = self._get_rule_move(visible_map)
        if rule_move is not None:
            return rule_move

        return self._lowest_risk_move(visible_map)

    def _neighbors(self, x: int, y: int) -> list[Coordinate]:
        neighbors: list[Coordinate] = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def _available_moves(
        self,
        visible_map: list[list[Cell]],
        excluded: set[Coordinate] | None = None,
    ) -> list[Coordinate]:
        excluded = excluded or set()
        moves = []
        for y in range(self.height):
            for x in range(self.width):
                if visible_map[y][x] == "-" and (x, y) not in excluded:
                    moves.append((x, y))
        return moves

    def _get_rule_move(self, visible_map: list[list[Cell]]) -> Coordinate | None:
        safe_moves, deduced_mines = self._infer_safe_moves_and_mines(visible_map)
        self.deduced_mines = deduced_mines
        self.deduced_safe_moves = deque(sorted(safe_moves, key=_move_sort_key))

        while self.deduced_safe_moves:
            next_x, next_y = self.deduced_safe_moves.popleft()
            if visible_map[next_y][next_x] == "-" and (next_x, next_y) not in self.deduced_mines:
                return next_x, next_y

        return None

    def _lowest_risk_move(self, visible_map: list[list[Cell]]) -> Coordinate | None:
        safe_moves, deduced_mines = self._infer_safe_moves_and_mines(visible_map)
        self.deduced_mines = deduced_mines

        if safe_moves:
            return sorted(safe_moves, key=_move_sort_key)[0]

        available_moves = set(self._available_moves(visible_map, deduced_mines))
        if not available_moves:
            return None

        clue_risks = self._clue_mine_risks(
            visible_map,
            deduced_mines,
            safe_moves,
        )

        return min(
            available_moves,
            key=lambda move: (
                clue_risks.get(move, 1.0),
                move[1],
                move[0],
            ),
        )

    def _clue_mine_risks(
        self,
        visible_map: list[list[Cell]],
        deduced_mines: set[Coordinate],
        safe_moves: set[Coordinate],
    ) -> dict[Coordinate, float]:
        risks: dict[Coordinate, float] = {}
        for cells, mine_count in self._number_constraints(
            visible_map,
            deduced_mines,
            safe_moves,
        ):
            if not cells:
                continue
            local_risk = max(0.0, min(1.0, mine_count / len(cells)))
            for move in cells:
                risks[move] = max(risks.get(move, 0.0), local_risk)
        return risks

    def _infer_safe_moves_and_mines(
        self,
        visible_map: list[list[Cell]],
    ) -> tuple[set[Coordinate], set[Coordinate]]:
        safe_moves: set[Coordinate] = set()
        deduced_mines: set[Coordinate] = set()

        changed = True
        while changed:
            changed = False
            before_safe_count = len(safe_moves)
            before_mine_count = len(deduced_mines)
            constraints = self._number_constraints(visible_map, deduced_mines, safe_moves)

            for cells, mine_count in constraints:
                if mine_count == 0:
                    safe_moves.update(cells)
                elif mine_count == len(cells):
                    deduced_mines.update(cells)

            for cells_a, mine_count_a in constraints:
                for cells_b, mine_count_b in constraints:
                    if not cells_a < cells_b:
                        continue

                    difference_cells = cells_b - cells_a
                    difference_mines = mine_count_b - mine_count_a
                    if difference_mines == 0:
                        safe_moves.update(difference_cells)
                    elif difference_mines == len(difference_cells):
                        deduced_mines.update(difference_cells)

            safe_moves.difference_update(deduced_mines)
            changed = len(safe_moves) != before_safe_count or len(deduced_mines) != before_mine_count

        return safe_moves, deduced_mines

    def _number_constraints(
        self,
        visible_map: list[list[Cell]],
        deduced_mines: set[Coordinate],
        safe_moves: set[Coordinate],
    ) -> list[tuple[frozenset[Coordinate], int]]:
        constraints = []

        for y in range(self.height):
            for x in range(self.width):
                cell_value = visible_map[y][x]
                if not (isinstance(cell_value, int) and cell_value > 0):
                    continue

                hidden_neighbors = set()
                known_mine_neighbors = 0
                for neighbor_x, neighbor_y in self._neighbors(x, y):
                    if visible_map[neighbor_y][neighbor_x] != "-":
                        continue
                    if (neighbor_x, neighbor_y) in deduced_mines:
                        known_mine_neighbors += 1
                    elif (neighbor_x, neighbor_y) not in safe_moves:
                        hidden_neighbors.add((neighbor_x, neighbor_y))

                remaining_mines = cell_value - known_mine_neighbors
                if hidden_neighbors and remaining_mines >= 0:
                    constraints.append((frozenset(hidden_neighbors), remaining_mines))

        return constraints


class MLMinesweeperBot(RuleBasedMinesweeperBot):
    def __init__(
        self,
        width: int,
        height: int,
        ml_model: PredictionModel,
        *,
        mine_count: int | None = None,
        max_constraint_nodes: int = 250_000,
        model_prior_strength: float = 0.75,
        model_weight: float = 0.6,
        tie_margin: float = 0.0025,
        lookahead_max_candidates: int = 4,
        lookahead_max_nodes: int = 100_000,
        lookahead_min_outcome_probability: float = 1e-6,
        outcome_correlation_strength: float = 1.0,
        cnn_input: bool = True,
    ):
        super().__init__(width, height)
        if not 0.0 <= model_weight <= 1.0:
            raise ValueError("model_weight must be between zero and one.")
        if tie_margin < 0.0:
            raise ValueError("tie_margin must be non-negative.")
        if mine_count is not None and not 0 <= mine_count <= width * height:
            raise ValueError("mine_count must fit within the board.")
        if max_constraint_nodes <= 0:
            raise ValueError("max_constraint_nodes must be greater than zero.")
        if model_prior_strength <= 0.0:
            raise ValueError("model_prior_strength must be greater than zero.")
        if lookahead_max_candidates < 0:
            raise ValueError(
                "lookahead_max_candidates must be non-negative."
            )
        if lookahead_max_nodes <= 0:
            raise ValueError("lookahead_max_nodes must be greater than zero.")
        if not 0.0 <= lookahead_min_outcome_probability <= 1.0:
            raise ValueError(
                "lookahead_min_outcome_probability must be between zero and one."
            )
        if not 0.0 <= outcome_correlation_strength <= 1.0:
            raise ValueError(
                "outcome_correlation_strength must be between zero and one."
            )
        self.ml_model = ml_model
        self.mine_count = mine_count
        self.max_constraint_nodes = max_constraint_nodes
        self.model_prior_strength = model_prior_strength
        self.model_weight = model_weight
        self.tie_margin = tie_margin
        self.lookahead_max_candidates = lookahead_max_candidates
        self.lookahead_max_nodes = lookahead_max_nodes
        self.lookahead_min_outcome_probability = (
            lookahead_min_outcome_probability
        )
        self.outcome_correlation_strength = (
            outcome_correlation_strength
        )
        self.cnn_input = cnn_input
        self.last_move_strategy: str | None = None
        self.strategy_stats = {
            "deduction": 0,
            "box": 0,
            "unconstrained": 0,
            "fallback": 0,
            "constraint_search_nodes": 0,
            "constraint_overflows": 0,
            "lookahead_decisions": 0,
            "lookahead_search_nodes": 0,
            "lookahead_budget_exhaustions": 0,
        }

    def get_next_move(self, visible_map: list[list[Cell]]) -> Coordinate | None:
        import numpy as np

        rule_move = self._get_rule_move(visible_map)
        if rule_move is not None:
            return self._record_move(rule_move, "deduction")

        predictions = np.asarray(
            self.ml_model.predict(self.preprocess_board_state(visible_map), verbose=0)
        ).reshape(-1)
        deduced_mines = set(self.deduced_mines)
        hidden_cells = set(self._available_moves(visible_map, deduced_mines))
        constraints = self._number_constraints(
            visible_map,
            deduced_mines,
            set(),
        )
        constrained_cells = {
            cell
            for cells, _ in constraints
            for cell in cells
        }
        model_mine_probabilities = {
            cell: max(
                0.02,
                min(
                    0.98,
                    1.0 - float(predictions[cell[1] * self.width + cell[0]]),
                ),
            )
            for cell in hidden_cells
        }
        remaining_mines = (
            None
            if self.mine_count is None
            else self.mine_count - len(deduced_mines)
        )
        inference = infer_mine_probabilities(
            constraints,
            hidden_cells=hidden_cells,
            model_mine_probabilities=model_mine_probabilities,
            remaining_mines=remaining_mines,
            prior_strength=self.model_prior_strength,
            max_search_nodes=self.max_constraint_nodes,
        )
        self.strategy_stats["constraint_search_nodes"] += inference.search_nodes
        self.strategy_stats["constraint_overflows"] += inference.overflowed_boxes

        posterior_candidates: dict[Coordinate, float] = {}
        if inference.consistent and inference.mine_probabilities:
            certain_mines = {
                move
                for move, mine_probability in inference.mine_probabilities.items()
                if mine_probability >= 1.0 - 1e-12
            }
            self.deduced_mines.update(certain_mines)
            posterior_candidates = {
                move: mine_probability
                for move, mine_probability in inference.mine_probabilities.items()
                if move not in certain_mines
            }
            certain_safe = {
                move: mine_probability
                for move, mine_probability in posterior_candidates.items()
                if mine_probability <= 1e-12
            }
            if certain_safe:
                move = self._lowest_probability_move(
                    certain_safe,
                    visible_map,
                    self.deduced_mines,
                )
                return self._record_move(move, "deduction")
            if posterior_candidates and inference.overflowed_boxes == 0:
                move = self._select_posterior_move(
                    posterior_candidates,
                    visible_map=visible_map,
                    deduced_mines=self.deduced_mines,
                    constraints=constraints,
                    hidden_cells=hidden_cells,
                    model_mine_probabilities=model_mine_probabilities,
                    remaining_mines=remaining_mines,
                )
                strategy = "box" if move in constrained_cells else "unconstrained"
                return self._record_move(move, strategy)

        clue_risks = self._clue_mine_risks(
            visible_map,
            self.deduced_mines,
            set(),
        )

        neighbor_candidates = self._prediction_candidates(
            visible_map,
            predictions,
            self.deduced_mines,
            clue_risks,
        )
        if inference.overflowed_boxes and posterior_candidates:
            candidates_by_move = {}
            for candidate in neighbor_candidates:
                _, _, x, y = candidate
                candidates_by_move[(x, y)] = candidate
            for move, mine_probability in posterior_candidates.items():
                x, y = move
                safety = 1.0 - mine_probability
                candidates_by_move[move] = (safety, safety, x, y)
            move = self._best_candidate(
                list(candidates_by_move.values()),
                visible_map,
                self.deduced_mines,
            )
            strategy = "box" if move in posterior_candidates else "fallback"
            return self._record_move(move, strategy)

        if neighbor_candidates:
            move = self._best_candidate(
                neighbor_candidates,
                visible_map,
                self.deduced_mines,
            )
            return self._record_move(move, "fallback")

        move = self._lowest_risk_move(visible_map)
        if move is None:
            self.last_move_strategy = None
            return None
        return self._record_move(move, "fallback")

    def preprocess_board_state(self, visible_map: list[list[Cell]]):
        import numpy as np

        if self.cnn_input:
            return np.asarray([encode_board_features(visible_map)], dtype=np.float32)
        return np.asarray([encode_board_state(visible_map)], dtype=np.float32)

    def _prediction_candidates(
        self,
        visible_map: list[list[Cell]],
        predictions,
        deduced_mines: set[Coordinate],
        clue_risks: dict[Coordinate, float],
    ) -> list[tuple[float, float, int, int]]:
        candidates = []
        for index, prediction in enumerate(predictions):
            x = index % self.width
            y = index // self.width
            move = (x, y)

            if visible_map[y][x] != "-":
                continue
            if move in deduced_mines:
                continue
            if not self._has_revealed_neighbor(x, y, visible_map):
                continue

            model_safety = float(prediction)
            clue_safety = 1.0 - clue_risks.get(move, 1.0)
            combined_safety = (
                self.model_weight * model_safety
                + (1.0 - self.model_weight) * clue_safety
            )
            candidates.append((combined_safety, model_safety, x, y))
        return candidates

    def _has_revealed_neighbor(self, x: int, y: int, visible_map: list[list[Cell]]) -> bool:
        return any(
            isinstance(visible_map[neighbor_y][neighbor_x], int)
            for neighbor_x, neighbor_y in self._neighbors(x, y)
        )

    def _best_candidate(
        self,
        candidates: list[tuple[float, float, int, int]],
        visible_map: list[list[Cell]],
        deduced_mines: set[Coordinate],
    ) -> Coordinate:
        best_score = max(candidate[0] for candidate in candidates)
        near_best = [
            candidate
            for candidate in candidates
            if best_score - candidate[0] <= self.tie_margin
        ]
        _, _, x, y = min(
            near_best,
            key=lambda candidate: (
                -self._information_gain(
                    (candidate[2], candidate[3]),
                    visible_map,
                    deduced_mines,
                ),
                candidate[3],
                candidate[2],
            ),
        )
        return x, y

    def _lowest_probability_move(
        self,
        mine_probabilities: dict[Coordinate, float],
        visible_map: list[list[Cell]],
        deduced_mines: set[Coordinate],
    ) -> Coordinate:
        lowest_probability = min(mine_probabilities.values())
        hidden_count = len(
            self._available_moves(visible_map, deduced_mines)
        )
        effective_margin = self._effective_tie_margin(hidden_count)
        near_best = [
            move
            for move, mine_probability in mine_probabilities.items()
            if mine_probability - lowest_probability <= effective_margin
        ]
        return min(
            near_best,
            key=lambda move: (
                -self._information_gain(move, visible_map, deduced_mines),
                move[1],
                move[0],
            ),
        )

    def _select_posterior_move(
        self,
        mine_probabilities: dict[Coordinate, float],
        *,
        visible_map: list[list[Cell]],
        deduced_mines: set[Coordinate],
        constraints: list[Constraint],
        hidden_cells: set[Coordinate],
        model_mine_probabilities: dict[Coordinate, float],
        remaining_mines: int | None,
    ) -> Coordinate:
        lowest_probability = min(mine_probabilities.values())
        unresolved_hidden = hidden_cells - deduced_mines
        effective_margin = self._effective_tie_margin(
            len(unresolved_hidden)
        )
        eligible_candidates = sorted(
            (
                (mine_probability, move)
                for move, mine_probability in mine_probabilities.items()
                if mine_probability - lowest_probability <= effective_margin
            ),
            key=lambda item: (
                item[0],
                item[1][1],
                item[1][0],
            ),
        )
        if (
            self.lookahead_max_candidates == 0
            or len(eligible_candidates) <= 1
        ):
            return self._lowest_probability_move(
                mine_probabilities,
                visible_map,
                deduced_mines,
            )

        shortlisted_moves = [
            move
            for _, move in eligible_candidates[
                : self.lookahead_max_candidates
            ]
        ]
        if len(shortlisted_moves) == 1:
            return shortlisted_moves[0]

        evaluations = {}
        remaining_node_budget = self.lookahead_max_nodes
        budget_exhausted = False
        has_valid_evaluation = False

        for candidate_index, move in enumerate(shortlisted_moves):
            if remaining_node_budget <= 0:
                budget_exhausted = True
                break

            hidden_neighbors = frozenset(
                neighbor
                for neighbor in self._neighbors(*move)
                if visible_map[neighbor[1]][neighbor[0]] == "-"
                and neighbor not in deduced_mines
            )
            has_known_mine_neighbor = any(
                neighbor in deduced_mines
                for neighbor in self._neighbors(*move)
            )
            evaluation = evaluate_safe_click(
                move,
                hidden_neighbors=hidden_neighbors,
                has_known_mine_neighbor=has_known_mine_neighbor,
                constraints=constraints,
                hidden_cells=hidden_cells,
                model_mine_probabilities=model_mine_probabilities,
                remaining_mines=remaining_mines,
                prior_strength=self.model_prior_strength,
                max_constraint_nodes=self.max_constraint_nodes,
                max_total_search_nodes=remaining_node_budget,
                min_outcome_probability=(
                    self.lookahead_min_outcome_probability
                ),
                outcome_correlation_strength=(
                    self.outcome_correlation_strength
                ),
            )
            evaluations[move] = evaluation
            remaining_node_budget -= evaluation.search_nodes
            self.strategy_stats[
                "lookahead_search_nodes"
            ] += evaluation.search_nodes
            has_valid_evaluation = (
                has_valid_evaluation or evaluation.valid
            )
            if evaluation.budget_exhausted:
                budget_exhausted = True
                break
            if (
                remaining_node_budget <= 0
                and candidate_index + 1 < len(shortlisted_moves)
            ):
                budget_exhausted = True
                break

        if budget_exhausted:
            self.strategy_stats["lookahead_budget_exhaustions"] += 1
        if has_valid_evaluation:
            self.strategy_stats["lookahead_decisions"] += 1

        def selection_key(move: Coordinate):
            evaluation = evaluations.get(move)
            expected_forced_cells = (
                evaluation.expected_forced_cells
                if evaluation is not None and evaluation.valid
                else 0.0
            )
            expected_entropy_reduction = (
                evaluation.expected_entropy_reduction
                if evaluation is not None and evaluation.valid
                else 0.0
            )
            zero_region_probability = (
                evaluation.zero_region_probability
                if evaluation is not None and evaluation.valid
                else 0.0
            )
            return (
                -expected_forced_cells,
                -expected_entropy_reduction,
                -zero_region_probability,
                -self._information_gain(
                    move,
                    visible_map,
                    deduced_mines,
                ),
                move[1],
                move[0],
            )

        return min(shortlisted_moves, key=selection_key)

    def _effective_tie_margin(self, hidden_count: int) -> float:
        hidden_fraction = hidden_count / (self.width * self.height)
        return self.tie_margin * max(
            0.25,
            min(1.0, hidden_fraction),
        )

    def _record_move(self, move: Coordinate, strategy: str) -> Coordinate:
        self.last_move_strategy = strategy
        self.strategy_stats[strategy] += 1
        return move

    def _information_gain(
        self,
        move: Coordinate,
        visible_map: list[list[Cell]],
        deduced_mines: set[Coordinate],
    ) -> int:
        x, y = move
        return sum(
            visible_map[neighbor_y][neighbor_x] == "-"
            and (neighbor_x, neighbor_y) not in deduced_mines
            for neighbor_x, neighbor_y in self._neighbors(x, y)
        )


def _move_sort_key(move: Coordinate) -> tuple[int, int]:
    x, y = move
    return y, x
