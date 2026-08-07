from __future__ import annotations

import copy
import random
from collections.abc import Callable

from minesweeper_ml.game import Cell, GameState, MinesweeperGame


PHASES = ("early", "middle", "late")


def encode_board_state(board_state: list[list[Cell]]) -> list[int]:
    encoded = []
    for row in board_state:
        for cell in row:
            if cell == "-":
                encoded.append(0)
            elif cell == "M":
                encoded.append(10)
            else:
                encoded.append(int(cell) + 1)
    return encoded


def encode_board_features(board_state: list[list[Cell]]) -> list[list[list[int]]]:
    if not board_state or not board_state[0]:
        raise ValueError("board_state must contain at least one cell.")

    width = len(board_state[0])
    encoded_board = []
    for row in board_state:
        if len(row) != width:
            raise ValueError("board_state rows must all have the same width.")
        encoded_row = []
        for cell in row:
            features = [0] * 10
            if cell in {"-", "F"}:
                features[0] = 1
            elif _is_revealed_clue(cell):
                features[int(cell) + 1] = 1
            else:
                raise ValueError(f"Unsupported visible cell value: {cell!r}")
            encoded_row.append(features)
        encoded_board.append(encoded_row)
    return encoded_board


def encode_ground_truth_map(ground_truth_map: list[list[Cell]]) -> list[int]:
    return [0 if cell == "M" else 1 for row in ground_truth_map for cell in row]


def frontier_mask(board_state: list[list[Cell]]) -> list[list[int]]:
    if not board_state or not board_state[0]:
        raise ValueError("board_state must contain at least one cell.")

    height = len(board_state)
    width = len(board_state[0])
    mask = [[0 for _ in range(width)] for _ in range(height)]

    for y, row in enumerate(board_state):
        if len(row) != width:
            raise ValueError("board_state rows must all have the same width.")
        for x, cell in enumerate(row):
            if cell != "-":
                continue
            if any(
                _is_revealed_clue(board_state[neighbor_y][neighbor_x])
                for neighbor_x, neighbor_y in _neighbors(x, y, width, height)
            ):
                mask[y][x] = 1

    return mask


def run_single_bot_game_and_collect_data(
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    *,
    seed: int | None = None,
    game_id: int = 0,
    mine_locations: list[tuple[int, int]] | None = None,
    bot_factory: Callable[[int, int, int], object] | None = None,
    max_samples_per_phase: int = 4,
    loss_weight: float = 2.0,
    close_probability_threshold: float = 0.01,
    close_probability_weight: float = 2.0,
    value_rollout_candidates: int = 2,
    value_rollout_max_moves: int = 200,
):
    rng = random.Random(seed)
    if max_samples_per_phase <= 0:
        raise ValueError("max_samples_per_phase must be greater than zero.")
    if loss_weight <= 0.0 or close_probability_weight <= 0.0:
        raise ValueError("trajectory weights must be greater than zero.")
    if close_probability_threshold < 0.0:
        raise ValueError(
            "close_probability_threshold must be non-negative."
        )
    if value_rollout_candidates < 0 or value_rollout_max_moves <= 0:
        raise ValueError("value rollout limits must be non-negative.")
    game = MinesweeperGame(
        width=width,
        height=height,
        mine_count=mine_count,
        mine_locations=mine_locations,
        seed=rng.randrange(2**63),
        first_safe=mine_locations is None,
    )
    bot = (bot_factory or _default_bot_factory)(width, height, mine_count)
    samples = []
    phase_counts = {phase: 0 for phase in PHASES}

    game.open_cell(0, 0)
    if game.state != GameState.IN_PROGRESS:
        return [], game.state

    while game.state == GameState.IN_PROGRESS:
        board_state = copy.deepcopy(game.visible_map)
        training_mask = frontier_mask(board_state)
        move = bot.get_next_move(copy.deepcopy(game.visible_map))
        if move is None:
            game.state = GameState.STOPPED
            break
        x, y = move
        if game.visible_map[y][x] != "-":
            game.state = GameState.STOPPED
            break

        strategy = getattr(bot, "last_move_strategy", None) or "policy"
        if (
            strategy != "deduction"
            and any(map(any, training_mask))
        ):
            phase = _game_phase(game)
            if phase_counts[phase] < max_samples_per_phase:
                decision = getattr(bot, "last_decision", None)
                eligible_candidates = tuple(
                    getattr(
                        decision,
                        "eligible_candidates",
                        ((x, y),),
                    )
                )
                value_targets, value_mask = evaluate_candidate_values(
                    game,
                    eligible_candidates or ((x, y),),
                    bot_factory=bot_factory or _default_bot_factory,
                    max_candidates=value_rollout_candidates,
                    max_moves=value_rollout_max_moves,
                )
                samples.append(
                    {
                        "game_id": game_id,
                        "phase": phase,
                        "policy": strategy,
                        "board_state": board_state,
                        "training_mask": training_mask,
                        "chosen_move": (x, y),
                        "mine_probabilities": (
                            tuple(
                                getattr(
                                    decision,
                                    "mine_probabilities",
                                    (),
                                )
                            )
                            if decision is not None
                            else ()
                        ),
                        "eligible_candidates": (
                            eligible_candidates
                            if decision is not None
                            else ((x, y),)
                        ),
                        "probability_gap": (
                            decision.probability_gap
                            if decision is not None
                            else None
                        ),
                        "value_targets": value_targets,
                        "value_mask": value_mask,
                    }
                )
                phase_counts[phase] += 1

        game.open_cell(x, y)

    ground_truth_map = copy.deepcopy(game.mine_map)
    layout_id = tuple(sorted(game.mine_locations))
    for sample in samples:
        x, y = sample["chosen_move"]
        sample.update(
            {
                "layout_id": layout_id,
                "ground_truth_map": copy.deepcopy(ground_truth_map),
                "is_safe": ground_truth_map[y][x] != "M",
                "game_result": game.state,
                "trajectory_weight": (
                    (loss_weight if game.state == GameState.LOST else 1.0)
                    * (
                        close_probability_weight
                        if sample["probability_gap"] is not None
                        and sample["probability_gap"]
                        <= close_probability_threshold
                        else 1.0
                    )
                ),
            }
        )
    return samples, game.state


def generate_training_data(
    num_games: int,
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    *,
    seed: int | None = None,
    verbose: bool = False,
    bot_factory: Callable[[int, int, int], object] | None = None,
    max_samples_per_phase: int = 4,
    loss_weight: float = 2.0,
    close_probability_threshold: float = 0.01,
    close_probability_weight: float = 2.0,
    value_rollout_candidates: int = 2,
    value_rollout_max_moves: int = 200,
):
    if num_games <= 0:
        raise ValueError("num_games must be greater than zero.")

    rng = random.Random(seed)
    full_dataset = []
    if verbose:
        print(f"Generating data for {num_games} games on {width}x{height} board...")

    for index in range(num_games):
        if verbose and (index + 1) % 100 == 0:
            print(f"  Completed {index + 1}/{num_games} games")
        game_steps, _ = run_single_bot_game_and_collect_data(
            width,
            height,
            mine_count,
            seed=rng.randrange(2**63),
            game_id=index,
            bot_factory=bot_factory,
            max_samples_per_phase=max_samples_per_phase,
            loss_weight=loss_weight,
            close_probability_threshold=close_probability_threshold,
            close_probability_weight=close_probability_weight,
            value_rollout_candidates=value_rollout_candidates,
            value_rollout_max_moves=value_rollout_max_moves,
        )
        full_dataset.extend(game_steps)

    return full_dataset


def dataset_to_arrays(training_dataset):
    import numpy as np

    if not training_dataset:
        raise ValueError("training_dataset must contain at least one data point.")

    feature_values = [
        encode_board_features(data_point["board_state"])
        for data_point in training_dataset
    ]
    label_values = [
        [
            [[0 if cell == "M" else 1] for cell in row]
            for row in data_point["ground_truth_map"]
        ]
        for data_point in training_dataset
    ]
    mask_values = [
        [[[value] for value in row] for row in data_point["training_mask"]]
        for data_point in training_dataset
    ]
    return (
        np.asarray(feature_values, dtype=np.float32),
        np.asarray(label_values, dtype=np.float32),
        np.asarray(mask_values, dtype=np.float32),
    )


def split_dataset_by_game(
    dataset,
    test_fraction: float = 0.2,
    seed: int = 42,
):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one.")

    game_layouts = {}
    layout_games = {}
    for sample in dataset:
        game_id = sample["game_id"]
        layout_id = sample.get("layout_id", ("game_id", game_id))
        if game_id in game_layouts and game_layouts[game_id] != layout_id:
            raise ValueError("one game_id cannot contain multiple mine layouts.")
        game_layouts[game_id] = layout_id
        layout_games.setdefault(layout_id, set()).add(game_id)

    if len(game_layouts) < 2:
        raise ValueError("dataset must contain at least two unique games.")
    if len(layout_games) < 2:
        raise ValueError("dataset must contain at least two unique mine layouts.")

    rng = random.Random(seed)
    layout_ids = sorted(layout_games, key=repr)
    rng.shuffle(layout_ids)
    test_count = max(1, min(len(layout_ids) - 1, round(len(layout_ids) * test_fraction)))
    test_ids = {
        game_id
        for layout_id in layout_ids[:test_count]
        for game_id in layout_games[layout_id]
    }
    train_dataset = [sample for sample in dataset if sample["game_id"] not in test_ids]
    test_dataset = [sample for sample in dataset if sample["game_id"] in test_ids]
    return train_dataset, test_dataset


def calculate_class_weights(labels, masks) -> dict[int, float]:
    import numpy as np

    if labels.shape != masks.shape:
        raise ValueError("labels and masks must have matching shapes.")

    active = masks[..., 0] > 0
    active_labels = labels[..., 0][active]
    mine_count = int(np.count_nonzero(active_labels < 0.5))
    safe_count = int(np.count_nonzero(active_labels >= 0.5))
    if mine_count == 0 or safe_count == 0:
        raise ValueError("masked training data must contain both mined and safe cells.")

    active_count = mine_count + safe_count
    return {
        0: active_count / (2 * mine_count),
        1: active_count / (2 * safe_count),
    }


def build_sample_weights(labels, masks, class_weights):
    import numpy as np

    if labels.shape != masks.shape:
        raise ValueError("labels and masks must have matching shapes.")

    cell_weights = np.where(
        labels[..., 0] >= 0.5,
        class_weights[1],
        class_weights[0],
    )
    return np.asarray(cell_weights * masks[..., 0], dtype=np.float32)


def _neighbors(x: int, y: int, width: int, height: int):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_x = x + dx
            neighbor_y = y + dy
            if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                yield neighbor_x, neighbor_y


def _is_revealed_clue(cell: Cell) -> bool:
    return isinstance(cell, int) and 0 <= cell <= 8


def _game_phase(game: MinesweeperGame) -> str:
    safe_cell_count = game.width * game.height - game.mine_count
    progress = 1.0 - (game.remaining_cells / safe_cell_count)
    if progress < 1 / 3:
        return "early"
    if progress < 2 / 3:
        return "middle"
    return "late"


class _UniformSafetyModel:
    def predict(self, board, verbose: int = 0):
        import numpy as np

        return np.full((*board.shape[:3], 1), 0.5, dtype=np.float32)


def _default_bot_factory(width: int, height: int, mine_count: int):
    from minesweeper_ml.bot.bots import MLMinesweeperBot

    return MLMinesweeperBot(
        width,
        height,
        _UniformSafetyModel(),
        mine_count=mine_count,
        endgame_max_worlds=0,
        cnn_input=True,
    )


def multitask_dataset_to_arrays(training_dataset):
    import numpy as np

    features, safety_targets, safety_masks = dataset_to_arrays(
        training_dataset
    )
    height, width = safety_targets.shape[1:3]
    value_targets = np.asarray(
        [
            [
                [[value] for value in row]
                for row in sample.get(
                    "value_targets",
                    [[0] * width for _ in range(height)],
                )
            ]
            for sample in training_dataset
        ],
        dtype=np.float32,
    )
    value_masks = np.asarray(
        [
            [
                [[value] for value in row]
                for row in sample.get(
                    "value_mask",
                    [[0] * width for _ in range(height)],
                )
            ]
            for sample in training_dataset
        ],
        dtype=np.float32,
    )
    return (
        features,
        {
            "safety": safety_targets,
            "value": value_targets,
        },
        {
            "safety": safety_masks,
            "value": value_masks,
        },
        np.asarray(
            [
                sample.get("trajectory_weight", 1.0)
                for sample in training_dataset
            ],
            dtype=np.float32,
        ),
    )


def evaluate_candidate_values(
    game: MinesweeperGame,
    candidates,
    *,
    bot_factory: Callable[[int, int, int], object],
    max_candidates: int,
    max_moves: int,
):
    if max_candidates < 0 or max_moves <= 0:
        raise ValueError("value rollout limits must be non-negative.")
    targets = [
        [0 for _ in range(game.width)]
        for _ in range(game.height)
    ]
    mask = [
        [0 for _ in range(game.width)]
        for _ in range(game.height)
    ]
    for x, y in tuple(dict.fromkeys(candidates))[:max_candidates]:
        if (
            not 0 <= x < game.width
            or not 0 <= y < game.height
            or game.visible_map[y][x] != "-"
            or game.mine_map[y][x] == "M"
        ):
            continue
        rollout = game.clone()
        rollout.open_cell(x, y)
        rollout_bot = bot_factory(
            game.width,
            game.height,
            game.mine_count,
        )
        if hasattr(rollout_bot, "use_candidate_value"):
            rollout_bot.use_candidate_value = False
        if hasattr(rollout_bot, "endgame_max_worlds"):
            rollout_bot.endgame_max_worlds = 0
        for _ in range(max_moves):
            if rollout.state != GameState.IN_PROGRESS:
                break
            move = rollout_bot.get_next_move(
                copy.deepcopy(rollout.visible_map)
            )
            if (
                move is None
                or rollout.visible_map[move[1]][move[0]] != "-"
            ):
                rollout.state = GameState.STOPPED
                break
            rollout.open_cell(*move)
        targets[y][x] = int(rollout.state == GameState.WON)
        mask[y][x] = 1
    return targets, mask
