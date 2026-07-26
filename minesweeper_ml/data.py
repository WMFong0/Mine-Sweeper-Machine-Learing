from __future__ import annotations

import copy
import random

from minesweeper_ml.game import Cell, GameState, MinesweeperGame


POLICY_WEIGHTS = {
    "rule": 0.5,
    "frontier": 0.3,
    "global": 0.2,
}
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
):
    from minesweeper_ml.bots import RuleBasedMinesweeperBot

    rng = random.Random(seed)
    game = MinesweeperGame(
        width=width,
        height=height,
        mine_count=mine_count,
        seed=rng.randrange(2**63),
        first_safe=True,
    )

    bot = RuleBasedMinesweeperBot(width, height)
    phase_samples: dict[str, dict] = {}
    phase_counts = {phase: 0 for phase in PHASES}

    game.open_cell(0, 0)
    if game.state != GameState.IN_PROGRESS:
        return [], game.state

    while game.state == GameState.IN_PROGRESS:
        board_state = copy.deepcopy(game.visible_map)
        ground_truth_map = copy.deepcopy(game.mine_map)
        training_mask = frontier_mask(board_state)
        rule_move = bot.get_next_move(game.visible_map)
        safe_hidden_moves = _safe_hidden_moves(game)
        safe_frontier_moves = [
            (x, y)
            for x, y in safe_hidden_moves
            if training_mask[y][x]
        ]

        policy_moves = []
        if rule_move is not None:
            policy_moves.append(("rule", [rule_move], POLICY_WEIGHTS["rule"]))
        if safe_frontier_moves:
            policy_moves.append(("frontier", safe_frontier_moves, POLICY_WEIGHTS["frontier"]))
        if safe_hidden_moves:
            policy_moves.append(("global", safe_hidden_moves, POLICY_WEIGHTS["global"]))

        if not policy_moves:
            break

        selected_policy, candidates, _ = rng.choices(
            policy_moves,
            weights=[policy[2] for policy in policy_moves],
            k=1,
        )[0]
        x, y = rng.choice(candidates)

        if any(value for row in training_mask for value in row):
            phase = _game_phase(game)
            phase_counts[phase] += 1
            sample = {
                "game_id": game_id,
                "layout_id": tuple(sorted(game.mine_locations)),
                "phase": phase,
                "policy": selected_policy,
                "board_state": board_state,
                "ground_truth_map": ground_truth_map,
                "training_mask": training_mask,
                "chosen_move": (x, y),
                "is_safe": game.mine_map[y][x] != "M",
            }
            if rng.randrange(phase_counts[phase]) == 0:
                phase_samples[phase] = sample

        game.open_cell(x, y)

    game_steps = []
    for phase in PHASES:
        if phase not in phase_samples:
            continue
        sample = phase_samples[phase]
        sample["game_result"] = game.state
        game_steps.append(sample)

    return game_steps, game.state


def generate_training_data(
    num_games: int,
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    *,
    seed: int | None = None,
    verbose: bool = False,
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


def _safe_hidden_moves(game: MinesweeperGame):
    return [
        (x, y)
        for y in range(game.height)
        for x in range(game.width)
        if game.visible_map[y][x] == "-" and game.mine_map[y][x] != "M"
    ]


def _game_phase(game: MinesweeperGame) -> str:
    safe_cell_count = game.width * game.height - game.mine_count
    progress = 1.0 - (game.remaining_cells / safe_cell_count)
    if progress < 1 / 3:
        return "early"
    if progress < 2 / 3:
        return "middle"
    return "late"
