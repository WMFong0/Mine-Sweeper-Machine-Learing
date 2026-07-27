from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from minesweeper_ml.bots import MLMinesweeperBot
from minesweeper_ml.data import (
    build_sample_weights,
    calculate_class_weights,
    dataset_to_arrays,
    generate_training_data,
    split_dataset_by_game,
)
from minesweeper_ml.game import GameState, MinesweeperGame
from minesweeper_ml.models import build_cnn_model


LEGACY_STRATEGY_OPTIONS = {
    "uniform_constraint_posterior": False,
    "exact_lookahead": False,
    "endgame_max_worlds": 0,
}


DEFAULTS = {
    "width": 10,
    "height": 10,
    "mines": 15,
    "games": 50000,
    "epochs": 50,
    "seed": 42,
    "eval_games": 100,
    "dev_games": 500,
}

SMOKE_DEFAULTS = {
    "width": 5,
    "height": 5,
    "mines": 5,
    "games": 2,
    "epochs": 1,
    "seed": 42,
    "eval_games": 2,
    "dev_games": 2,
}


class MinesweeperArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed_args = super().parse_args(args, namespace)
        defaults = SMOKE_DEFAULTS if parsed_args.mode == "smoke" else DEFAULTS
        for name, value in defaults.items():
            if getattr(parsed_args, name) is None:
                setattr(parsed_args, name, value)
        return parsed_args


def play_user_game(width: int = 5, height: int = 5, mine_count: int = 5) -> None:
    game = MinesweeperGame(width=width, height=height, mine_count=mine_count)
    game.display_mine_map()

    while game.state == GameState.IN_PROGRESS:
        print("===" * 30)
        game.display_visible_map()
        raw_value = input("Input your next x,y, or -1,-1 / empty to exit: ").replace(" ", "")

        if raw_value in {"", "-1,-1"}:
            game.state = GameState.STOPPED
            break

        try:
            x_text, y_text = raw_value.split(",")
            x = int(x_text)
            y = int(y_text)
        except ValueError:
            print("Invalid input. Use x,y format.")
            continue

        if not (0 <= x < width and 0 <= y < height):
            print(f"Invalid input. x must be 0-{width - 1} and y must be 0-{height - 1}.")
            continue
        if game.visible_map[y][x] != "-":
            print("Cell already opened, please choose another cell.")
            continue

        game.open_cell(x, y)

    _print_result(game)


def train_cnn_bot(
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    num_games: int = 50000,
    epochs: int = 50,
    seed: int = 42,
):
    training_dataset = generate_training_data(
        num_games,
        width,
        height,
        mine_count,
        seed=seed,
        verbose=True,
    )
    train_dataset, test_dataset = split_dataset_by_game(training_dataset, seed=seed)
    x_train, y_train, train_masks = dataset_to_arrays(train_dataset)
    x_test, y_test, test_masks = dataset_to_arrays(test_dataset)
    class_weights = calculate_class_weights(y_train, train_masks)
    train_weights = build_sample_weights(y_train, train_masks, class_weights)
    test_weights = build_sample_weights(y_test, test_masks, class_weights)

    model = build_cnn_model(width, height)
    model.fit(
        x_train,
        y_train,
        sample_weight=train_weights,
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        sample_weight=test_weights,
        verbose=0,
    )
    print(f"CNN masked test loss: {loss:.4f}")
    print(f"CNN masked test accuracy: {accuracy:.4f}")
    return model


def play_ml_bot_game(
    model,
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    move_delay: float = 1.0,
    seed: int | None = None,
    bot_options: dict | None = None,
) -> None:
    game = MinesweeperGame(
        width=width,
        height=height,
        mine_count=mine_count,
        seed=seed,
        first_safe=True,
    )

    options = {"mine_count": mine_count, "cnn_input": True}
    options.update(bot_options or {})
    bot = MLMinesweeperBot(width, height, model, **options)
    print("Bot making initial move at (0,0).")
    game.open_cell(0, 0)

    while game.state == GameState.IN_PROGRESS:
        print("===" * 30)
        game.display_visible_map()
        print(f"Remaining cells to open: {game.remaining_cells}")
        if move_delay > 0:
            time.sleep(move_delay)

        bot_move = bot.get_next_move(game.visible_map)
        if bot_move is None:
            print("ML Bot has no safe moves to make or no moves left. Avoiding guess.")
            game.state = GameState.STOPPED
            break

        x, y = bot_move
        print(f"ML Bot chose to open: {x},{y}")
        if not game.open_cell(x, y):
            print("ML Bot stepped on a mine!")
            break

    _print_result(game)


def evaluate_ml_bot_games(
    model,
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    num_games: int = 100,
    seed: int = 42,
    bot_options: dict | None = None,
):
    if num_games <= 0:
        raise ValueError("num_games must be greater than zero.")

    rng = random.Random(seed)
    summary = {
        "won": 0,
        "lost": 0,
        "stopped": 0,
        "average_safe_moves": 0.0,
        "move_strategies": {
            "deduction": 0,
            "box": 0,
            "unconstrained": 0,
            "fallback": 0,
        },
        "constraint_search_nodes": 0,
        "constraint_overflows": 0,
        "lookahead_decisions": 0,
        "lookahead_search_nodes": 0,
        "lookahead_budget_exhaustions": 0,
        "endgame_decisions": 0,
        "endgame_search_nodes": 0,
        "endgame_evaluated_states": 0,
        "endgame_budget_exhaustions": 0,
        "median_uncertain_move_ms": 0.0,
        "fallback_move_rate": 0.0,
    }
    total_safe_moves = 0
    uncertain_move_latencies = []

    for _ in range(num_games):
        game = MinesweeperGame(
            width=width,
            height=height,
            mine_count=mine_count,
            seed=rng.randrange(2**63),
            first_safe=True,
        )
        options = {"mine_count": mine_count, "cnn_input": True}
        options.update(bot_options or {})
        bot = MLMinesweeperBot(width, height, model, **options)
        game.open_cell(0, 0)
        safe_moves = 1

        while game.state == GameState.IN_PROGRESS:
            move_started_at = time.perf_counter()
            bot_move = bot.get_next_move(game.visible_map)
            move_elapsed_ms = (time.perf_counter() - move_started_at) * 1000
            if bot.last_move_strategy not in {None, "deduction"}:
                uncertain_move_latencies.append(move_elapsed_ms)
            if bot_move is None:
                game.state = GameState.STOPPED
                break

            x, y = bot_move
            if game.open_cell(x, y):
                safe_moves += 1

        for strategy in summary["move_strategies"]:
            summary["move_strategies"][strategy] += bot.strategy_stats[strategy]
        summary["constraint_search_nodes"] += bot.strategy_stats[
            "constraint_search_nodes"
        ]
        summary["constraint_overflows"] += bot.strategy_stats[
            "constraint_overflows"
        ]
        summary["lookahead_decisions"] += bot.strategy_stats[
            "lookahead_decisions"
        ]
        summary["lookahead_search_nodes"] += bot.strategy_stats[
            "lookahead_search_nodes"
        ]
        summary["lookahead_budget_exhaustions"] += bot.strategy_stats[
            "lookahead_budget_exhaustions"
        ]
        for metric in (
            "endgame_decisions",
            "endgame_search_nodes",
            "endgame_evaluated_states",
            "endgame_budget_exhaustions",
        ):
            summary[metric] += bot.strategy_stats.get(metric, 0)
        total_safe_moves += safe_moves
        if game.state == GameState.WON:
            summary["won"] += 1
        elif game.state == GameState.LOST:
            summary["lost"] += 1
        else:
            summary["stopped"] += 1

    summary["average_safe_moves"] = total_safe_moves / num_games
    if uncertain_move_latencies:
        summary["median_uncertain_move_ms"] = statistics.median(
            uncertain_move_latencies
        )
    uncertain_moves = sum(
        summary["move_strategies"][strategy]
        for strategy in ("box", "unconstrained", "fallback")
    )
    if uncertain_moves:
        summary["fallback_move_rate"] = (
            summary["move_strategies"]["fallback"] / uncertain_moves
        )
    return summary


def print_evaluation_summary(summary) -> None:
    print(
        "Fresh-game evaluation: "
        f"{summary['won']} won, "
        f"{summary['lost']} lost, "
        f"{summary['stopped']} stopped, "
        f"{summary['average_safe_moves']:.2f} average safe moves"
    )
    strategies = summary["move_strategies"]
    print(
        "Moves: "
        f"{strategies['deduction']} deduction, "
        f"{strategies['box']} box, "
        f"{strategies['unconstrained']} unconstrained, "
        f"{strategies['fallback']} fallback"
    )
    print(
        f"{summary['constraint_overflows']} constraint overflows, "
        f"{summary['median_uncertain_move_ms']:.2f} ms median uncertain move"
    )
    print(
        f"{summary['fallback_move_rate']:.2%} uncertain-move fallback rate"
    )
    exhaustion_label = (
        "budget exhaustion"
        if summary["lookahead_budget_exhaustions"] == 1
        else "budget exhaustions"
    )
    print(
        f"{summary['lookahead_decisions']} lookahead decisions, "
        f"{summary['lookahead_search_nodes']} lookahead search nodes, "
        f"{summary['lookahead_budget_exhaustions']} {exhaustion_label}"
    )
    endgame_exhaustion_label = (
        "budget exhaustion"
        if summary["endgame_budget_exhaustions"] == 1
        else "budget exhaustions"
    )
    print(
        f"{summary['endgame_decisions']} endgame decisions, "
        f"{summary['endgame_search_nodes']} endgame search nodes, "
        f"{summary['endgame_evaluated_states']} states, "
        f"{summary['endgame_budget_exhaustions']} "
        f"{endgame_exhaustion_label}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = MinesweeperArgumentParser(description="Play or train the Minesweeper ML project.")
    parser.add_argument(
        "--mode",
        choices=["user", "train-cnn", "train-upgrade", "smoke"],
        default="user",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--mines", type=int)
    parser.add_argument("--games", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-games", type=int, dest="eval_games")
    parser.add_argument("--dev-games", type=int, dest="dev_games")
    parser.add_argument("--teacher-model")
    parser.add_argument("--model-out", default="minesweeper_upgraded.keras")
    return parser


def move_delay_for_mode(mode: str) -> float:
    return 0.0 if mode == "smoke" else 1.0


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "user":
        play_user_game(args.width, args.height, args.mines)
        return

    if args.mode == "train-upgrade":
        from minesweeper_ml.training import train_upgrade_pipeline

        teacher_model = None
        if args.teacher_model:
            from tensorflow import keras

            teacher_model = keras.models.load_model(args.teacher_model)
        model, report, teacher_model = train_upgrade_pipeline(
            width=args.width,
            height=args.height,
            mine_count=args.mines,
            num_games=args.games,
            epochs=args.epochs,
            seed=args.seed,
            development_games=args.dev_games,
            teacher_model=teacher_model,
        )
        from minesweeper_ml.benchmark import compare_bot_factories

        selected_options = report["selected"]["bot_options"]

        def baseline_factory(width, height, mine_count):
            return MLMinesweeperBot(
                width,
                height,
                teacher_model,
                mine_count=mine_count,
                symmetry_ensemble=False,
                use_candidate_value=False,
                **LEGACY_STRATEGY_OPTIONS,
            )

        def candidate_factory(width, height, mine_count):
            return MLMinesweeperBot(
                width,
                height,
                model,
                **selected_options,
            )

        report["paired_benchmark"] = compare_bot_factories(
            baseline_factory,
            candidate_factory,
            width=args.width,
            height=args.height,
            mine_count=args.mines,
            num_games=args.eval_games,
            seed=args.seed + 3,
        )
        model.save(args.model_out)
        print(json.dumps(report, sort_keys=True))
        return

    model = train_cnn_bot(
        args.width,
        args.height,
        args.mines,
        args.games,
        args.epochs,
        seed=args.seed,
    )
    summary = evaluate_ml_bot_games(
        model,
        args.width,
        args.height,
        args.mines,
        num_games=args.eval_games,
        seed=args.seed + 1,
    )
    print_evaluation_summary(summary)
    play_ml_bot_game(
        model,
        args.width,
        args.height,
        args.mines,
        move_delay=move_delay_for_mode(args.mode),
        seed=args.seed + 2,
    )


def _print_result(game: MinesweeperGame) -> None:
    if game.state == GameState.WON:
        game.display_visible_map()
        print("You win!")
    elif game.state == GameState.LOST:
        game.display_mine_map()
        print("You stepped on a mine. Game over")
    elif game.state == GameState.STOPPED:
        print("Game stopped by User or Bot could not find a safe move.")
    else:
        print("Game ended unexpectedly.")


if __name__ == "__main__":
    main()
