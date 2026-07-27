from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from minesweeper_ml.cli import (
    build_parser,
    evaluate_ml_bot_games,
    move_delay_for_mode,
    play_ml_bot_game,
    print_evaluation_summary,
    train_cnn_bot,
)


class RecordingModel:
    def __init__(self):
        self.fit_arguments = None
        self.evaluate_arguments = None

    def fit(self, features, labels, **kwargs):
        self.fit_arguments = (features, labels, kwargs)

    def evaluate(self, features, labels, **kwargs):
        self.evaluate_arguments = (features, labels, kwargs)
        return 0.25, 0.75


class UnexpectedPredictionModel:
    def predict(self, board, verbose=0):
        raise AssertionError("A completed first-click game should not request a prediction.")


class MineCountRequiringBot:
    def __init__(self, width, height, model, *, mine_count, cnn_input):
        self.strategy_stats = {
            "deduction": 0,
            "box": 0,
            "unconstrained": 0,
            "fallback": 0,
            "constraint_search_nodes": 0,
            "constraint_overflows": 0,
            "lookahead_decisions": 4,
            "lookahead_search_nodes": 123,
            "lookahead_budget_exhaustions": 1,
        }
        self.last_move_strategy = None

    def get_next_move(self, visible_map):
        raise AssertionError("The first click should complete this test game.")


class CliTest(unittest.TestCase):
    def test_smoke_mode_uses_fast_training_defaults(self):
        args = build_parser().parse_args(["--mode", "smoke"])

        self.assertEqual("smoke", args.mode)
        self.assertEqual(5, args.width)
        self.assertEqual(5, args.height)
        self.assertEqual(5, args.mines)
        self.assertEqual(2, args.games)
        self.assertEqual(1, args.epochs)

    def test_smoke_mode_skips_bot_move_delay(self):
        self.assertEqual(0.0, move_delay_for_mode("smoke"))
        self.assertEqual(1.0, move_delay_for_mode("train-cnn"))

    def test_training_defaults_include_seed_and_evaluation_games(self):
        args = build_parser().parse_args(["--mode", "train-cnn"])

        self.assertEqual(42, args.seed)
        self.assertEqual(100, args.eval_games)
        self.assertEqual(500, args.dev_games)

    def test_upgrade_mode_has_a_checkpoint_output(self):
        args = build_parser().parse_args(["--mode", "train-upgrade"])

        self.assertEqual("minesweeper_upgraded.keras", args.model_out)

    def test_smoke_mode_evaluates_two_seeded_games(self):
        args = build_parser().parse_args(["--mode", "smoke"])

        self.assertEqual(42, args.seed)
        self.assertEqual(2, args.eval_games)

    def test_training_splits_games_and_passes_frontier_sample_weights(self):
        dataset = [
            {
                "game_id": game_id,
                "board_state": [[1, "-"], ["-", "-"]],
                "ground_truth_map": [[1, "M"], [1, 1]],
                "training_mask": [[0, 1], [1, 1]],
            }
            for game_id in range(2)
        ]
        model = RecordingModel()

        with (
            patch("minesweeper_ml.cli.generate_training_data", return_value=dataset),
            patch("minesweeper_ml.cli.build_cnn_model", return_value=model),
            redirect_stdout(StringIO()),
        ):
            result = train_cnn_bot(
                width=2,
                height=2,
                mine_count=1,
                num_games=2,
                epochs=1,
                seed=11,
            )

        self.assertIs(model, result)
        train_features, train_labels, fit_options = model.fit_arguments
        test_features, test_labels, evaluate_options = model.evaluate_arguments
        self.assertEqual((1, 2, 2, 10), train_features.shape)
        self.assertEqual((1, 2, 2, 1), train_labels.shape)
        self.assertEqual((1, 2, 2), fit_options["sample_weight"].shape)
        self.assertEqual((1, 2, 2, 10), test_features.shape)
        self.assertEqual((1, 2, 2, 1), test_labels.shape)
        self.assertEqual((1, 2, 2), evaluate_options["sample_weight"].shape)

    def test_evaluation_opens_the_safe_zero_zero_first(self):
        summary = evaluate_ml_bot_games(
            UnexpectedPredictionModel(),
            width=2,
            height=1,
            mine_count=1,
            num_games=3,
            seed=9,
        )

        self.assertEqual(3, summary["won"])
        self.assertEqual(0, summary["lost"])
        self.assertEqual(0, summary["stopped"])
        self.assertEqual(1.0, summary["average_safe_moves"])
        self.assertEqual(
            {
                "deduction": 0,
                "box": 0,
                "unconstrained": 0,
                "fallback": 0,
            },
            summary["move_strategies"],
        )
        self.assertEqual(0.0, summary["median_uncertain_move_ms"])

    def test_evaluation_passes_mine_count_to_the_ml_bot(self):
        with patch("minesweeper_ml.cli.MLMinesweeperBot", MineCountRequiringBot):
            summary = evaluate_ml_bot_games(
                UnexpectedPredictionModel(),
                width=2,
                height=1,
                mine_count=1,
                num_games=1,
                seed=9,
            )

        self.assertEqual(1, summary["won"])
        self.assertEqual(4, summary["lookahead_decisions"])
        self.assertEqual(123, summary["lookahead_search_nodes"])
        self.assertEqual(1, summary["lookahead_budget_exhaustions"])

    def test_bot_game_passes_mine_count_to_the_ml_bot(self):
        with (
            patch("minesweeper_ml.cli.MLMinesweeperBot", MineCountRequiringBot),
            redirect_stdout(StringIO()),
        ):
            play_ml_bot_game(
                UnexpectedPredictionModel(),
                width=2,
                height=1,
                mine_count=1,
                move_delay=0,
                seed=9,
            )

    def test_evaluation_summary_reports_constraint_strategy_metrics(self):
        summary = {
            "won": 8,
            "lost": 2,
            "stopped": 0,
            "average_safe_moves": 29.5,
            "move_strategies": {
                "deduction": 10,
                "box": 3,
                "unconstrained": 2,
                "fallback": 1,
            },
            "constraint_overflows": 0,
            "lookahead_decisions": 4,
            "lookahead_search_nodes": 123,
            "lookahead_budget_exhaustions": 1,
            "median_uncertain_move_ms": 1.25,
            "fallback_move_rate": 1 / 6,
        }

        output = StringIO()
        with redirect_stdout(output):
            print_evaluation_summary(summary)

        self.assertIn(
            "Moves: 10 deduction, 3 box, 2 unconstrained, 1 fallback",
            output.getvalue(),
        )
        self.assertIn("0 constraint overflows", output.getvalue())
        self.assertIn("1.25 ms median uncertain move", output.getvalue())
        self.assertIn("16.67% uncertain-move fallback rate", output.getvalue())
        self.assertIn(
            "4 lookahead decisions, 123 lookahead search nodes, "
            "1 budget exhaustion",
            output.getvalue(),
        )
