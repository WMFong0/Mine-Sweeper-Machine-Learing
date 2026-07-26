import random
import unittest

from minesweeper_ml.bots import MLMinesweeperBot, RuleBasedMinesweeperBot


class SpatialPredictionModel:
    def __init__(self, best_coordinate, width, height):
        self.best_coordinate = best_coordinate
        self.width = width
        self.height = height
        self.last_input = None

    def predict(self, board, verbose=0):
        import numpy as np

        self.last_input = board
        predictions = np.full((1, self.height, self.width, 1), 0.1, dtype=np.float32)
        x, y = self.best_coordinate
        predictions[0, y, x, 0] = 0.9
        return predictions


class ScoreMapPredictionModel:
    def __init__(self, scores):
        self.scores = scores
        self.last_input = None

    def predict(self, board, verbose=0):
        import numpy as np

        self.last_input = board
        return np.asarray(self.scores, dtype=np.float32).reshape(
            1,
            len(self.scores),
            len(self.scores[0]),
            1,
        )


class RuleBasedMinesweeperBotTest(unittest.TestCase):
    def test_clue_risks_use_the_most_conservative_visible_constraint(self):
        visible_map = [
            ["-", "-", "-", "-", "-"],
            [0, 1, 0, 2, 0],
        ]
        bot = RuleBasedMinesweeperBot(width=5, height=2)

        risks = bot._clue_mine_risks(visible_map, set(), set())

        self.assertAlmostEqual(1 / 3, risks[(0, 0)])
        self.assertAlmostEqual(2 / 3, risks[(2, 0)])
        self.assertAlmostEqual(2 / 3, risks[(4, 0)])

    def test_uses_subset_constraints_to_choose_a_deduced_safe_move(self):
        visible_map = [
            [0, "-", "-", "-"],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        bot = RuleBasedMinesweeperBot(width=4, height=3)

        random.seed(1)

        self.assertEqual((3, 0), bot.get_next_move(visible_map))

    def test_probability_fallback_prefers_the_lowest_estimated_mine_risk(self):
        visible_map = [
            ["-", "-", "-", "-", "-"],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        bot = RuleBasedMinesweeperBot(width=5, height=3)

        random.seed(0)

        self.assertEqual((0, 0), bot.get_next_move(visible_map))


class MLMinesweeperBotTest(unittest.TestCase):
    def test_uses_evaluated_scoring_defaults(self):
        bot = MLMinesweeperBot(
            width=1,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5]]),
        )

        self.assertEqual(0.6, bot.model_weight)
        self.assertEqual(0.01, bot.tie_margin)

    def test_combines_model_confidence_with_clue_safety(self):
        model = ScoreMapPredictionModel(
            [
                [0.80, 0.10, 0.10, 0.90, 0.10],
                [0.10, 0.10, 0.10, 0.10, 0.10],
            ]
        )
        bot = MLMinesweeperBot(width=5, height=2, ml_model=model)
        visible_map = [
            ["-", "-", "-", "-", "-"],
            [0, 1, 0, 2, 0],
        ]

        self.assertEqual((0, 0), bot.get_next_move(visible_map))

    def test_equal_clue_risk_follows_the_stronger_model_score(self):
        model = ScoreMapPredictionModel([[0.60, 0.80], [0.10, 0.10]])
        bot = MLMinesweeperBot(width=2, height=2, ml_model=model)
        visible_map = [["-", "-"], [1, 1]]

        self.assertEqual((1, 0), bot.get_next_move(visible_map))

    def test_ranks_model_predictions_below_one_half_when_a_guess_is_required(self):
        model = ScoreMapPredictionModel([[0.20, 0.40], [0.10, 0.10]])
        bot = MLMinesweeperBot(width=2, height=2, ml_model=model)
        visible_map = [["-", "-"], [1, 1]]

        self.assertEqual((1, 0), bot.get_next_move(visible_map))

    def test_deterministic_safe_move_bypasses_model_prediction(self):
        model = SpatialPredictionModel(best_coordinate=(0, 0), width=4, height=3)
        bot = MLMinesweeperBot(width=4, height=3, ml_model=model)
        visible_map = [
            [0, "-", "-", "-"],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]

        self.assertEqual((3, 0), bot.get_next_move(visible_map))
        self.assertIsNone(model.last_input)

    def test_rejects_invalid_scoring_parameters(self):
        model = ScoreMapPredictionModel([[0.5]])

        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, model_weight=-0.01)
        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, model_weight=1.01)
        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, tie_margin=-0.01)

    def test_rejects_invalid_constraint_parameters(self):
        model = ScoreMapPredictionModel([[0.5]])

        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, mine_count=-1)
        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, mine_count=2)
        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, max_constraint_nodes=0)
        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, model_prior_strength=0.0)

    def test_global_mine_budget_records_a_certain_safe_cell_as_a_deduction(self):
        model = ScoreMapPredictionModel(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )
        bot = MLMinesweeperBot(
            width=5,
            height=2,
            ml_model=model,
            mine_count=1,
        )
        visible_map = [
            ["-", "-", "-", "-", "-"],
            [1, "-", "-", "-", "-"],
        ]

        self.assertEqual((2, 0), bot.get_next_move(visible_map))
        self.assertEqual("deduction", bot.last_move_strategy)

    def test_constraint_overflow_uses_existing_hybrid_fallback(self):
        model = ScoreMapPredictionModel([[0.60, 0.80], [0.10, 0.10]])
        bot = MLMinesweeperBot(
            width=2,
            height=2,
            ml_model=model,
            mine_count=1,
            max_constraint_nodes=1,
        )
        visible_map = [["-", "-"], [1, 1]]

        self.assertEqual((1, 0), bot.get_next_move(visible_map))
        self.assertEqual("fallback", bot.last_move_strategy)
        self.assertGreater(bot.strategy_stats["constraint_overflows"], 0)

    def test_constraint_overflow_keeps_non_certain_completed_box_scores(self):
        model = ScoreMapPredictionModel(
            [
                [0.99, 0.10, 0.10, 0.10, 0.10, 0.95, 0.95],
                [0.10, 0.10, 0.10, 0.10, 0.00, 0.95, 0.10],
            ]
        )
        bot = MLMinesweeperBot(
            width=7,
            height=2,
            ml_model=model,
            max_constraint_nodes=5,
        )
        visible_map = [
            ["-", "-", 0, 0, 0, "-", "-"],
            [1, 1, 0, 0, "-", "-", 1],
        ]

        self.assertEqual((0, 0), bot.get_next_move(visible_map))
        self.assertEqual("box", bot.last_move_strategy)
        self.assertEqual(1, bot.strategy_stats["constraint_overflows"])

    def test_near_equal_scores_prefer_the_move_with_more_hidden_neighbors(self):
        model = ScoreMapPredictionModel(
            [
                [0.81, 0.80, 0.10, 0.10],
                [0.10, 0.10, 0.10, 0.10],
            ]
        )
        bot = MLMinesweeperBot(width=4, height=2, ml_model=model)
        visible_map = [
            ["-", "-", "-", "-"],
            [1, "-", "-", "-"],
        ]

        self.assertEqual((1, 0), bot.get_next_move(visible_map))

    def test_materially_safer_score_wins_over_information_gain(self):
        model = ScoreMapPredictionModel(
            [
                [0.90, 0.70, 0.10, 0.10],
                [0.10, 0.10, 0.10, 0.10],
            ]
        )
        bot = MLMinesweeperBot(width=4, height=2, ml_model=model)
        visible_map = [
            ["-", "-", "-", "-"],
            [1, "-", "-", "-"],
        ]

        self.assertEqual((0, 0), bot.get_next_move(visible_map))

    def test_maps_spatial_predictions_to_frontier_coordinates(self):
        model = SpatialPredictionModel(best_coordinate=(1, 1), width=3, height=2)
        bot = MLMinesweeperBot(width=3, height=2, ml_model=model)
        visible_map = [
            [1, "-", "-"],
            ["-", "-", "-"],
        ]

        self.assertEqual((1, 1), bot.get_next_move(visible_map))
        self.assertEqual((1, 2, 3, 10), model.last_input.shape)

    def test_keeps_the_choice_inside_the_trained_frontier(self):
        model = SpatialPredictionModel(best_coordinate=(3, 0), width=4, height=3)
        bot = MLMinesweeperBot(width=4, height=3, ml_model=model)
        visible_map = [
            ["-", "-", "-", "-"],
            ["-", 1, "-", "-"],
            ["-", "-", "-", "-"],
        ]

        move = bot.get_next_move(visible_map)

        self.assertNotEqual((3, 0), move)
        self.assertTrue(bot._has_revealed_neighbor(*move, visible_map))

    def test_does_not_treat_a_flag_as_revealed_frontier_evidence(self):
        model = SpatialPredictionModel(best_coordinate=(2, 1), width=3, height=2)
        bot = MLMinesweeperBot(width=3, height=2, ml_model=model)
        visible_map = [
            ["-", "-", "F"],
            ["-", "-", "-"],
        ]

        self.assertEqual((0, 0), bot.get_next_move(visible_map))
