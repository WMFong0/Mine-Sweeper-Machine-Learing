import random
import unittest
from unittest.mock import patch

from minesweeper_ml.bots import MLMinesweeperBot, RuleBasedMinesweeperBot
from minesweeper_ml.lookahead import LookaheadEvaluation


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
        self.assertEqual(0.75, bot.model_prior_strength)
        self.assertEqual(0.0025, bot.tie_margin)
        self.assertEqual(4, bot.lookahead_max_candidates)
        self.assertEqual(100_000, bot.lookahead_max_nodes)
        self.assertEqual(1e-6, bot.lookahead_min_outcome_probability)

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

    def test_rejects_invalid_lookahead_parameters(self):
        model = ScoreMapPredictionModel([[0.5]])

        with self.assertRaises(ValueError):
            MLMinesweeperBot(
                1,
                1,
                model,
                lookahead_max_candidates=-1,
            )
        with self.assertRaises(ValueError):
            MLMinesweeperBot(1, 1, model, lookahead_max_nodes=0)
        with self.assertRaises(ValueError):
            MLMinesweeperBot(
                1,
                1,
                model,
                lookahead_min_outcome_probability=-0.1,
            )
        with self.assertRaises(ValueError):
            MLMinesweeperBot(
                1,
                1,
                model,
                lookahead_min_outcome_probability=1.1,
            )

    def test_effective_tie_margin_shrinks_in_the_endgame(self):
        bot = MLMinesweeperBot(
            width=10,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5] * 10]),
            tie_margin=0.01,
        )

        self.assertAlmostEqual(0.01, bot._effective_tie_margin(hidden_count=10))
        self.assertAlmostEqual(0.0025, bot._effective_tie_margin(hidden_count=1))

    def test_lookahead_never_evaluates_a_cell_outside_dynamic_margin(self):
        bot = MLMinesweeperBot(
            width=10,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5] * 10]),
            tie_margin=0.01,
        )
        visible_map = [["-"] * 10]
        probabilities = {
            (0, 0): 0.10,
            (1, 0): 0.105,
            (2, 0): 0.111,
        }

        with patch(
            "minesweeper_ml.bots.evaluate_safe_click",
            return_value=_lookahead_evaluation(expected_forced_cells=0.0),
        ) as scorer:
            move = bot._select_posterior_move(
                probabilities,
                visible_map=visible_map,
                deduced_mines=set(),
                constraints=[],
                hidden_cells={(x, 0) for x in range(10)},
                model_mine_probabilities={(x, 0): 0.5 for x in range(10)},
                remaining_mines=None,
            )

        self.assertIn(move, {(0, 0), (1, 0)})
        self.assertEqual(
            [(0, 0), (1, 0)],
            [call.args[0] for call in scorer.call_args_list],
        )

    def test_lookahead_caps_candidates_in_probability_then_row_major_order(self):
        bot = MLMinesweeperBot(
            width=5,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5] * 5]),
            tie_margin=0.1,
            lookahead_max_candidates=2,
        )
        visible_map = [["-"] * 5]

        with patch(
            "minesweeper_ml.bots.evaluate_safe_click",
            return_value=_lookahead_evaluation(expected_forced_cells=0.0),
        ) as scorer:
            bot._select_posterior_move(
                {
                    (0, 0): 0.12,
                    (1, 0): 0.11,
                    (2, 0): 0.10,
                    (3, 0): 0.11,
                },
                visible_map=visible_map,
                deduced_mines=set(),
                constraints=[],
                hidden_cells={(x, 0) for x in range(5)},
                model_mine_probabilities={(x, 0): 0.5 for x in range(5)},
                remaining_mines=None,
            )

        self.assertEqual(
            [(2, 0), (1, 0)],
            [call.args[0] for call in scorer.call_args_list],
        )

    def test_lookahead_prefers_more_expected_forced_cells(self):
        bot = MLMinesweeperBot(
            width=2,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5, 0.5]]),
            tie_margin=0.01,
        )
        visible_map = [["-", "-"]]

        with patch(
            "minesweeper_ml.bots.evaluate_safe_click",
            side_effect=[
                _lookahead_evaluation(expected_forced_cells=0.5),
                _lookahead_evaluation(expected_forced_cells=1.5),
            ],
        ):
            move = bot._select_posterior_move(
                {(0, 0): 0.10, (1, 0): 0.105},
                visible_map=visible_map,
                deduced_mines=set(),
                constraints=[],
                hidden_cells={(0, 0), (1, 0)},
                model_mine_probabilities={(0, 0): 0.5, (1, 0): 0.5},
                remaining_mines=None,
            )

        self.assertEqual((1, 0), move)
        self.assertEqual(1, bot.strategy_stats["lookahead_decisions"])

    def test_lookahead_uses_entropy_then_zero_region_to_break_ties(self):
        bot = MLMinesweeperBot(
            width=3,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5, 0.5, 0.5]]),
        )
        visible_map = [["-", "-", "-"]]

        with patch(
            "minesweeper_ml.bots.evaluate_safe_click",
            side_effect=[
                _lookahead_evaluation(
                    expected_forced_cells=1.0,
                    expected_entropy_reduction=0.2,
                    zero_region_probability=0.9,
                ),
                _lookahead_evaluation(
                    expected_forced_cells=1.0,
                    expected_entropy_reduction=0.3,
                    zero_region_probability=0.1,
                ),
                _lookahead_evaluation(
                    expected_forced_cells=1.0,
                    expected_entropy_reduction=0.3,
                    zero_region_probability=0.8,
                ),
            ],
        ):
            move = bot._select_posterior_move(
                {
                    (0, 0): 0.10,
                    (1, 0): 0.10,
                    (2, 0): 0.10,
                },
                visible_map=visible_map,
                deduced_mines=set(),
                constraints=[],
                hidden_cells={(0, 0), (1, 0), (2, 0)},
                model_mine_probabilities={
                    (0, 0): 0.5,
                    (1, 0): 0.5,
                    (2, 0): 0.5,
                },
                remaining_mines=None,
            )

        self.assertEqual((2, 0), move)

    def test_zero_candidate_limit_disables_lookahead(self):
        bot = MLMinesweeperBot(
            width=4,
            height=2,
            ml_model=ScoreMapPredictionModel([[0.5] * 4, [0.5] * 4]),
            tie_margin=0.01,
            lookahead_max_candidates=0,
        )
        visible_map = [
            ["-", "-", "-", "-"],
            [1, "-", "-", "-"],
        ]
        hidden_cells = {
            (x, y)
            for y, row in enumerate(visible_map)
            for x, value in enumerate(row)
            if value == "-"
        }

        with patch("minesweeper_ml.bots.evaluate_safe_click") as scorer:
            move = bot._select_posterior_move(
                {(0, 0): 0.10, (1, 0): 0.105},
                visible_map=visible_map,
                deduced_mines=set(),
                constraints=[],
                hidden_cells=hidden_cells,
                model_mine_probabilities={
                    cell: 0.5 for cell in hidden_cells
                },
                remaining_mines=None,
            )

        self.assertEqual((1, 0), move)
        scorer.assert_not_called()

    def test_one_candidate_limit_avoids_unnecessary_lookahead(self):
        bot = MLMinesweeperBot(
            width=2,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5, 0.5]]),
            lookahead_max_candidates=1,
        )
        visible_map = [["-", "-"]]

        with patch("minesweeper_ml.bots.evaluate_safe_click") as scorer:
            move = bot._select_posterior_move(
                {(0, 0): 0.10, (1, 0): 0.105},
                visible_map=visible_map,
                deduced_mines=set(),
                constraints=[],
                hidden_cells={(0, 0), (1, 0)},
                model_mine_probabilities={(0, 0): 0.5, (1, 0): 0.5},
                remaining_mines=None,
            )

        self.assertEqual((0, 0), move)
        scorer.assert_not_called()

    def test_lookahead_budget_exhaustion_falls_back_deterministically(self):
        bot = MLMinesweeperBot(
            width=4,
            height=2,
            ml_model=ScoreMapPredictionModel([[0.5] * 4, [0.5] * 4]),
            tie_margin=0.01,
        )
        visible_map = [
            ["-", "-", "-", "-"],
            [1, "-", "-", "-"],
        ]
        hidden_cells = {
            (x, y)
            for y, row in enumerate(visible_map)
            for x, value in enumerate(row)
            if value == "-"
        }

        with patch(
            "minesweeper_ml.bots.evaluate_safe_click",
            return_value=_lookahead_evaluation(
                valid=False,
                budget_exhausted=True,
                search_nodes=100_000,
            ),
        ):
            moves = [
                bot._select_posterior_move(
                    {(0, 0): 0.10, (1, 0): 0.105},
                    visible_map=visible_map,
                    deduced_mines=set(),
                    constraints=[],
                    hidden_cells=hidden_cells,
                    model_mine_probabilities={
                        cell: 0.5 for cell in hidden_cells
                    },
                    remaining_mines=None,
                )
                for _ in range(2)
            ]

        self.assertEqual([(1, 0), (1, 0)], moves)
        self.assertEqual(
            2,
            bot.strategy_stats["lookahead_budget_exhaustions"],
        )
        self.assertEqual(
            200_000,
            bot.strategy_stats["lookahead_search_nodes"],
        )

    def test_partial_budget_exhaustion_keeps_completed_candidate_score(self):
        bot = MLMinesweeperBot(
            width=2,
            height=1,
            ml_model=ScoreMapPredictionModel([[0.5, 0.5]]),
            lookahead_max_nodes=100,
        )
        visible_map = [["-", "-"]]

        with patch(
            "minesweeper_ml.bots.evaluate_safe_click",
            side_effect=[
                _lookahead_evaluation(
                    expected_forced_cells=1.0,
                    search_nodes=90,
                ),
                _lookahead_evaluation(
                    valid=False,
                    budget_exhausted=True,
                    search_nodes=10,
                ),
            ],
        ):
            move = bot._select_posterior_move(
                {(0, 0): 0.10, (1, 0): 0.10},
                visible_map=visible_map,
                deduced_mines=set(),
                constraints=[],
                hidden_cells={(0, 0), (1, 0)},
                model_mine_probabilities={(0, 0): 0.5, (1, 0): 0.5},
                remaining_mines=None,
            )

        self.assertEqual((0, 0), move)
        self.assertEqual(1, bot.strategy_stats["lookahead_decisions"])
        self.assertEqual(
            1,
            bot.strategy_stats["lookahead_budget_exhaustions"],
        )
        self.assertEqual(100, bot.strategy_stats["lookahead_search_nodes"])

    def test_real_lookahead_changes_end_to_end_move_selection(self):
        scores = [
            [0.301625, 0.788729, 0.383552, 0.533665, 0.656705],
            [0.701596, 0.248279, 0.827334, 0.451968, 0.670399],
            [0.421937, 0.593413, 0.742202, 0.223084, 0.659523],
            [0.631690, 0.218985, 0.252209, 0.359092, 0.805340],
            [0.596335, 0.713219, 0.216447, 0.323539, 0.395663],
        ]
        visible_map = [
            [2, "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-"],
            ["-", "-", "-", 1, "-"],
            ["-", "-", "-", "-", "-"],
        ]
        baseline = MLMinesweeperBot(
            5,
            5,
            ScoreMapPredictionModel(scores),
            mine_count=5,
            model_prior_strength=0.75,
            tie_margin=0.01,
            lookahead_max_candidates=0,
        )
        lookahead = MLMinesweeperBot(
            5,
            5,
            ScoreMapPredictionModel(scores),
            mine_count=5,
            model_prior_strength=0.75,
            tie_margin=0.01,
            lookahead_max_candidates=4,
            exact_lookahead=True,
        )

        self.assertEqual((2, 2), baseline.get_next_move(visible_map))
        self.assertEqual((4, 3), lookahead.get_next_move(visible_map))
        self.assertEqual(1, lookahead.strategy_stats["lookahead_decisions"])
        self.assertEqual(
            0,
            lookahead.strategy_stats["lookahead_search_nodes"],
        )

    def test_value_head_reorders_only_posterior_tied_candidates(self):
        bot = MLMinesweeperBot(
            3,
            1,
            ScoreMapPredictionModel([[0.5, 0.5, 0.5]]),
            tie_margin=0.01,
            lookahead_max_candidates=0,
        )
        visible_map = [["-", "-", "-"]]

        tied = bot._select_posterior_move(
            {(0, 0): 0.10, (1, 0): 0.105, (2, 0): 0.20},
            visible_map=visible_map,
            deduced_mines=set(),
            constraints=[],
            hidden_cells={(0, 0), (1, 0), (2, 0)},
            model_mine_probabilities={
                (0, 0): 0.5,
                (1, 0): 0.5,
                (2, 0): 0.5,
            },
            remaining_mines=None,
            candidate_values={(0, 0): 0.1, (1, 0): 0.9, (2, 0): 1.0},
        )
        materially_safer = bot._select_posterior_move(
            {(0, 0): 0.10, (1, 0): 0.12},
            visible_map=visible_map,
            deduced_mines=set(),
            constraints=[],
            hidden_cells={(0, 0), (1, 0), (2, 0)},
            model_mine_probabilities={
                (0, 0): 0.5,
                (1, 0): 0.5,
                (2, 0): 0.5,
            },
            remaining_mines=None,
            candidate_values={(0, 0): 0.1, (1, 0): 1.0},
        )

        self.assertEqual((1, 0), tied)
        self.assertEqual((0, 0), materially_safer)

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
        bot = MLMinesweeperBot(
            width=4,
            height=2,
            ml_model=model,
            model_prior_strength=0.5,
            tie_margin=0.01,
        )
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


def _lookahead_evaluation(
    *,
    expected_forced_cells=0.0,
    expected_entropy_reduction=0.0,
    zero_region_probability=0.0,
    search_nodes=10,
    budget_exhausted=False,
    valid=True,
):
    return LookaheadEvaluation(
        expected_forced_cells=expected_forced_cells,
        expected_entropy_reduction=expected_entropy_reduction,
        zero_region_probability=zero_region_probability,
        search_nodes=search_nodes,
        budget_exhausted=budget_exhausted,
        valid=valid,
    )
