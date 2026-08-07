import sys
import unittest

from minesweeper_ml.game import build_constraint_context
from minesweeper_ml.game import solve_endgame


class ExactEndgameTest(unittest.TestCase):
    def test_edge_click_maximizes_eventual_win_probability(self):
        hidden_cells = {(0, 0), (1, 0), (2, 0)}

        result = solve_endgame(
            [],
            hidden_cells=hidden_cells,
            remaining_mines=1,
            width=3,
            height=1,
            max_worlds=256,
            max_search_nodes=10_000,
            max_constraint_nodes=10_000,
        )

        self.assertTrue(result.valid)
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(3, result.legal_worlds)
        self.assertEqual((0, 0), result.move)
        self.assertAlmostEqual(2.0 / 3.0, result.win_probability)

    def test_declines_positions_above_the_world_limit(self):
        hidden_cells = {(index, 0) for index in range(10)}

        result = solve_endgame(
            [],
            hidden_cells=hidden_cells,
            remaining_mines=5,
            width=10,
            height=1,
            max_worlds=32,
            max_search_nodes=10_000,
            max_constraint_nodes=10_000,
        )

        self.assertFalse(result.valid)
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(252, result.legal_worlds)
        self.assertIsNone(result.move)

    def test_rebuilds_a_cnn_weighted_root_as_uniform(self):
        hidden_cells = {(0, 0), (1, 0), (2, 0)}
        constraint = (frozenset(hidden_cells), 1)
        weighted_context = build_constraint_context(
            [constraint],
            hidden_cells=hidden_cells,
            model_mine_probabilities={
                (0, 0): 0.01,
                (1, 0): 0.5,
                (2, 0): 0.99,
            },
            remaining_mines=1,
            prior_strength=1.0,
            max_search_nodes=10_000,
        )

        result = solve_endgame(
            [constraint],
            hidden_cells=hidden_cells,
            remaining_mines=1,
            width=3,
            height=1,
            inference_context=weighted_context,
        )

        self.assertTrue(result.valid)
        self.assertEqual((0, 0), result.move)
        self.assertAlmostEqual(2.0 / 3.0, result.win_probability)

    def test_deep_safe_chain_falls_back_without_recursion_error(self):
        hidden_cells = {(index, 0) for index in range(92)}
        uncertain = frozenset({(90, 0), (91, 0)})
        recursion_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(80)
            result = solve_endgame(
                [(uncertain, 1)],
                hidden_cells=hidden_cells,
                remaining_mines=1,
                width=92,
                height=1,
                max_search_nodes=5_000,
            )
        finally:
            sys.setrecursionlimit(recursion_limit)

        self.assertFalse(result.valid)
        self.assertTrue(result.budget_exhausted)
        self.assertIsNone(result.move)


if __name__ == "__main__":
    unittest.main()
