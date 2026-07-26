import unittest

from minesweeper_ml.lookahead import (
    evaluate_safe_click,
    poisson_binomial_distribution,
)


class PosteriorLookaheadTest(unittest.TestCase):
    def test_poisson_binomial_distribution(self):
        self.assertEqual(
            {0: 0.375, 1: 0.5, 2: 0.125},
            poisson_binomial_distribution([0.25, 0.5]),
        )

    def test_safe_click_scores_expected_new_deductions(self):
        candidate = (0, 0)
        neighbors = frozenset({(1, 0), (2, 0)})
        hidden_cells = {candidate, *neighbors}
        priors = {cell: 0.5 for cell in hidden_cells}

        result = evaluate_safe_click(
            candidate,
            hidden_neighbors=neighbors,
            has_known_mine_neighbor=False,
            constraints=[],
            hidden_cells=hidden_cells,
            model_mine_probabilities=priors,
            remaining_mines=None,
            prior_strength=1.0,
            max_constraint_nodes=1_000,
            max_total_search_nodes=1_000,
            min_outcome_probability=1e-6,
        )

        self.assertTrue(result.valid)
        self.assertFalse(result.budget_exhausted)
        self.assertAlmostEqual(1.0, result.expected_forced_cells)
        self.assertGreater(result.expected_entropy_reduction, 0.0)
        self.assertAlmostEqual(0.25, result.zero_region_probability)

    def test_impossible_clue_outcomes_are_ignored(self):
        candidate = (0, 0)
        neighbors = frozenset({(1, 0), (2, 0)})
        hidden_cells = {candidate, *neighbors}
        priors = {cell: 0.5 for cell in hidden_cells}

        result = evaluate_safe_click(
            candidate,
            hidden_neighbors=neighbors,
            has_known_mine_neighbor=False,
            constraints=[(neighbors, 1)],
            hidden_cells=hidden_cells,
            model_mine_probabilities=priors,
            remaining_mines=None,
            prior_strength=1.0,
            max_constraint_nodes=1_000,
            max_total_search_nodes=1_000,
            min_outcome_probability=1e-6,
        )

        self.assertTrue(result.valid)
        self.assertAlmostEqual(0.0, result.expected_forced_cells)
        self.assertAlmostEqual(0.0, result.expected_entropy_reduction)
        self.assertAlmostEqual(0.0, result.zero_region_probability)

    def test_shared_budget_exhaustion_returns_invalid_evaluation(self):
        candidate = (0, 0)

        result = evaluate_safe_click(
            candidate,
            hidden_neighbors=frozenset({(1, 0)}),
            has_known_mine_neighbor=False,
            constraints=[],
            hidden_cells={candidate, (1, 0)},
            model_mine_probabilities={
                candidate: 0.5,
                (1, 0): 0.5,
            },
            remaining_mines=None,
            prior_strength=1.0,
            max_constraint_nodes=1_000,
            max_total_search_nodes=1,
            min_outcome_probability=1e-6,
        )

        self.assertFalse(result.valid)
        self.assertTrue(result.budget_exhausted)
        self.assertLessEqual(result.search_nodes, 1)

    def test_per_box_overflow_is_not_reported_as_budget_exhaustion(self):
        candidate = (0, 0)

        result = evaluate_safe_click(
            candidate,
            hidden_neighbors=frozenset({(1, 0)}),
            has_known_mine_neighbor=False,
            constraints=[],
            hidden_cells={candidate, (1, 0)},
            model_mine_probabilities={
                candidate: 0.5,
                (1, 0): 0.5,
            },
            remaining_mines=None,
            prior_strength=1.0,
            max_constraint_nodes=1,
            max_total_search_nodes=100,
            min_outcome_probability=1e-6,
        )

        self.assertFalse(result.valid)
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(1, result.search_nodes)

    def test_zero_probability_outcomes_never_consume_search_budget(self):
        candidate = (0, 0)
        neighbor = (1, 0)
        options = {
            "hidden_neighbors": frozenset({neighbor}),
            "has_known_mine_neighbor": False,
            "constraints": [(frozenset({neighbor}), 0)],
            "hidden_cells": {candidate, neighbor},
            "model_mine_probabilities": {
                candidate: 0.5,
                neighbor: 0.5,
            },
            "remaining_mines": None,
            "prior_strength": 1.0,
            "max_constraint_nodes": 1_000,
            "max_total_search_nodes": 1_000,
        }

        zero_threshold = evaluate_safe_click(
            candidate,
            min_outcome_probability=0.0,
            **options,
        )
        positive_threshold = evaluate_safe_click(
            candidate,
            min_outcome_probability=1e-6,
            **options,
        )

        self.assertTrue(zero_threshold.valid)
        self.assertEqual(
            positive_threshold.search_nodes,
            zero_threshold.search_nodes,
        )

    def test_known_neighboring_mine_prevents_zero_region_score(self):
        candidate = (0, 0)

        result = evaluate_safe_click(
            candidate,
            hidden_neighbors=frozenset({(1, 0)}),
            has_known_mine_neighbor=True,
            constraints=[],
            hidden_cells={candidate, (1, 0)},
            model_mine_probabilities={
                candidate: 0.5,
                (1, 0): 0.5,
            },
            remaining_mines=None,
            prior_strength=1.0,
            max_constraint_nodes=1_000,
            max_total_search_nodes=1_000,
            min_outcome_probability=1e-6,
        )

        self.assertTrue(result.valid)
        self.assertEqual(0.0, result.zero_region_probability)


if __name__ == "__main__":
    unittest.main()
