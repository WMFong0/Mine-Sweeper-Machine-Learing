import unittest

from minesweeper_ml.training import (
    evaluate_bot_factory,
    exact_mcnemar_p_value,
    paired_win_statistics,
)


class PairedBenchmarkTest(unittest.TestCase):
    def test_exact_mcnemar_uses_two_sided_binomial_tail(self):
        self.assertAlmostEqual(
            0.001953125,
            exact_mcnemar_p_value(10, 0),
        )
        self.assertEqual(1.0, exact_mcnemar_p_value(4, 4))
        self.assertEqual(1.0, exact_mcnemar_p_value(0, 0))

    def test_paired_statistics_count_completed_maps_only(self):
        result = paired_win_statistics(
            [True, True, False, False],
            [True, False, True, False],
        )

        self.assertEqual(1, result["baseline_only"])
        self.assertEqual(1, result["candidate_only"])
        self.assertEqual(0.5, result["baseline_win_rate"])
        self.assertEqual(0.5, result["candidate_win_rate"])
        self.assertEqual(1.0, result["mcnemar_exact_p"])

    def test_evaluation_reports_endgame_search_metrics(self):
        class MetricBot:
            last_move_strategy = None
            strategy_stats = {
                "lookahead_decisions": 3,
                "lookahead_search_nodes": 234,
                "lookahead_budget_exhaustions": 1,
                "endgame_decisions": 2,
                "endgame_search_nodes": 123,
                "endgame_evaluated_states": 17,
                "endgame_budget_exhaustions": 1,
                "consensus_decisions": 7,
            }

        summary, _ = evaluate_bot_factory(
            lambda width, height, mines: MetricBot(),
            width=2,
            height=1,
            mine_count=1,
            layouts=[((1, 0),)],
        )

        self.assertEqual(2, summary["endgame_decisions"])
        self.assertEqual(123, summary["endgame_search_nodes"])
        self.assertEqual(17, summary["endgame_evaluated_states"])
        self.assertEqual(1, summary["endgame_budget_exhaustions"])
        self.assertEqual(3, summary["lookahead_decisions"])
        self.assertEqual(234, summary["lookahead_search_nodes"])
        self.assertEqual(1, summary["lookahead_budget_exhaustions"])
        self.assertEqual(7, summary["consensus_decisions"])


if __name__ == "__main__":
    unittest.main()
