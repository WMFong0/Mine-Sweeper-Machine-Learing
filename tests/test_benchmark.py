import unittest

from minesweeper_ml.benchmark import (
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


if __name__ == "__main__":
    unittest.main()
