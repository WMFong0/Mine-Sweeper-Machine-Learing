import unittest

from minesweeper_ml.consensus import average_algorithm_ranks


class AverageAlgorithmRanksTest(unittest.TestCase):
    def test_majority_ranking_wins_after_cross_signal_normalization(self):
        candidates = ((0, 0), (1, 0), (2, 0))

        result = average_algorithm_ranks(
            candidates,
            (
                {(0, 0): 0.99, (1, 0): 0.80, (2, 0): 0.10},
                {(0, 0): 1.0, (1, 0): 100.0, (2, 0): 0.0},
                {(0, 0): 0.0, (1, 0): 2.0, (2, 0): 1.0},
            ),
        )

        self.assertEqual((1, 0), max(result, key=result.get))
        self.assertAlmostEqual(0.5, result[(0, 0)])
        self.assertAlmostEqual(5.0 / 6.0, result[(1, 0)])
        self.assertAlmostEqual(1.0 / 6.0, result[(2, 0)])

    def test_ties_share_rank_and_constant_signal_abstains(self):
        candidates = ((0, 0), (1, 0), (2, 0))

        result = average_algorithm_ranks(
            candidates,
            (
                dict.fromkeys(candidates, 0.5),
                {(0, 0): 4.0, (1, 0): 4.0, (2, 0): 1.0},
            ),
        )

        self.assertEqual(
            {(0, 0): 0.75, (1, 0): 0.75, (2, 0): 0.0},
            result,
        )


if __name__ == "__main__":
    unittest.main()
