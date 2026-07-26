import unittest

from minesweeper_ml.constraints import (
    build_constraint_boxes,
    enumerate_constraint_box,
    infer_mine_probabilities,
)


class ConstraintBoxTest(unittest.TestCase):
    def test_builds_boxes_from_constraints_that_share_cells(self):
        constraints = [
            (frozenset({(0, 0), (1, 0)}), 1),
            (frozenset({(1, 0), (2, 0)}), 1),
            (frozenset({(4, 0), (5, 0)}), 1),
        ]

        boxes = build_constraint_boxes(constraints)

        self.assertEqual(
            [
                frozenset({(0, 0), (1, 0), (2, 0)}),
                frozenset({(4, 0), (5, 0)}),
            ],
            [box.cells for box in boxes],
        )
        self.assertEqual([2, 1], [len(box.constraints) for box in boxes])

    def test_weights_only_assignments_that_satisfy_every_constraint(self):
        first = (0, 0)
        middle = (1, 0)
        last = (2, 0)
        box = build_constraint_boxes(
            [
                (frozenset({first, middle}), 1),
                (frozenset({middle, last}), 1),
            ]
        )[0]

        result = enumerate_constraint_box(
            box,
            {
                first: 0.1,
                middle: 0.4,
                last: 0.1,
            },
            prior_strength=1.0,
            max_search_nodes=1_000,
        )

        self.assertFalse(result.overflowed)
        self.assertEqual(2, result.assignment_count)
        self.assertAlmostEqual(0.324, result.weight_by_mine_count[1])
        self.assertAlmostEqual(0.006, result.weight_by_mine_count[2])
        self.assertAlmostEqual(0.006 / 0.33, result.mine_probabilities[first])
        self.assertAlmostEqual(0.324 / 0.33, result.mine_probabilities[middle])
        self.assertAlmostEqual(0.006 / 0.33, result.mine_probabilities[last])

    def test_global_mine_budget_couples_a_box_with_unconstrained_cells(self):
        first = (0, 0)
        middle = (1, 0)
        last = (2, 0)
        unconstrained = (4, 0)

        result = infer_mine_probabilities(
            [
                (frozenset({first, middle}), 1),
                (frozenset({middle, last}), 1),
            ],
            hidden_cells={first, middle, last, unconstrained},
            model_mine_probabilities={
                first: 0.5,
                middle: 0.5,
                last: 0.5,
            },
            remaining_mines=1,
            prior_strength=1.0,
            max_search_nodes=1_000,
        )

        self.assertTrue(result.consistent)
        self.assertTrue(result.globally_coupled)
        self.assertEqual(0, result.overflowed_boxes)
        self.assertAlmostEqual(0.0, result.mine_probabilities[first])
        self.assertAlmostEqual(1.0, result.mine_probabilities[middle])
        self.assertAlmostEqual(0.0, result.mine_probabilities[last])
        self.assertAlmostEqual(0.0, result.mine_probabilities[unconstrained])

    def test_overflow_keeps_completed_local_boxes_without_global_coupling(self):
        solved = (0, 0)
        large_cells = frozenset({(2, 0), (3, 0), (4, 0)})

        result = infer_mine_probabilities(
            [
                (frozenset({solved}), 0),
                (large_cells, 1),
            ],
            hidden_cells={solved, *large_cells},
            model_mine_probabilities={
                solved: 0.5,
                **{cell: 0.5 for cell in large_cells},
            },
            remaining_mines=1,
            prior_strength=1.0,
            max_search_nodes=3,
        )

        self.assertTrue(result.consistent)
        self.assertFalse(result.globally_coupled)
        self.assertEqual(1, result.overflowed_boxes)
        self.assertEqual({solved: 0.0}, result.mine_probabilities)

    def test_inconsistent_constraints_return_no_probabilities(self):
        cell = (0, 0)

        result = infer_mine_probabilities(
            [
                (frozenset({cell}), 0),
                (frozenset({cell}), 1),
            ],
            hidden_cells={cell},
            model_mine_probabilities={cell: 0.5},
            remaining_mines=1,
            prior_strength=1.0,
            max_search_nodes=100,
        )

        self.assertFalse(result.consistent)
        self.assertFalse(result.globally_coupled)
        self.assertEqual({}, result.mine_probabilities)

    def test_extreme_prior_strength_does_not_underflow_every_assignment(self):
        cell = (0, 0)
        box = build_constraint_boxes(
            [(frozenset({cell}), 1)]
        )[0]

        result = enumerate_constraint_box(
            box,
            {cell: 0.5},
            prior_strength=2_000.0,
            max_search_nodes=100,
        )

        self.assertEqual(1, result.assignment_count)
        self.assertGreater(result.weight_by_mine_count[1], 0.0)
        self.assertEqual({cell: 1.0}, result.mine_probabilities)

    def test_nonzero_empty_constraint_is_inconsistent(self):
        result = infer_mine_probabilities(
            [(frozenset(), 1)],
            hidden_cells=set(),
            model_mine_probabilities={},
            remaining_mines=0,
            prior_strength=1.0,
            max_search_nodes=100,
        )

        self.assertFalse(result.consistent)
        self.assertFalse(result.globally_coupled)
        self.assertEqual({}, result.mine_probabilities)


if __name__ == "__main__":
    unittest.main()
