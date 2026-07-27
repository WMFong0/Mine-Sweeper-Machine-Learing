import itertools
import math
import random
import unittest

from minesweeper_ml.constraints import (
    build_constraint_context,
    build_constraint_boxes,
    count_legal_worlds,
    enumerate_constraint_box,
    infer_safe_click_outcomes,
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
        self.assertFalse(result.budget_exhausted)

    def test_counts_complete_worlds_across_boxes_and_unconstrained_cells(self):
        first, second = (0, 0), (1, 0)
        unconstrained = {(3, 0), (4, 0)}
        hidden_cells = {first, second, *unconstrained}
        context = build_constraint_context(
            [(frozenset({first, second}), 1)],
            hidden_cells=hidden_cells,
            model_mine_probabilities={cell: 0.5 for cell in hidden_cells},
            remaining_mines=2,
            prior_strength=0.0,
            max_search_nodes=1_000,
        )

        self.assertEqual(4, count_legal_worlds(context))

    def test_global_marginals_preserve_box_weights_and_combinatorics(self):
        first, second = (0, 0), (1, 0)
        outside = {(3, 0), (4, 0)}
        hidden_cells = {first, second, *outside}
        context = build_constraint_context(
            [(frozenset({first, second}), 1)],
            hidden_cells=hidden_cells,
            model_mine_probabilities={
                first: 0.1,
                second: 0.9,
                **{cell: 0.5 for cell in outside},
            },
            remaining_mines=2,
            prior_strength=1.0,
            max_search_nodes=1_000,
        )

        self.assertAlmostEqual(0.01 / 0.82, context.mine_probabilities[first])
        self.assertAlmostEqual(0.81 / 0.82, context.mine_probabilities[second])
        for cell in outside:
            self.assertAlmostEqual(0.5, context.mine_probabilities[cell])

    def test_total_search_budget_stops_conditional_inference(self):
        cells = {(0, 0), (1, 0), (3, 0), (4, 0)}

        result = infer_mine_probabilities(
            [
                (frozenset({(0, 0), (1, 0)}), 1),
                (frozenset({(3, 0), (4, 0)}), 1),
            ],
            hidden_cells=cells,
            model_mine_probabilities={cell: 0.5 for cell in cells},
            remaining_mines=2,
            prior_strength=1.0,
            max_search_nodes=100,
            max_total_search_nodes=4,
        )

        self.assertTrue(result.budget_exhausted)
        self.assertLessEqual(result.search_nodes, 4)
        self.assertEqual(0, result.overflowed_boxes)
        self.assertFalse(result.globally_coupled)

    def test_rejects_non_positive_total_search_budget(self):
        with self.assertRaises(ValueError):
            infer_mine_probabilities(
                [],
                hidden_cells=set(),
                model_mine_probabilities={},
                remaining_mines=0,
                prior_strength=1.0,
                max_search_nodes=100,
                max_total_search_nodes=0,
            )

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

    def test_safe_click_outcomes_preserve_correlated_neighbor_counts(self):
        candidate = (0, 0)
        neighbors = frozenset({(1, 0), (2, 0)})
        hidden_cells = {candidate, *neighbors}
        context = build_constraint_context(
            [(neighbors, 1)],
            hidden_cells=hidden_cells,
            model_mine_probabilities={cell: 0.5 for cell in hidden_cells},
            remaining_mines=None,
            prior_strength=1.0,
            max_search_nodes=1_000,
        )

        result = infer_safe_click_outcomes(
            context,
            candidate,
            neighbors,
        )

        self.assertTrue(result.consistent)
        self.assertEqual({1: 1.0}, result.outcome_probabilities)
        self.assertAlmostEqual(
            0.5,
            result.mine_probabilities_by_outcome[1][(1, 0)],
        )
        self.assertAlmostEqual(
            0.5,
            result.mine_probabilities_by_outcome[1][(2, 0)],
        )

    def test_safe_click_outcomes_honor_global_mine_budget(self):
        candidate = (0, 0)
        first = (1, 0)
        second = (2, 0)
        outside = (4, 0)
        hidden_cells = {candidate, first, second, outside}
        context = build_constraint_context(
            [(frozenset({first, second}), 1)],
            hidden_cells=hidden_cells,
            model_mine_probabilities={
                candidate: 0.5,
                first: 0.5,
                second: 0.5,
                outside: 0.9,
            },
            remaining_mines=1,
            prior_strength=1.0,
            max_search_nodes=1_000,
        )

        result = infer_safe_click_outcomes(
            context,
            candidate,
            frozenset({first, outside}),
        )

        self.assertTrue(result.consistent)
        self.assertEqual({0: 0.5, 1: 0.5}, result.outcome_probabilities)
        self.assertEqual(
            0.0,
            result.mine_probabilities_by_outcome[0][first],
        )
        self.assertEqual(
            1.0,
            result.mine_probabilities_by_outcome[1][first],
        )

    def test_exact_outcomes_match_randomized_brute_force(self):
        rng = random.Random(9182)
        cells = tuple((index, 0) for index in range(5))
        candidate = cells[0]
        query = frozenset(cells[1:4])

        for case in range(100):
            planted = (0, *(rng.randrange(2) for _ in cells[1:]))
            remaining_mines = sum(planted)
            constraints = [
                (frozenset(cells), remaining_mines),
                *[
                    (
                        frozenset(subset),
                        sum(
                            planted[cells.index(cell)]
                            for cell in subset
                        ),
                    )
                    for subset in (
                        rng.sample(cells, rng.randrange(2, len(cells)))
                        for _ in range(2)
                    )
                ],
            ]
            priors = {
                cell: rng.uniform(0.1, 0.9)
                for cell in cells
            }
            expected_weights = {}
            expected_mines = {}
            for assignment in itertools.product((0, 1), repeat=len(cells)):
                if (
                    assignment[0]
                    or sum(assignment) != remaining_mines
                    or any(
                        sum(
                            assignment[cells.index(cell)]
                            for cell in constrained_cells
                        )
                        != target
                        for constrained_cells, target in constraints
                    )
                ):
                    continue
                outcome = sum(
                    assignment[cells.index(cell)]
                    for cell in query
                )
                weight = (
                    math.prod(
                        priors[cell]
                        if assignment[index]
                        else 1.0 - priors[cell]
                        for index, cell in enumerate(cells)
                    )
                    ** 0.75
                )
                expected_weights[outcome] = (
                    expected_weights.get(outcome, 0.0) + weight
                )
                for index, cell in enumerate(cells):
                    if assignment[index]:
                        expected_mines[outcome, cell] = (
                            expected_mines.get((outcome, cell), 0.0)
                            + weight
                        )
            expected_total = sum(expected_weights.values())
            context = build_constraint_context(
                constraints,
                hidden_cells=set(cells),
                model_mine_probabilities=priors,
                remaining_mines=remaining_mines,
                prior_strength=0.75,
                max_search_nodes=10_000,
            )
            result = infer_safe_click_outcomes(
                context,
                candidate,
                query,
            )

            with self.subTest(case=case):
                self.assertTrue(result.consistent)
                for outcome, weight in expected_weights.items():
                    self.assertAlmostEqual(
                        weight / expected_total,
                        result.outcome_probabilities[outcome],
                    )
                    for cell in cells:
                        self.assertAlmostEqual(
                            expected_mines.get((outcome, cell), 0.0)
                            / weight,
                            result.mine_probabilities_by_outcome[
                                outcome
                            ][cell],
                        )


if __name__ == "__main__":
    unittest.main()
