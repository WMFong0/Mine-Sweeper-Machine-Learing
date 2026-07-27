from collections import Counter
import unittest

from minesweeper_ml.data import (
    build_sample_weights,
    calculate_class_weights,
    encode_board_state,
    encode_board_features,
    encode_ground_truth_map,
    evaluate_candidate_values,
    frontier_mask,
    generate_training_data,
    multitask_dataset_to_arrays,
    run_single_bot_game_and_collect_data,
    split_dataset_by_game,
)
from minesweeper_ml.game import GameState


class TrainingDataTest(unittest.TestCase):
    def test_encode_board_state_uses_notebook_numeric_mapping(self):
        board_state = [
            ["-", 0, 2],
            [8, "M", 1],
        ]

        self.assertEqual([0, 1, 3, 9, 10, 2], encode_board_state(board_state))

    def test_encode_ground_truth_map_marks_safe_cells_with_one_and_mines_with_zero(self):
        ground_truth_map = [
            [0, "M"],
            [2, 1],
        ]

        self.assertEqual([1, 0, 1, 1], encode_ground_truth_map(ground_truth_map))

    def test_dataset_to_arrays_rejects_an_empty_dataset(self):
        from minesweeper_ml.data import dataset_to_arrays

        with self.assertRaises(ValueError):
            dataset_to_arrays([])

    def test_frontier_mask_marks_only_hidden_neighbors_of_revealed_cells(self):
        board_state = [
            [1, "-", "-"],
            ["-", "-", "-"],
        ]

        self.assertEqual(
            [
                [0, 1, 0],
                [1, 1, 0],
            ],
            frontier_mask(board_state),
        )

    def test_seeded_generation_is_reproducible_and_phase_capped(self):
        first = generate_training_data(20, 5, 5, 5, seed=123)
        second = generate_training_data(20, 5, 5, 5, seed=123)

        self.assertEqual(first, second)
        self.assertTrue(first)

        phase_counts = Counter((sample["game_id"], sample["phase"]) for sample in first)
        self.assertTrue(all(count <= 4 for count in phase_counts.values()))
        self.assertGreater(len({sample["phase"] for sample in first}), 1)
        self.assertTrue(
            all(sample["policy"] != "oracle-safe" for sample in first)
        )

    def test_generated_games_open_a_safe_zero_zero_first(self):
        samples = generate_training_data(8, 5, 5, 5, seed=321)

        self.assertTrue(samples)
        self.assertTrue(all(sample["ground_truth_map"][0][0] != "M" for sample in samples))
        self.assertTrue(all(sample["board_state"][0][0] != "-" for sample in samples))

    def test_on_policy_generation_does_not_replace_a_mined_move(self):
        class MinedMoveBot:
            last_move_strategy = "box"
            last_decision = None

            def get_next_move(self, visible_map):
                return (1, 0)

        samples, result = run_single_bot_game_and_collect_data(
            width=3,
            height=1,
            mine_count=1,
            mine_locations=[(1, 0)],
            bot_factory=lambda *_: MinedMoveBot(),
        )

        self.assertEqual(GameState.LOST, result)
        self.assertEqual((1, 0), samples[0]["chosen_move"])
        self.assertFalse(samples[0]["is_safe"])
        self.assertEqual(2.0, samples[0]["trajectory_weight"])

    def test_close_probability_losses_receive_both_weights(self):
        class Decision:
            probability_gap = 0.002

        class CloseLossBot:
            last_move_strategy = "box"
            last_decision = Decision()

            def get_next_move(self, visible_map):
                return (1, 0)

        samples, _ = run_single_bot_game_and_collect_data(
            width=3,
            height=1,
            mine_count=1,
            mine_locations=[(1, 0)],
            bot_factory=lambda *_: CloseLossBot(),
            loss_weight=3.0,
            close_probability_weight=4.0,
            close_probability_threshold=0.01,
        )

        self.assertEqual(12.0, samples[0]["trajectory_weight"])

    def test_candidate_value_labels_mask_mines_and_score_full_wins(self):
        class FinishBot:
            def get_next_move(self, visible_map):
                return (1, 0)

        from minesweeper_ml.game import MinesweeperGame

        game = MinesweeperGame(
            width=2,
            height=2,
            mine_count=1,
            mine_locations=[(1, 1)],
        )
        game.open_cell(0, 0)

        targets, mask = evaluate_candidate_values(
            game,
            [(0, 1), (1, 1)],
            bot_factory=lambda *_: FinishBot(),
            max_candidates=2,
            max_moves=10,
        )

        self.assertEqual(1, targets[1][0])
        self.assertEqual(1, mask[1][0])
        self.assertEqual(0, targets[1][1])
        self.assertEqual(0, mask[1][1])

    def test_multitask_arrays_include_value_targets_and_trajectory_weights(self):
        sample = {
            "board_state": [[1, "-"], ["-", "-"]],
            "ground_truth_map": [[1, "M"], [0, 1]],
            "training_mask": [[0, 1], [1, 1]],
            "value_targets": [[0, 0], [1, 0]],
            "value_mask": [[0, 0], [1, 0]],
            "trajectory_weight": 3.0,
        }

        features, targets, masks, trajectory_weights = (
            multitask_dataset_to_arrays([sample])
        )

        self.assertEqual((1, 2, 2, 10), features.shape)
        self.assertEqual({"safety", "value"}, set(targets))
        self.assertEqual(1, targets["value"][0, 1, 0, 0])
        self.assertEqual(1, masks["value"][0, 1, 0, 0])
        self.assertEqual([3.0], trajectory_weights.tolist())

    def test_board_features_are_one_hot_spatial_channels(self):
        features = encode_board_features([["-", 0, 2]])

        self.assertEqual(10, len(features[0][0]))
        self.assertEqual(1, features[0][0][0])
        self.assertEqual(1, features[0][1][1])
        self.assertEqual(1, features[0][2][3])
        self.assertTrue(all(sum(cell_features) == 1 for cell_features in features[0]))

    def test_dataset_arrays_include_spatial_labels_and_masks(self):
        from minesweeper_ml.data import dataset_to_arrays

        sample = {
            "board_state": [[1, "-"], ["-", "-"]],
            "ground_truth_map": [[1, "M"], [0, 1]],
            "training_mask": [[0, 1], [1, 1]],
        }

        features, labels, masks = dataset_to_arrays([sample])

        self.assertEqual((1, 2, 2, 10), features.shape)
        self.assertEqual((1, 2, 2, 1), labels.shape)
        self.assertEqual((1, 2, 2, 1), masks.shape)
        self.assertEqual(0, labels[0, 0, 1, 0])
        self.assertEqual(1, masks[0, 1, 1, 0])

    def test_split_dataset_by_game_has_no_game_overlap(self):
        dataset = [{"game_id": game_id} for game_id in range(10) for _ in range(2)]

        train, test = split_dataset_by_game(dataset, seed=7)

        train_ids = {sample["game_id"] for sample in train}
        test_ids = {sample["game_id"] for sample in test}
        self.assertTrue(train_ids)
        self.assertTrue(test_ids)
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(set(range(10)), train_ids | test_ids)

    def test_split_keeps_duplicate_mine_layouts_on_one_side(self):
        dataset = [
            {"game_id": 0, "layout_id": ((1, 0),)},
            {"game_id": 1, "layout_id": ((1, 0),)},
            {"game_id": 2, "layout_id": ((0, 1),)},
            {"game_id": 3, "layout_id": ((1, 1),)},
        ]

        train, test = split_dataset_by_game(dataset, seed=5)

        train_layouts = {sample["layout_id"] for sample in train}
        test_layouts = {sample["layout_id"] for sample in test}
        self.assertTrue(train_layouts.isdisjoint(test_layouts))

    def test_sample_weights_balance_safe_and_mined_frontier_cells(self):
        import numpy as np

        labels = np.array([[[[1], [1]], [[1], [0]]]], dtype=np.float32)
        masks = np.array([[[[0], [1]], [[1], [1]]]], dtype=np.float32)

        class_weights = calculate_class_weights(labels, masks)
        sample_weights = build_sample_weights(labels, masks, class_weights)

        self.assertAlmostEqual(3 / 4, class_weights[1])
        self.assertAlmostEqual(3 / 2, class_weights[0])
        self.assertEqual((1, 2, 2), sample_weights.shape)
        self.assertEqual(0.0, sample_weights[0, 0, 0])
        self.assertAlmostEqual(3 / 2, sample_weights[0, 1, 1])

    def test_class_weights_reject_a_dataset_without_mined_frontier_cells(self):
        import numpy as np

        labels = np.ones((1, 2, 2, 1), dtype=np.float32)
        masks = np.ones_like(labels)

        with self.assertRaises(ValueError):
            calculate_class_weights(labels, masks)
