import unittest

import numpy as np

from minesweeper_ml.training import prepare_multitask_training_arrays


class MultitaskTrainingTest(unittest.TestCase):
    def test_symmetry_augmentation_keeps_targets_and_masks_aligned(self):
        features = np.zeros((1, 2, 2, 10), dtype=np.float32)
        features[0, 0, 1, 0] = 1
        safety = np.zeros((1, 2, 2, 1), dtype=np.float32)
        safety[0, 0, 1, 0] = 1
        value = np.zeros_like(safety)
        value[0, 1, 0, 0] = 1
        safety_mask = safety.copy()
        value_mask = value.copy()

        augmented, targets, weights = prepare_multitask_training_arrays(
            features,
            {"safety": safety, "value": value},
            {"safety": safety_mask, "value": value_mask},
            np.asarray([2.0], dtype=np.float32),
            augment=True,
        )

        self.assertEqual(8, len(augmented))
        self.assertTrue(
            np.array_equal(
                augmented[..., :1] > 0,
                targets["safety"] > 0,
            )
        )
        self.assertTrue(
            np.array_equal(
                targets["safety"][..., 0] > 0,
                weights["safety"] > 0,
            )
        )
        self.assertEqual(8, np.count_nonzero(weights["value"]))
        self.assertTrue(
            np.all(weights["value"][weights["value"] > 0] == 2.0)
        )


if __name__ == "__main__":
    unittest.main()
