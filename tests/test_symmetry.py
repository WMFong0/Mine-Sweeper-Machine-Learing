import unittest

import numpy as np

from minesweeper_ml.game import (
    inverse_transform_spatial,
    predict_spatial_maps,
    spatial_symmetries,
    transform_spatial,
)


class SpatialSymmetryTest(unittest.TestCase):
    def test_square_transforms_round_trip(self):
        values = np.arange(27).reshape(3, 3, 3)

        for symmetry in spatial_symmetries(3, 3):
            with self.subTest(symmetry=symmetry):
                self.assertTrue(
                    np.array_equal(
                        values,
                        inverse_transform_spatial(
                            transform_spatial(values, symmetry),
                            symmetry,
                        ),
                    )
                )

    def test_rectangular_transforms_preserve_shape_and_round_trip(self):
        values = np.arange(24).reshape(2, 4, 3)
        symmetries = spatial_symmetries(4, 2)

        self.assertEqual(4, len(symmetries))
        for symmetry in symmetries:
            with self.subTest(symmetry=symmetry):
                transformed = transform_spatial(values, symmetry)
                self.assertEqual(values.shape, transformed.shape)
                self.assertTrue(
                    np.array_equal(
                        values,
                        inverse_transform_spatial(
                            transformed,
                            symmetry,
                        ),
                    )
                )

    def test_ensemble_predicts_every_symmetry_in_one_batch(self):
        class EquivariantModel:
            output_names = ["safety", "value"]

            def __init__(self):
                self.calls = []

            def predict(self, batch, verbose=0):
                self.calls.append(batch.shape)
                return {
                    "safety": batch[..., :1],
                    "value": 1.0 - batch[..., :1],
                }

        model = EquivariantModel()
        board = np.arange(9, dtype=np.float32).reshape(1, 3, 3, 1) / 8

        predictions = predict_spatial_maps(
            model,
            board,
            symmetry_ensemble=True,
        )

        self.assertEqual([(8, 3, 3, 1)], model.calls)
        self.assertTrue(np.allclose(board[0], predictions["safety"]))
        self.assertTrue(
            np.allclose(1.0 - board[0], predictions["value"])
        )


if __name__ == "__main__":
    unittest.main()
