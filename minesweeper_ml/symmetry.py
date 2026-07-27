from __future__ import annotations

from collections.abc import Mapping, Sequence


_SQUARE_SYMMETRIES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
)
_RECTANGULAR_SYMMETRIES = (
    "identity",
    "rot180",
    "flip_horizontal",
    "flip_vertical",
)
_INVERSES = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_horizontal": "flip_horizontal",
    "flip_vertical": "flip_vertical",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


def spatial_symmetries(width: int, height: int) -> tuple[str, ...]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero.")
    return (
        _SQUARE_SYMMETRIES
        if width == height
        else _RECTANGULAR_SYMMETRIES
    )


def transform_spatial(values, symmetry: str):
    import numpy as np

    if symmetry not in _INVERSES:
        raise ValueError(f"Unknown spatial symmetry: {symmetry}")
    return {
        "identity": lambda: np.asarray(values),
        "rot90": lambda: np.rot90(values, 1, axes=(0, 1)),
        "rot180": lambda: np.rot90(values, 2, axes=(0, 1)),
        "rot270": lambda: np.rot90(values, 3, axes=(0, 1)),
        "flip_horizontal": lambda: np.flip(values, axis=1),
        "flip_vertical": lambda: np.flip(values, axis=0),
        "transpose": lambda: np.swapaxes(values, 0, 1),
        "anti_transpose": lambda: np.rot90(
            np.swapaxes(values, 0, 1),
            2,
            axes=(0, 1),
        ),
    }[symmetry]()


def inverse_transform_spatial(values, symmetry: str):
    if symmetry not in _INVERSES:
        raise ValueError(f"Unknown spatial symmetry: {symmetry}")
    return transform_spatial(values, _INVERSES[symmetry])


def normalize_prediction_outputs(model, prediction) -> dict[str, object]:
    if isinstance(prediction, Mapping):
        return dict(prediction)
    if isinstance(prediction, Sequence) and not isinstance(
        prediction,
        (str, bytes),
    ):
        names = tuple(
            getattr(
                model,
                "output_names",
                ("safety", "value")[: len(prediction)],
            )
        )
        return dict(zip(names, prediction))
    return {"safety": prediction}


def predict_spatial_maps(
    model,
    board_batch,
    *,
    symmetry_ensemble: bool,
) -> dict[str, object]:
    import numpy as np

    board = np.asarray(board_batch)
    if board.ndim != 4 or board.shape[0] != 1:
        raise ValueError("board_batch must have shape (1, height, width, channels).")
    if not symmetry_ensemble:
        return {
            name: np.asarray(values)[0]
            for name, values in normalize_prediction_outputs(
                model,
                model.predict(board, verbose=0),
            ).items()
        }

    height, width = board.shape[1:3]
    symmetries = spatial_symmetries(width, height)
    transformed = np.stack(
        [
            transform_spatial(board[0], symmetry)
            for symmetry in symmetries
        ]
    )
    predictions = normalize_prediction_outputs(
        model,
        model.predict(transformed, verbose=0),
    )
    return {
        name: np.mean(
            np.stack(
                [
                    inverse_transform_spatial(values[index], symmetry)
                    for index, symmetry in enumerate(symmetries)
                ]
            ),
            axis=0,
        )
        for name, raw_values in predictions.items()
        for values in (np.asarray(raw_values),)
    }


def augment_spatial_arrays(*arrays):
    import numpy as np

    if not arrays:
        raise ValueError("at least one spatial array is required.")
    normalized = tuple(np.asarray(values) for values in arrays)
    if any(
        values.ndim < 3 or values.shape[:3] != normalized[0].shape[:3]
        for values in normalized
    ):
        raise ValueError("spatial arrays must share batch, height, and width.")
    _, height, width = normalized[0].shape[:3]
    symmetries = spatial_symmetries(width, height)
    return tuple(
        np.stack(
            [
                transform_spatial(values[index], symmetry)
                for index in range(len(values))
                for symmetry in symmetries
            ]
        )
        for values in normalized
    )
