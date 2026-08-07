from __future__ import annotations

from minesweeper_ml.game import (
    augment_spatial_arrays,
    spatial_symmetries,
)


def prepare_multitask_training_arrays(
    features,
    targets,
    masks,
    trajectory_weights,
    *,
    augment: bool,
):
    import numpy as np

    features = np.asarray(features, dtype=np.float32)
    safety_targets = np.asarray(targets["safety"], dtype=np.float32)
    value_targets = np.asarray(targets["value"], dtype=np.float32)
    safety_masks = np.asarray(masks["safety"], dtype=np.float32)
    value_masks = np.asarray(masks["value"], dtype=np.float32)
    trajectory_weights = np.asarray(
        trajectory_weights,
        dtype=np.float32,
    )
    if augment:
        (
            features,
            safety_targets,
            value_targets,
            safety_masks,
            value_masks,
        ) = augment_spatial_arrays(
            features,
            safety_targets,
            value_targets,
            safety_masks,
            value_masks,
        )
        trajectory_weights = np.repeat(
            trajectory_weights,
            len(
                spatial_symmetries(
                    features.shape[2],
                    features.shape[1],
                )
            ),
        )
    safety_weights = _balanced_cell_weights(
        safety_targets,
        safety_masks,
    ) * trajectory_weights[:, None, None]
    value_weights = (
        value_masks[..., 0]
        * trajectory_weights[:, None, None]
    )
    return (
        features,
        {
            "safety": safety_targets,
            "value": value_targets,
        },
        {
            "safety": safety_weights.astype(np.float32),
            "value": value_weights.astype(np.float32),
        },
    )


def _balanced_cell_weights(targets, masks):
    import numpy as np

    active = masks[..., 0] > 0
    labels = targets[..., 0][active]
    mines = int(np.count_nonzero(labels < 0.5))
    safe = int(np.count_nonzero(labels >= 0.5))
    if not mines or not safe:
        return masks[..., 0].astype(np.float32)
    total = mines + safe
    return (
        np.where(
            targets[..., 0] >= 0.5,
            total / (2 * safe),
            total / (2 * mines),
        )
        * masks[..., 0]
    ).astype(np.float32)


def fit_safety_model(
    train_dataset,
    test_dataset,
    *,
    width: int,
    height: int,
    architecture: str,
    epochs: int,
    batch_size: int = 32,
    augment: bool = True,
):
    import numpy as np

    from minesweeper_ml.training.data import dataset_to_arrays
    from minesweeper_ml.training.models import build_cnn_model

    x_train, y_train, train_masks = dataset_to_arrays(train_dataset)
    x_test, y_test, test_masks = dataset_to_arrays(test_dataset)
    if augment:
        x_train, y_train, train_masks = augment_spatial_arrays(
            x_train,
            y_train,
            train_masks,
        )
    model = build_cnn_model(
        width,
        height,
        architecture=architecture,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=_balanced_cell_weights(
            y_train,
            train_masks,
        ),
        validation_data=(
            x_test,
            y_test,
            _balanced_cell_weights(y_test, test_masks),
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    metrics = model.evaluate(
        x_test,
        y_test,
        sample_weight=_balanced_cell_weights(y_test, test_masks),
        verbose=0,
        return_dict=True,
    )
    return model, {
        name: float(value)
        for name, value in metrics.items()
        if np.isscalar(value)
    }


def fit_multitask_model(
    train_dataset,
    test_dataset,
    *,
    width: int,
    height: int,
    architecture: str,
    epochs: int,
    batch_size: int = 32,
    augment: bool = True,
):
    import numpy as np

    from minesweeper_ml.training.data import multitask_dataset_to_arrays
    from minesweeper_ml.training.models import build_cnn_model

    train_arrays = multitask_dataset_to_arrays(train_dataset)
    test_arrays = multitask_dataset_to_arrays(test_dataset)
    x_train, y_train, train_weights = (
        prepare_multitask_training_arrays(
            *train_arrays,
            augment=augment,
        )
    )
    x_test, y_test, test_weights = (
        prepare_multitask_training_arrays(
            *test_arrays,
            augment=False,
        )
    )
    model = build_cnn_model(
        width,
        height,
        architecture=architecture,
        value_head=True,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=train_weights,
        validation_data=(x_test, y_test, test_weights),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    metrics = model.evaluate(
        x_test,
        y_test,
        sample_weight=test_weights,
        verbose=0,
        return_dict=True,
    )
    return model, {
        name: float(value)
        for name, value in metrics.items()
        if np.isscalar(value)
    }


def train_upgrade_pipeline(
    *,
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    num_games: int = 50_000,
    epochs: int = 50,
    seed: int = 42,
    development_games: int = 500,
    teacher_model=None,
    architectures: tuple[str, ...] = (
        "local_3x3",
        "paper_5x5",
    ),
):
    import random

    from minesweeper_ml.training.benchmark import evaluate_bot_factory
    from minesweeper_ml.bot.bots import MLMinesweeperBot
    from minesweeper_ml.training.data import (
        generate_training_data,
        split_dataset_by_game,
    )
    from minesweeper_ml.game import MinesweeperGame

    if num_games < 2 or development_games <= 0 or epochs <= 0:
        raise ValueError("training and development sizes must be positive.")
    warmup_report = None
    if teacher_model is None:
        warmup = generate_training_data(
            max(2, num_games // 5),
            width,
            height,
            mine_count,
            seed=seed,
            verbose=True,
            value_rollout_candidates=0,
        )
        warmup_train, warmup_test = split_dataset_by_game(
            warmup,
            seed=seed,
        )
        teacher_model, warmup_report = fit_safety_model(
            warmup_train,
            warmup_test,
            width=width,
            height=height,
            architecture="local_3x3",
            epochs=max(1, epochs // 3),
        )

    def teacher_factory(
        bot_width: int,
        bot_height: int,
        bot_mines: int,
    ):
        return MLMinesweeperBot(
            bot_width,
            bot_height,
            teacher_model,
            mine_count=bot_mines,
            exact_lookahead=True,
            symmetry_ensemble=True,
            use_candidate_value=False,
        )

    on_policy = generate_training_data(
        num_games,
        width,
        height,
        mine_count,
        seed=seed + 1,
        verbose=True,
        bot_factory=teacher_factory,
        value_rollout_candidates=2,
    )
    train_dataset, test_dataset = split_dataset_by_game(
        on_policy,
        seed=seed,
    )
    rng = random.Random(seed + 2)
    development_layouts = [
        tuple(
            MinesweeperGame(
                width,
                height,
                mine_count,
                seed=rng.randrange(2**63),
                first_safe=True,
            ).mine_locations
        )
        for _ in range(development_games)
    ]
    models = {}
    candidates = []
    for architecture in architectures:
        model, cell_metrics = fit_multitask_model(
            train_dataset,
            test_dataset,
            width=width,
            height=height,
            architecture=architecture,
            epochs=epochs,
        )
        models[architecture] = model
        for use_value, value_margin in (
            (False, 0.01),
            (True, 0.0),
            (True, 0.01),
            (True, 0.05),
        ):
            bot_options = {
                "mine_count": mine_count,
                "exact_lookahead": True,
                "symmetry_ensemble": True,
                "use_candidate_value": use_value,
                "value_tie_margin": value_margin,
            }

            def candidate_factory(
                bot_width: int,
                bot_height: int,
                bot_mines: int,
                *,
                candidate_model=model,
                options=bot_options,
            ):
                return MLMinesweeperBot(
                    bot_width,
                    bot_height,
                    candidate_model,
                    **options,
                )

            development, _ = evaluate_bot_factory(
                candidate_factory,
                width=width,
                height=height,
                mine_count=mine_count,
                layouts=development_layouts,
            )
            candidates.append(
                {
                    "architecture": architecture,
                    "use_candidate_value": use_value,
                    "value_tie_margin": value_margin,
                    "cell_metrics": cell_metrics,
                    "development": development,
                    "bot_options": bot_options,
                }
            )
    selected = max(candidates, key=_development_candidate_key)
    return (
        models[selected["architecture"]],
        {
            "warmup_metrics": warmup_report,
            "sample_count": len(on_policy),
            "candidates": candidates,
            "selected": selected,
        },
        teacher_model,
    )


def _development_candidate_key(candidate: dict):
    development = candidate["development"]
    return (
        development["won"],
        development["average_safe_moves"],
        -development["median_uncertain_move_ms"],
        candidate["architecture"] == "local_3x3",
        candidate["use_candidate_value"],
        -candidate["value_tie_margin"],
    )
