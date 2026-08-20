from __future__ import annotations

from minesweeper_ml.game import (
    augment_spatial_arrays,
    spatial_symmetries,
)

CURRICULUM_STAGES = (
    (3, 3, None),
    (10, 10, None),
    (20, 20, None),
)
CURRICULUM_WEIGHTS = (1, 3, 6)
TRAINING_CANDIDATE_OPTIONS = (
    (False, 0.01),
    (True, 0.0),
    (True, 0.01),
    (True, 0.05),
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


def _safe_mine_count(value: int, width: int, height: int) -> int:
    capacity = max(1, width * height - 1)
    if value <= 0:
        return 1
    return min(capacity, value)


def _derive_stage_mine_count(
    width: int,
    height: int,
    reference_width: int,
    reference_height: int,
    reference_mines: int,
) -> int:
    density = reference_mines / max(1, reference_width * reference_height)
    return _safe_mine_count(round(width * height * density), width, height)


def _split_stage_budget(total_games: int, weights: tuple[int, ...] = CURRICULUM_WEIGHTS):
    if total_games < len(weights):
        raise ValueError("curriculum budgets require at least one game per stage.")
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("curriculum weights must be positive.")
    total_weight = sum(weights)
    raw = [total_games * weight / total_weight for weight in weights]
    counts = [max(1, int(value)) for value in raw]
    deficit = total_games - sum(counts)
    ordered = sorted(
        range(len(weights)),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    idx = 0
    while deficit > 0:
        counts[ordered[idx % len(ordered)]] += 1
        deficit -= 1
        idx += 1
    while deficit < 0:
        ordered_backfill = sorted(
            range(len(weights)),
            key=lambda index: counts[index],
            reverse=True,
        )
        donor = ordered_backfill[0]
        if counts[donor] <= 1:
            raise ValueError("Cannot allocate curriculum split while keeping all stages non-empty.")
        counts[donor] -= 1
        deficit += 1
    return tuple(counts)


def _resolve_curriculum_stages(
    width: int,
    height: int,
    mine_count: int,
    stages: tuple[tuple[int, int, int | None], ...] | None = None,
):
    stage_list = stages or CURRICULUM_STAGES
    if len(stage_list) != 3:
        raise ValueError("curriculum must contain exactly three stages.")
    return tuple(
        (
            stage_width,
            stage_height,
            _derive_stage_mine_count(
                stage_width,
                stage_height,
                width,
                height,
                mine_count,
            )
            if stage_mines is None
            else _safe_mine_count(stage_mines, stage_width, stage_height),
        )
        for stage_width, stage_height, stage_mines in stage_list
    )


def _model_supports_board(model, width: int, height: int) -> bool:
    shape = getattr(model, "input_shape", None)
    if shape is None:
        return False
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) != 4:
        return False
    _, model_height, model_width, channels = shape
    return (
        model_height in (None, height)
        and model_width in (None, width)
        and channels in (None, 10)
    )


def _strip_candidate_model(record: dict) -> dict:
    return {
        "architecture": record["architecture"],
        "use_candidate_value": record["use_candidate_value"],
        "value_tie_margin": record["value_tie_margin"],
        "cell_metrics": record["cell_metrics"],
        "development": record["development"],
        "bot_options": record["bot_options"],
    }


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
    initial_model=None,
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
    if initial_model is not None and _model_supports_board(initial_model, width, height):
        model = initial_model
    else:
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


def _train_curriculum_stage(
    *,
    width: int,
    height: int,
    mine_count: int,
    num_games: int,
    development_games: int,
    epochs: int,
    seed: int,
    teacher_model,
    architectures: tuple[str, ...],
    stage_initial_models: dict[str, object],
):
    import random

    from minesweeper_ml.training.benchmark import evaluate_bot_factory
    from minesweeper_ml.bot.bots import MLMinesweeperBot
    from minesweeper_ml.training.data import (
        generate_training_data,
        split_dataset_by_game,
    )
    from minesweeper_ml.game import MinesweeperGame

    warmup_metrics = None
    active_teacher_model = teacher_model
    if active_teacher_model is None or not _model_supports_board(
        active_teacher_model,
        width,
        height,
    ):
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
        active_teacher_model, warmup_metrics = fit_safety_model(
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
            active_teacher_model,
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

    candidates = []
    best_by_arch = {}
    for architecture in architectures:
        initial_model = stage_initial_models.get(architecture)
        if initial_model is not None and not _model_supports_board(
            initial_model,
            width,
            height,
        ):
            initial_model = None

        candidate_model, cell_metrics = fit_multitask_model(
            train_dataset,
            test_dataset,
            width=width,
            height=height,
            architecture=architecture,
            epochs=epochs,
            initial_model=initial_model,
        )

        architecture_candidates = []
        for use_value, value_margin in TRAINING_CANDIDATE_OPTIONS:
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
                model=candidate_model,
                options=bot_options,
            ):
                return MLMinesweeperBot(
                    bot_width,
                    bot_height,
                    model,
                    **options,
                )

            development, _ = evaluate_bot_factory(
                candidate_factory,
                width=width,
                height=height,
                mine_count=mine_count,
                layouts=development_layouts,
            )
            record = {
                "architecture": architecture,
                "use_candidate_value": use_value,
                "value_tie_margin": value_margin,
                "cell_metrics": cell_metrics,
                "development": development,
                "bot_options": bot_options,
                "candidate_model": candidate_model,
            }
            candidates.append(record)
            architecture_candidates.append(record)

        best_by_arch[architecture] = max(
            architecture_candidates,
            key=_development_candidate_key,
        )

    selected = max(candidates, key=_development_candidate_key)
    return {
        "model": selected["candidate_model"],
        "sample_count": len(on_policy),
        "warmup_metrics": warmup_metrics,
        "candidates": [_strip_candidate_model(record) for record in candidates],
        "selected": _strip_candidate_model(selected),
        "best_model_by_arch": {
            architecture: record["candidate_model"]
            for architecture, record in best_by_arch.items()
        },
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
    curriculum: bool = True,
    curriculum_stages: tuple[tuple[int, int, int | None], ...] | None = None,
):
    if num_games < 2 or development_games <= 0 or epochs <= 0:
        raise ValueError("training and development sizes must be positive.")
    if curriculum:
        stage_definitions = _resolve_curriculum_stages(
            width,
            height,
            mine_count,
            stages=curriculum_stages,
        )
        stage_game_counts = _split_stage_budget(num_games)
        stage_development_counts = _split_stage_budget(development_games)

        stages = []
        best_by_arch = {}
        current_teacher = teacher_model
        total_samples = 0
        for stage_index, (
            (stage_width, stage_height, stage_mines),
            stage_game_count,
            stage_development_count,
        ) in enumerate(
            zip(
                stage_definitions,
                stage_game_counts,
                stage_development_counts,
            )
        ):
            stage_seed = seed + stage_index * 10
            stage_result = _train_curriculum_stage(
                width=stage_width,
                height=stage_height,
                mine_count=stage_mines,
                num_games=stage_game_count,
                development_games=stage_development_count,
                epochs=epochs,
                seed=stage_seed,
                teacher_model=current_teacher if current_teacher else None,
                architectures=architectures,
                stage_initial_models=best_by_arch,
            )
            current_teacher = stage_result["model"]
            best_by_arch = stage_result["best_model_by_arch"]
            total_samples += stage_result["sample_count"]

            stages.append(
                {
                    "stage_index": stage_index,
                    "width": stage_width,
                    "height": stage_height,
                    "mine_count": stage_mines,
                    "num_games": stage_game_count,
                    "development_games": stage_development_count,
                    "sample_count": stage_result["sample_count"],
                    "candidates": stage_result["candidates"],
                    "selected": stage_result["selected"],
                    "warmup_metrics": stage_result["warmup_metrics"],
                }
            )

        return (
            current_teacher,
            {
                "curriculum": True,
                "curriculum_stages": stage_definitions,
                "sample_count": total_samples,
                "stages": stages,
                "candidates": stages[-1]["candidates"],
                "selected": stages[-1]["selected"],
            },
            current_teacher,
        )

    # legacy single-stage path for backwards compatibility
    final_stage = _train_curriculum_stage(
        width=width,
        height=height,
        mine_count=mine_count,
        num_games=num_games,
        development_games=development_games,
        epochs=epochs,
        seed=seed,
        teacher_model=teacher_model,
        architectures=architectures,
        stage_initial_models={},
    )
    return (
        final_stage["model"],
        {
            "curriculum": False,
            "sample_count": final_stage["sample_count"],
            "candidates": final_stage["candidates"],
            "selected": final_stage["selected"],
            "warmup_metrics": final_stage["warmup_metrics"],
        },
        final_stage["model"],
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
