"""Core game primitives and solver primitives."""

from minesweeper_ml.game.game import Cell, Coordinate, GameState, MinesweeperGame
from minesweeper_ml.game.constraints import (
    Constraint,
    ConstraintBox,
    WeightedAssignment,
    BoxEnumeration,
    MineProbabilityInference,
    ConstraintInferenceContext,
    SafeClickOutcomeInference,
    build_constraint_boxes,
    enumerate_constraint_box,
    count_legal_worlds,
    build_constraint_context,
    infer_mine_probabilities,
    infer_safe_click_outcomes,
)
from minesweeper_ml.game.endgame import EndgameEvaluation, solve_endgame
from minesweeper_ml.game.symmetry import (
    spatial_symmetries,
    transform_spatial,
    inverse_transform_spatial,
    normalize_prediction_outputs,
    predict_spatial_maps,
    augment_spatial_arrays,
)

__all__ = [
    "Cell",
    "Coordinate",
    "GameState",
    "MinesweeperGame",
    "Constraint",
    "ConstraintBox",
    "WeightedAssignment",
    "BoxEnumeration",
    "MineProbabilityInference",
    "ConstraintInferenceContext",
    "SafeClickOutcomeInference",
    "build_constraint_boxes",
    "enumerate_constraint_box",
    "count_legal_worlds",
    "build_constraint_context",
    "infer_mine_probabilities",
    "infer_safe_click_outcomes",
    "EndgameEvaluation",
    "solve_endgame",
    "spatial_symmetries",
    "transform_spatial",
    "inverse_transform_spatial",
    "normalize_prediction_outputs",
    "predict_spatial_maps",
    "augment_spatial_arrays",
]
