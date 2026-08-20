"""Core game primitives and exported solver interfaces."""

from minesweeper_ml.game.game import (
    Cell,
    Coordinate,
    GameState,
    MinesweeperGame,
)
from minesweeper_ml.game.constraints import (
    Constraint,
    ConstraintBox,
    WeightedAssignment,
    BoxEnumeration,
    MineProbabilityInference,
    ConstraintInferenceContext,
    SafeClickOutcomeInference,
    build_constraint_boxes,
    build_constraint_context,
    count_legal_worlds,
    enumerate_constraint_box,
    infer_mine_probabilities,
    infer_safe_click_outcomes,
)
from minesweeper_ml.game.endgame import EndgameEvaluation, solve_endgame
from minesweeper_ml.game.symmetry import (
    augment_spatial_arrays,
    inverse_transform_spatial,
    normalize_prediction_outputs,
    predict_spatial_maps,
    spatial_symmetries,
    transform_spatial,
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
    "build_constraint_context",
    "count_legal_worlds",
    "enumerate_constraint_box",
    "infer_mine_probabilities",
    "infer_safe_click_outcomes",
    "EndgameEvaluation",
    "solve_endgame",
    "augment_spatial_arrays",
    "inverse_transform_spatial",
    "normalize_prediction_outputs",
    "predict_spatial_maps",
    "spatial_symmetries",
    "transform_spatial",
]
