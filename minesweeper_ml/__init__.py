"""Importable pieces for the Minesweeper machine-learning project."""

import sys

from minesweeper_ml import bot as bots
sys.modules[__name__ + ".bots"] = bots

from minesweeper_ml.game import GameState, MinesweeperGame

__all__ = ["GameState", "MinesweeperGame"]
