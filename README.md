# Minesweeper Machine Learning

Refactored Minesweeper and ML training code split out of the original Colab export.

## Layout

- `mine_sweeper_machine_learning.py` - thin launcher
- `minesweeper_ml/game.py` - board state, mines, flags, reveal logic
- `minesweeper_ml/bots.py` - rule-based and ML bot decision logic
- `minesweeper_ml/constraints.py` - weighted constraint boxes and global mine inference
- `minesweeper_ml/lookahead.py` - bounded hypothetical clue scoring
- `minesweeper_ml/data.py` - game simulation and training dataset encoding
- `minesweeper_ml/models.py` - dense and fully convolutional Keras model builders
- `minesweeper_ml/cli.py` - command-line modes
- `tests/` - smoke tests for game, data, CLI, and bot rules

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

Play manually:

```powershell
python .\mine_sweeper_machine_learning.py --mode user --width 5 --height 5 --mines 5
```

Run a tiny training smoke test:

```powershell
python .\mine_sweeper_machine_learning.py --mode smoke
```

Run the full CNN path:

```powershell
python .\mine_sweeper_machine_learning.py --mode train-cnn --games 50000 --epochs 50 --seed 42 --eval-games 100
```

## Tests

```powershell
python -m unittest discover -s tests
```

## Bot Strategy

The rule-based bot now applies basic Minesweeper deductions before using a risk fallback:

- number cells whose remaining mine count is `0` mark adjacent hidden cells safe
- number cells whose remaining mine count equals adjacent hidden cells mark those cells as mines
- subset constraints can infer extra safe cells or mines
- fallback picks the hidden cell with the lowest estimated mine risk from visible numbered constraints

The ML bot uses deterministic safe moves first. It then groups hidden frontier
cells into connected constraint boxes, enumerates layouts that satisfy every
visible clue, and weights those layouts with clamped CNN mine probabilities.
Box mine-count distributions are coupled to unconstrained cells through the
known total mine count.

The bot opens from a shortlist based on the lowest posterior mine probability.
Its `0.01` risk margin shrinks as the board approaches the endgame. For up to
four near-equal cells, bounded one-step lookahead conditions the click safe,
scores clue-consistent outcomes by expected deductions and entropy reduction,
then uses zero-region probability, information gain, and row-major order as
tie-breakers.

Base enumeration is limited to 250,000 search nodes per box. Lookahead shares
a separate 100,000-node budget per move and falls back deterministically to
information gain if a hypothesis is inconsistent, overflows, or exhausts the
budget. An oversized or inconsistent base box still falls back to the previous
60% model / 40% clue scorer.

In the baseline 5,000-board comparison on 10x10 boards with 15 mines, the
previous hybrid won 3,456 games and the constraint-box strategy won 4,043:
69.12% versus 80.86%. The paired
McNemar exact p-value was `1.64e-70`. The selected CNN prior strength was `0.5`;
the run had zero solver overflows, zero fallback moves, and a 0.657 ms median
uncertain-move latency after model prediction caching. Move sources were
131,492 deductions, 5,491 constraint-box choices, and 2,216 unconstrained-cell
choices.

The posterior-lookahead configuration was tuned separately on 500 development
boards, selecting a `0.0025` risk margin with CNN prior strength `0.75`. On
5,000 untouched paired boards, current constraint boxes won 4,025 games and
lookahead won 4,087: 80.50% versus 81.74%, a 1.24 percentage-point gain.
Lookahead won 485 boards that the baseline lost, while the baseline won 423
that lookahead lost; the McNemar exact p-value was `0.04287`.

Median uncertain-move latency was 7.68 ms and p95 was 16.18 ms on the CPU
Colab run. There were no base solver overflows or fallback moves and one shared
lookahead-budget exhaustion across all 5,000 games.

## Training Data

Every generated and evaluated game opens `(0, 0)` first. Mine generation
protects only that cell; every other cell, including its neighbors, remains
eligible for a mine.

After the first click, the offline generator mixes three exploration policies:

- 50% rule/risk bot moves for realistic trajectories
- 30% safe frontier exploration for varied local clue states
- 20% safe global exploration to break a single corner-expansion pattern

Ground truth is used only to choose safe exploration moves during offline data
generation. It is never included in model inputs.

Each game contributes at most one randomly selected state from each early,
middle, and late phase. Training and test data are split by whole game, so
states from one mine layout cannot leak across the split.

The model receives a ten-channel spatial board and predicts a safety map with a
fully convolutional network. Loss and accuracy are calculated only for hidden
frontier cells, with mine and safe labels class-balanced through sample weights.
