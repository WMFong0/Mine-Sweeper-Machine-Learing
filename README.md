# Minesweeper Machine Learning

Refactored Minesweeper and ML training code split out of the original Colab export.

## Layout

- `mine_sweeper_machine_learning.py` - thin launcher
- `minesweeper_ml/game.py` - board state, mines, flags, reveal logic
- `minesweeper_ml/bots.py` - rule-based and ML bot decision logic
- `minesweeper_ml/constraints.py` - weighted constraint boxes and global mine inference
- `minesweeper_ml/lookahead.py` - exact and control hypothetical clue scoring
- `minesweeper_ml/data.py` - on-policy trajectories and rollout labels
- `minesweeper_ml/models.py` - single- and dual-head convolutional models
- `minesweeper_ml/symmetry.py` - D4 augmentation and test-time ensembling
- `minesweeper_ml/training.py` - warm-start multitask training tournament
- `minesweeper_ml/benchmark.py` - paired completed-map evaluation
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

Run the complete win-rate experiment:

```powershell
python .\mine_sweeper_machine_learning.py --mode train-upgrade --games 50000 --epochs 50 --dev-games 500 --eval-games 5000 --model-out minesweeper_upgraded.keras
```

`train-upgrade` first creates or loads a safety teacher, generates on-policy
uncertain states, trains both CNN architectures with D4 augmentation, tunes
value tie-breaking on the development boards, and performs one paired final
comparison. Pass `--teacher-model existing.keras` to start from an existing
checkpoint.

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

The upgrade strategy caches every legal weighted box assignment. Candidate-safe
clue distributions and clue-conditioned cell marginals are derived from that
joint distribution, so mutually exclusive cells are never treated as
independent and lookahead does not repeat constraint searches. The accepted
Poisson-binomial lookahead remains the default until the upgrade passes the
paired final gate; set `exact_lookahead=True` on `MLMinesweeperBot` to enable
the new inference.

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

After the first click, every trajectory move comes from an
`MLMinesweeperBot`. The generator never filters candidates with `mine_map`, so
the policy can lose exactly as it can in deployment. States are captured
immediately before uncertain decisions, with up to four samples per game phase.
Lost games and posterior gaps at or below `0.01` receive configurable training
weight multipliers.

The mine map is consulted only after candidate selection to create safety
labels and bounded counterfactual value labels. A value label is active only
for a candidate known retrospectively to be safe, and equals one only when
opening that candidate and continuing with the constraint policy completes the
whole board. Training and test data remain split by whole mine layout.

The multitask model receives a ten-channel spatial board and predicts both cell
safety and `P(whole-board win | safe click)`. Safety loss is restricted to the
hidden frontier and class-balanced; value loss is restricted to evaluated safe
candidates. Square boards train on all eight rotations/reflections and inference
can average those eight aligned predictions in one batch.

Two trunks are trained under identical data and seeds:

- `local_3x3`: the existing 32/64/32 spatial stack
- `paper_5x5`: a wider 64/128/64 stack with 5x5 kernels

Development selection uses completed-map wins, then average safe moves, then
latency. The final strategy is accepted only for at least a one-percentage-point
paired gain, exact McNemar `p < 0.05`, and median uncertain-move latency below
100 ms.

## Google Colab

The Colab workflow needs two code cells and no embedded `BUNDLE_B64`.

```python
!git clone --branch codex/posterior-lookahead https://github.com/WMFong0/Mine-Sweeper-Machine-Learing.git
%cd Mine-Sweeper-Machine-Learing
!pip install -r requirements.txt
```

```python
!python mine_sweeper_machine_learning.py --mode train-upgrade --games 50000 --epochs 50 --dev-games 500 --eval-games 5000 --seed 42 --model-out /content/minesweeper_upgraded.keras
```
