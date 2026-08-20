# Mine-Sweeper-Machine-Learning

A research machine-learning project that blends constraint solving, probabilistic inference, and a small CNN stack to study robust Minesweeper policies under uncertainty.

**ASSUMPTION**

- Core game dynamics follow standard Minesweeper: only visible clue cells are observed before each move.
- The policy never reads the hidden mine map directly; all mine beliefs are inferred from clue constraints and model priors.
- The first click is fixed to `(0, 0)` for deterministic benchmark seeding.
- Full game success means all non-mine cells are resolved safely.
- When exact inference is too expensive, bounded fallbacks replace exponential exhaustive search.
- Evaluation uses matched-board pair replay to control for dataset randomness.

**RESULT / EVALUATION**

| Protocol | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| 5,000 × 10×10 × 15 mines | Hybrid: 69.12% | Constraint boxes: 80.86% | +11.74 pts |
| 5,000 untouched × 10×10 × 15 mines | Constraint boxes: 80.50% | Posterior lookahead: 81.74% | +1.24 pts |
| 500 untouched × 10×10 × 15 mines | PSEQ-D256: 85.2% | Consensus: 83.8% | +1.4 pts |
| 200 Expert boards (neutral CNN backbone) | 46.5% | 49.0% | +2.5 pts |
| 150 Expert development layouts | 55/150 completed maps | 36.67% completion | baseline reference |

Observed replay notes:

- Untouched Expert replay reached 40.0% on 30×16 × 99 mines.
- Model behavior is faster on small/mid boards and more deliberate on dense large boards where lookahead/deep inference budgets are heavily exercised.

**HOW TO RUN**

1) Environment setup.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Play mode.

```bash
python -m minesweeper_ml --mode user --width 5 --height 5 --mines 5
```

3) Smoke run (fast sanity check).

```bash
python -m minesweeper_ml --mode smoke
```

4) Train baseline single-stage CNN.

```bash
python -m minesweeper_ml --mode train-cnn --games 50000 --epochs 50 --seed 42 --eval-games 100
```

5) Train upgraded bot with curriculum stages (default: 3×3, 10×10, 20×20).

```bash
python -m minesweeper_ml --mode train-upgrade --games 50000 --epochs 50 --dev-games 500 --eval-games 5000 --seed 42 --model-out minesweeper_upgraded.keras
```

6) Optional teacher-based comparison during upgrade.

```bash
python -m minesweeper_ml --mode train-upgrade --teacher-model path/to/teacher.keras --model-out upgraded.keras
```

7) Run tests.

```bash
python -m unittest discover -s tests
```

**THE ALGORITHM BEHIND**

1. **Deterministic symbolic inference**
   - Deduce forced mines/safe cells from direct clue constraints until no theorem-like inference remains.
2. **Constraint-box decomposition**
   - Frontier cells are grouped into independent local boxes.
   - Each box is enumerated under clue constraints to collect legal mine worlds.
3. **Posterior inference**
   - `P(cell is mine) = valid_worlds_with_mine(cell) / total_valid_worlds`.
   - Safe probability is therefore `1 - P(mine)` and drives low-risk choices.
4. **PSEQ action ranking**
   - Uncertain moves are ranked by safety, expected value gain, and uncertainty shaping.
   - This balances conservativeness and information gathering.
5. **Neural assist**
   - CNN safety/value heads are used when symbolic inference is weak, especially on noisy frontier boundaries.
6. **Endgame and lookahead**
   - Bounded lookahead (`D256`) evaluates uncertain endgames where exact symbolic methods become too broad.
7. **Fallback routing**
   - If search budget is exceeded, the pipeline falls back to safer heuristic layers to keep runtime bounded.
8. **No hidden-map leakage**
   - Every decision in evaluation is based on visible state and learned/statistical context only.

**THE 3-STAGE CURRICULUM (EXPLAINED)**

`train_upgrade_pipeline` now follows this order by default:

- Stage 1: `3×3` board curriculum starter.
- Stage 2: `10×10` board refinement.
- Stage 3: `20×20` board transfer and scaling.

Allocation defaults to weights `1 : 3 : 6`, so `--games 50000` becomes `5000 / 15000 / 30000` per stage.
Mine counts are derived from the target density when stage-specific mine counts are not supplied.

Only the synthetic generated training boards are used for fitting parameters.

- **How many expert boards are used for training?**  
  None. Expert boards are reserved for evaluation/development sanity checks, not backprop updates.

**EXPERT DEVELOPMENT LAYOUTS**

- "Expert boards" in this repo are curated layouts used to probe generalization outside random generation.
- "Expert development layouts" are fixed, reused boards intended to stress deep inference and check post-merge behavior.
- They are separate from the train set by design to prevent leakage and preserve a realistic validation signal.

**HOW THE BOT FINDS MINES (INTUITIVELY)**

- Think of each clue as a local equation over neighboring hidden cells.
- If an equation forces a variable (cell) to be mine or safe, the bot acts with certainty.
- If not forced, it counts compatible mine assignments in frontier boxes and uses normalized frequency as posterior risk.
- The action policy then picks the move with the best trade-off between immediate safety and expected information benefit.

**DATASET & TRAINING PHILOSOPHY**

- Training data are generated in-situ (synthetic games) using the same bot stack.
- `train-upgrade` runs candidate architectures over candidate settings and keeps the best-performing one by paired benchmark.
- Development metrics are collected per stage and reported in the upgrade report payload.
- Paired benchmarking is used to avoid noise from random board variation.

**LIMITATIONS / NEXT CHALLENGES**

- High-density late game can still hit world-budget limits.
- Future work: adaptive curriculum pacing, uncertainty calibration, and stronger transfer tuning from large-stage experience.
