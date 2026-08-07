# Mine-Sweeper-Machine-Learning

A research-style minesweeper-solving project that combines constraint satisfaction, probabilistic inference, and neural guidance to study move policies under uncertainty.

**ASSUMPTION**

- The board is treated as a single-agent decision process with partially observable state; only revealed clues are observed before selecting the next action.
- A move is defined by the same API contract as classic Minesweeper: selecting a safe/uncertain cell may immediately end the game if it is a mine.
- Training games are generated from synthetic boards, not solved with hidden-map leakage during policy evaluation.
- Every game in the benchmark opens `(0, 0)` first to keep evaluation deterministic across seeds.
- A full map is counted as a win only when the entire field is safely resolved.
- When board uncertainty is too high or combinatorics are explosive, the solver uses bounded fallbacks instead of exponential search.
- Candidate policies are evaluated with paired, head-to-head board matching to reduce evaluator noise.

**RESULT / EVALUATION**

| Paired evaluation | Baseline | Candidate | Outcome |
| --- | ---: | ---: | --- |
| 5,000 boards, 10x10 with 15 mines | Hybrid: 69.12% | Constraint boxes: 80.86% | +11.74 points, McNemar `p=1.64e-70` |
| 5,000 untouched boards, 10x10 with 15 mines | Constraint boxes: 80.50% | Posterior lookahead: 81.74% | +1.24 points, McNemar `p=0.04287` |
| 200 Expert boards, neutral CNN | Previous Expert profile: 46.5% | Strict PSEQ profile: 49.0% | +2.5 points |
| 200 untouched Expert boards, neutral CNN | PSEQ-D256: 33.0% | Consensus: 32.5% | Consensus lost by one game (`p=1.0`) |
| 500 untouched 10x10 boards, neutral CNN | PSEQ-D256: 85.2% | Consensus: 83.8% | +1.4 points |

**PSEQ-D256 replay across all historical layouts**

| Board | Completed-map wins | Win rate | Median uncertain move | p95 uncertain move |
| --- | ---: | ---: | ---: | ---: |
| 10x10, 15 mines | 832/1,000 | **83.2%** | 24.91 ms | 26.54 ms |
| 16x16, 40 mines | 162/200 | **81.0%** | 38.52 ms | 78.41 ms |
| Expert, 30x16 with 99 mines, untouched | 200/500 | **40.0%** | 113.76 ms | 548.25 ms |
| Expert development layouts | 55/150 | **36.67%** | 103.18 ms | 544.62 ms |

Observed behavior notes:
- Untouched Expert replay reached 40.0% with historical board batches.
- It made 202 D256 decisions, with 52 endgame budget exhaustions and 12 constraint overflows.
- The expert policy is materially slower but increases depth of search before speculative moves.

**HOW TO RUN**

1) Create environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Run from the compatibility entrypoint:

```bash
python mine_sweeper_machine_learning.py --mode user --width 5 --height 5 --mines 5
python mine_sweeper_machine_learning.py --mode smoke
python mine_sweeper_machine_learning.py --mode train-cnn --games 50000 --epochs 50 --seed 42 --eval-games 100
python mine_sweeper_machine_learning.py --mode train-upgrade --games 50000 --epochs 50 --dev-games 500 --eval-games 5000 --seed 42 --model-out minesweeper_upgraded.keras
```

3) Or run via module form (same commands):

```bash
python -m minesweeper_ml --mode user --width 5 --height 5 --mines 5
python -m minesweeper_ml --mode train-upgrade --games 50000 --epochs 50 --dev-games 500 --eval-games 5000 --seed 42 --model-out minesweeper_upgraded.keras
```

4) Run tests:

```bash
python -m unittest discover -s tests
```

5) Optional Colab quickstart:

```python
!git clone --branch codex/posterior-lookahead https://github.com/WMFong0/Mine-Sweeper-Machine-Learing.git
%cd Mine-Sweeper-Machine-Learing
!pip install -r requirements.txt
!python mine_sweeper_machine_learning.py --mode train-upgrade --games 50000 --epochs 50 --dev-games 500 --eval-games 5000 --seed 42 --model-out /content/minesweeper_upgraded.keras
```

**THE ALGORITHM BEHIND**

1. **Safe opening policy** opens `(0, 0)` first to remove first-move variance.
2. **Deterministic inference** applies direct clue rules repeatedly to mark certain mines/safe cells.
3. **Constraint box extraction** groups frontier-revealed dependencies; each box is solved as an independent CSP-like enumeration under clue constraints.
4. **Exact local probabilities** are computed from legal assignments and normalized across mines remaining and unconstrained cells.
5. **Posterior ranking (PSEQ)** ranks uncertain cells using:
   - guaranteed safe probability (`S`)
   - expected guaranteed safe cells gained (`E`)
   - uncertainty/entropy (`Q`)
6. **Expert path** switches to stricter risk filtering at high board scale and increases PSEQ budget.
7. **D256 endgame search** performs bounded recursive lookahead on legal worlds until at most 256 legal worlds remain.
8. **Neural assist** uses CNN outputs for safe probability and completion-value shaping when symbolic information is ambiguous.
9. **Symmetry augmentation** (D4 transforms) lets the value/safety network evaluate equivalent states in batch.
10. **Consensus policy (research mode)** blends independent scorers (constraint, PSEQ, CNN, local information gain); kept for ablation, not default.
11. **Bounded fallbacks** keep runtime practical with strict caps on box enumeration and lookahead node budgets.

**DATASET & TRAINING PHILOSOPHY**

- Data is generated by the deployed constraint bot to avoid hidden-map cheating.
- States are recorded before uncertain decisions; uncertain and loss-heavy trajectories are intentionally over-represented.
- Labels include safety target and optional completion/value targets (`P(finish | safe click)`).
- Splits are done by full mine layout (`split_dataset_by_game`) to minimize leakage.
- Benchmark uses paired board replay to isolate policy quality from sample-shape effects.

**LIMITATIONS / WHAT TO TRY NEXT**

- Current bottlenecks are high-density endgames where D256 budget saturates.
- Future work: adaptive policy switching, stronger value priors, and better uncertainty calibration under extreme mine-density tails.
