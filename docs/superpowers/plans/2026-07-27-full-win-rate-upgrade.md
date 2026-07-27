# Full Win-Rate Upgrade Implementation Plan

> **For Codex:** Execute task-by-task with test-driven development. Preserve
> the accepted 81.74% strategy as the control until the final paired benchmark
> passes the evaluation gate.

**Goal:** Implement exact correlated lookahead, on-policy uncertain-state data,
D4 augmentation and test-time ensembling, a candidate-value model head, and a
3x3-versus-5x5 architecture tournament.

**Architecture:** Constraint enumeration becomes a reusable weighted joint
distribution. Data generation consumes the deployed bot contract and emits
multitask masks and trajectory weights. Models remain backward compatible
through normalized named outputs. Evaluation selects behavior by completed-map
wins rather than per-cell accuracy.

**Tech Stack:** Python 3.12, NumPy, TensorFlow/Keras, pytest, SciPy-compatible
exact binomial calculations implemented with the standard library.

---

## Task 1: Exact Correlated Clue Outcomes

**Files:**
- Modify: `minesweeper_ml/constraints.py`
- Modify: `minesweeper_ml/lookahead.py`
- Modify: `minesweeper_ml/bots.py`
- Test: `tests/test_constraints.py`
- Test: `tests/test_lookahead.py`

1. Add failing tests where exactly one of two correlated cells is mined and
   assert the candidate-safe clue distribution has `P(1) = 1`.
2. Add randomized small-board tests comparing exact outcome mass and
   conditional marginals with brute-force legal assignments.
3. Extend box enumeration with compact assignment masks and stable log-scaled
   weights grouped by local mine count.
4. Add an inference context that couples boxes and unconstrained cells under the
   global mine budget.
5. Compute candidate-safe clue outcomes and outcome-conditioned marginals from
   the cached assignments.
6. Switch lookahead to the exact context; preserve deterministic fallback when
   the context is incomplete.
7. Run `pytest tests/test_constraints.py tests/test_lookahead.py tests/test_bots.py`.

## Task 2: Symmetry Operations

**Files:**
- Create: `minesweeper_ml/symmetry.py`
- Modify: `minesweeper_ml/bots.py`
- Test: `tests/test_symmetry.py`
- Test: `tests/test_bots.py`

1. Add failing round-trip tests for all square and rectangular transforms.
2. Add a batched prediction test proving transformed outputs align after inverse
   transforms.
3. Implement shape-preserving transforms and inverses over the first two spatial
   axes.
4. Add one-call test-time ensembling that normalizes legacy and named model
   outputs.
5. Run `pytest tests/test_symmetry.py tests/test_bots.py`.

## Task 3: On-Policy Uncertain-State Data

**Files:**
- Modify: `minesweeper_ml/data.py`
- Modify: `minesweeper_ml/game.py`
- Test: `tests/test_data.py`
- Test: `tests/test_game.py`

1. Add a policy-spy test proving the generator opens the bot-selected mine and
   loses instead of replacing it with a hidden-map-safe move.
2. Add tests for first move `(0,0)`, no mine at `(0,0)`, uncertain-only
   collection, phase caps, loss weighting, and close-gap weighting.
3. Expose a bot factory protocol and immutable decision metadata.
4. Collect states before uncertain moves and attach result-dependent weights
   only after the game ends.
5. Keep ground-truth access inside retrospective label construction.
6. Add deterministic game cloning for rollout environments.
7. Run `pytest tests/test_data.py tests/test_game.py`.

## Task 4: Candidate-Value Rollouts

**Files:**
- Modify: `minesweeper_ml/data.py`
- Modify: `minesweeper_ml/bots.py`
- Test: `tests/test_data.py`
- Test: `tests/test_bots.py`

1. Add failing tests for safe-candidate value masks, winning and losing rollout
   labels, and zero labels being masked for mined candidates.
2. Add a failing bot test proving value can reorder only candidates inside the
   posterior tie margin.
3. Implement bounded counterfactual continuation with value tie-breaking
   disabled.
4. Record dense value targets and masks for evaluated candidates.
5. Add value-map tie-breaking ahead of exact lookahead while preserving mine
   probability as the primary key.
6. Run `pytest tests/test_data.py tests/test_bots.py`.

## Task 5: Multitask Models and Augmentation

**Files:**
- Modify: `minesweeper_ml/models.py`
- Modify: `minesweeper_ml/data.py`
- Create: `minesweeper_ml/training.py`
- Test: `tests/test_models.py`
- Test: `tests/test_data.py`
- Create: `tests/test_training.py`

1. Add TensorFlow-optional structural tests for named dual heads and both
   architecture variants.
2. Add array tests proving every label and mask receives the same symmetry as
   its input.
3. Implement `local_3x3` and `paper_5x5` trunks with optional named value head.
4. Add multitask array conversion and per-head sample weights.
5. Add deterministic D4 augmentation and common training/evaluation helpers.
6. Preserve `build_cnn_model(width, height)` behavior for old callers.
7. Run `pytest tests/test_models.py tests/test_data.py tests/test_training.py`.

## Task 6: CLI, Colab, and Benchmark Selection

**Files:**
- Modify: `minesweeper_ml/cli.py`
- Create: `minesweeper_ml/benchmark.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`
- Create: `tests/test_benchmark.py`

1. Add tests for on-policy training flags, architecture names, symmetry
   ensembling, value rollouts, and paired completed-map statistics.
2. Implement warm-start on-policy generation, multitask fitting, and
   architecture selection on 500 development boards.
3. Implement a 5000-board paired comparator with exact McNemar p-value, latency
   percentiles, fallback rates, and move-source counts.
4. Keep the Colab workflow free of embedded source bundles: clone the branch,
   install requirements, run the training command, and run the paired benchmark.
5. Document checkpoint and benchmark commands.
6. Run `pytest` and `python -m compileall minesweeper_ml tests`.

## Task 7: Train, Tune, and Gate

1. Train both architectures with identical on-policy data and D4 augmentation.
2. Tune value tie margins and exact-lookahead settings on the 500 development
   boards only.
3. Freeze the winner before opening the untouched 5000-board set.
4. Compare against the accepted 4087/5000 baseline.
5. Make the combined strategy the default only when the acceptance gate passes.
6. Commit and push the tested branch to update PR #1.
