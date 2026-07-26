# Posterior Lookahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve completed-map win rate by adding dynamic posterior candidate margins and bounded one-step constraint lookahead to `MLMinesweeperBot`.

**Architecture:** Keep exact mine-layout inference in `constraints.py`, including a new optional total-node cap for conditional calls. Put Poisson-binomial outcome generation and hypothetical clue scoring in a focused `lookahead.py` module, then let `bots.py` shortlist near-equal posterior candidates and rank their evaluations. Existing rule deductions, CNN priors, box inference, overflow fallback, first click, and mine generation remain unchanged.

**Tech Stack:** Python 3, standard-library `dataclasses` and `math`, NumPy-backed model predictions, `unittest`, existing constraint-box solver.

## Global Constraints

- Immediate posterior mine probability remains the primary decision criterion.
- Use `effective_margin = tie_margin * clamp(hidden_cells / board_cells, 0.25, 1.0)`.
- Evaluate at most four candidates, ordered by mine probability and then row-major position.
- Use only the visible board, configured mine total, CNN output, deduced mines, and hypothetical constraints.
- Never access `mine_locations`, `ground_truth_map`, or future game state.
- Share a deterministic 100,000-node lookahead budget across all conditional inference calls for a move.
- Fall back deterministically to adjacent-hidden information gain on inconsistent hypotheses, overflow, exhausted budget, or no valid outcomes.
- Keep the existing 250,000 per-box limit and oversized-box behavior for the base move inference.
- Keep median uncertain-move latency below 100 ms.
- Preserve first move `(0, 0)` and the guarantee that `(0, 0)` contains no mine.

---

### Task 1: Bound Conditional Constraint Inference

**Files:**
- Modify: `minesweeper_ml/constraints.py`
- Test: `tests/test_constraints.py`

**Interfaces:**
- Consumes: Existing `infer_mine_probabilities(..., max_search_nodes: int)`.
- Produces: `MineProbabilityInference.budget_exhausted: bool` and optional `max_total_search_nodes: int | None = None`.

- [ ] **Step 1: Write failing total-budget tests**

Add tests that create two independent boxes and verify a small total cap stops before completing both boxes without changing ordinary per-box overflow reporting:

```python
def test_total_search_budget_stops_conditional_inference(self):
    cells = {(0, 0), (1, 0), (3, 0), (4, 0)}
    result = infer_mine_probabilities(
        [
            (frozenset({(0, 0), (1, 0)}), 1),
            (frozenset({(3, 0), (4, 0)}), 1),
        ],
        hidden_cells=cells,
        model_mine_probabilities={cell: 0.5 for cell in cells},
        remaining_mines=2,
        prior_strength=1.0,
        max_search_nodes=100,
        max_total_search_nodes=4,
    )

    self.assertTrue(result.budget_exhausted)
    self.assertLessEqual(result.search_nodes, 4)
    self.assertEqual(0, result.overflowed_boxes)
```

Also assert that an existing unconstrained call reports `budget_exhausted is False`.

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```powershell
python -m unittest tests.test_constraints.ConstraintBoxTest.test_total_search_budget_stops_conditional_inference -v
```

Expected: failure because `max_total_search_nodes` is not accepted.

- [ ] **Step 3: Add total-budget state to inference**

Extend the result and function signature:

```python
@dataclass(frozen=True)
class MineProbabilityInference:
    mine_probabilities: dict[Coordinate, float]
    consistent: bool
    globally_coupled: bool
    overflowed_boxes: int
    search_nodes: int
    budget_exhausted: bool = False


def infer_mine_probabilities(
    constraints: list[Constraint],
    *,
    hidden_cells: set[Coordinate],
    model_mine_probabilities: dict[Coordinate, float],
    remaining_mines: int | None,
    prior_strength: float,
    max_search_nodes: int,
    max_total_search_nodes: int | None = None,
) -> MineProbabilityInference:
```

Validate a supplied total cap is positive. Before each box enumeration, calculate the remaining total budget and pass `min(max_search_nodes, remaining_total)` to the enumerator. Mark `budget_exhausted=True` when that smaller cap prevents completion; do not count it as an ordinary oversized-box overflow. Return no globally coupled result after exhaustion because not all boxes were evaluated.

Change `enumerate_constraint_box` to check the node limit before visiting a node so `search_nodes` never exceeds the configured cap:

```python
if search_nodes >= max_search_nodes:
    overflowed = True
    return
search_nodes += 1
```

- [ ] **Step 4: Run all constraint tests**

Run:

```powershell
python -m unittest tests.test_constraints -v
```

Expected: all constraint tests pass and existing overflow semantics remain deterministic.

- [ ] **Step 5: Commit bounded inference**

```powershell
git add minesweeper_ml/constraints.py tests/test_constraints.py
git commit -m "Bound conditional constraint inference"
```

---

### Task 2: Score Hypothetical Safe Clicks

**Files:**
- Create: `minesweeper_ml/lookahead.py`
- Create: `tests/test_lookahead.py`

**Interfaces:**
- Consumes: `Constraint`, `MineProbabilityInference`, and `infer_mine_probabilities` from `minesweeper_ml.constraints`.
- Produces:

```python
@dataclass(frozen=True)
class LookaheadEvaluation:
    expected_forced_cells: float
    expected_entropy_reduction: float
    zero_region_probability: float
    search_nodes: int
    budget_exhausted: bool
    valid: bool


def poisson_binomial_distribution(
    probabilities: list[float],
) -> dict[int, float]:
    ...


def evaluate_safe_click(
    candidate: Coordinate,
    *,
    hidden_neighbors: frozenset[Coordinate],
    has_known_mine_neighbor: bool,
    constraints: list[Constraint],
    hidden_cells: set[Coordinate],
    model_mine_probabilities: dict[Coordinate, float],
    remaining_mines: int | None,
    prior_strength: float,
    max_constraint_nodes: int,
    max_total_search_nodes: int,
    min_outcome_probability: float,
) -> LookaheadEvaluation:
    ...
```

- [ ] **Step 1: Write failing distribution and lookahead tests**

Cover exact small Poisson-binomial values:

```python
def test_poisson_binomial_distribution(self):
    self.assertEqual(
        {0: 0.375, 1: 0.5, 2: 0.125},
        poisson_binomial_distribution([0.25, 0.5]),
    )
```

Add a constraint scenario where conditioning one candidate safe and revealing its clue forces more cells than another candidate. Assert the richer candidate has larger `expected_forced_cells`.

Add an inconsistent clue-outcome scenario and assert its probability mass is omitted before normalization. Add a budget of one node and assert `valid is False`, `budget_exhausted is True`, and `search_nodes <= 1`.

- [ ] **Step 2: Run lookahead tests and confirm import failure**

Run:

```powershell
python -m unittest tests.test_lookahead -v
```

Expected: import failure because `minesweeper_ml.lookahead` does not exist.

- [ ] **Step 3: Implement Poisson-binomial and entropy helpers**

Use dynamic programming rather than enumerating neighbor layouts:

```python
def poisson_binomial_distribution(
    probabilities: list[float],
) -> dict[int, float]:
    distribution = {0: 1.0}
    for probability in probabilities:
        updated: dict[int, float] = {}
        for mine_count, weight in distribution.items():
            updated[mine_count] = (
                updated.get(mine_count, 0.0)
                + weight * (1.0 - probability)
            )
            updated[mine_count + 1] = (
                updated.get(mine_count + 1, 0.0)
                + weight * probability
            )
        distribution = updated
    return distribution
```

Calculate binary entropy with natural logarithms, treating probabilities at zero or one as zero entropy. Build complete scoring maps by starting with clamped model priors for hidden cells and overriding them with exact inference results.

- [ ] **Step 4: Implement safe-click evaluation**

For each candidate:

1. Add `(frozenset({candidate}), 0)` and infer the safe-conditioned board.
2. Use conditioned neighbor probabilities, falling back to model priors only for locally unconstrained neighbors.
3. Generate the unresolved-neighbor mine-count distribution.
4. For each outcome at or above `min_outcome_probability`, add `(hidden_neighbors, mine_count)` and infer again.
5. Ignore inconsistent outcomes.
6. Measure cells newly reaching probability zero or one relative to the safe-conditioned baseline.
7. Measure baseline entropy minus outcome entropy.
8. Normalize expected values by consistent outcome mass.
9. Set zero-region probability to the normalized zero-mine outcome only when there is no already-deduced mine neighbor.

Subtract every inference’s `search_nodes` from the supplied total. Stop immediately and return an invalid budget-exhausted result when none remains. Treat ordinary overflow as invalid without marking a shared-budget exhaustion.

- [ ] **Step 5: Run lookahead and constraint tests**

Run:

```powershell
python -m unittest tests.test_lookahead tests.test_constraints -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the lookahead scorer**

```powershell
git add minesweeper_ml/lookahead.py tests/test_lookahead.py
git commit -m "Add posterior lookahead scoring"
```

---

### Task 3: Integrate Dynamic Candidate Selection

**Files:**
- Modify: `minesweeper_ml/bots.py`
- Test: `tests/test_bots.py`

**Interfaces:**
- Consumes: `LookaheadEvaluation` and `evaluate_safe_click`.
- Produces these optional `MLMinesweeperBot` constructor parameters:

```python
lookahead_max_candidates: int = 4
lookahead_max_nodes: int = 100_000
lookahead_min_outcome_probability: float = 1e-6
```

- [ ] **Step 1: Write failing constructor and margin tests**

Assert defaults and validation:

```python
self.assertEqual(4, bot.lookahead_max_candidates)
self.assertEqual(100_000, bot.lookahead_max_nodes)
self.assertEqual(1e-6, bot.lookahead_min_outcome_probability)
```

Reject negative candidate counts, non-positive node budgets, and outcome thresholds outside `[0.0, 1.0]`.

Test `_effective_tie_margin` on a 10-cell board:

```python
self.assertAlmostEqual(0.01, bot._effective_tie_margin(hidden_count=10))
self.assertAlmostEqual(0.0025, bot._effective_tie_margin(hidden_count=1))
```

- [ ] **Step 2: Run focused constructor and margin tests**

Run:

```powershell
python -m unittest tests.test_bots.MLMinesweeperBotTest.test_uses_evaluated_scoring_defaults tests.test_bots.MLMinesweeperBotTest.test_effective_tie_margin_shrinks_in_the_endgame -v
```

Expected: failures for missing lookahead attributes and method.

- [ ] **Step 3: Add constructor state and metrics**

Validate and store all three parameters. Extend `strategy_stats`:

```python
"lookahead_decisions": 0,
"lookahead_search_nodes": 0,
"lookahead_budget_exhaustions": 0,
```

Implement:

```python
def _effective_tie_margin(self, hidden_count: int) -> float:
    board_cells = self.width * self.height
    hidden_fraction = hidden_count / board_cells
    return self.tie_margin * max(0.25, min(1.0, hidden_fraction))
```

- [ ] **Step 4: Write failing selection tests**

Patch `minesweeper_ml.bots.evaluate_safe_click` with deterministic `LookaheadEvaluation` values to isolate selection behavior:

- A candidate outside `lowest_probability + effective_margin` is never evaluated or selected.
- No more than `lookahead_max_candidates` are evaluated, in probability then row-major order.
- A near-equal candidate with more expected forced cells beats the slightly safer candidate.
- With equal forced cells, entropy then zero-region probability then adjacent-hidden count then row-major resolve ties.
- `lookahead_max_candidates=0` makes no scorer calls and uses dynamic-margin information gain.
- Exhaustion increments the metric and yields the same deterministic information-gain selection on repeated calls.

- [ ] **Step 5: Integrate posterior selection**

Replace the direct `_lowest_probability_move` call for successful, non-overflowed posterior inference with a method that receives constraints, hidden cells, complete clamped model priors, and remaining mine count.

Shortlist using:

```python
lowest_probability = min(mine_probabilities.values())
effective_margin = self._effective_tie_margin(len(hidden_cells))
shortlist = sorted(
    (
        (mine_probability, move)
        for move, mine_probability in mine_probabilities.items()
        if mine_probability - lowest_probability <= effective_margin
    ),
    key=lambda item: (item[0], item[1][1], item[1][0]),
)[: self.lookahead_max_candidates]
```

Use the shared node budget across candidates. Rank each candidate by:

```python
(
    -expected_forced_cells,
    -expected_entropy_reduction,
    -zero_region_probability,
    -self._information_gain(move, visible_map, deduced_mines),
    move[1],
    move[0],
)
```

An invalid candidate receives zero for the three lookahead fields. Increment `lookahead_decisions` once when at least one candidate receives a valid lookahead evaluation. Preserve `_lowest_probability_move` as the deterministic no-lookahead selector and apply the dynamic margin there as well.

Build CNN mine priors for every unresolved hidden cell, not just the current frontier, because a hypothetical clue can make previously unconstrained neighbors part of a box.

- [ ] **Step 6: Run all bot and lookahead tests**

Run:

```powershell
python -m unittest tests.test_bots tests.test_lookahead -v
```

Expected: all tests pass, including existing rule, fallback, and overflow choices.

- [ ] **Step 7: Commit bot integration**

```powershell
git add minesweeper_ml/bots.py tests/test_bots.py
git commit -m "Use bounded lookahead for posterior ties"
```

---

### Task 4: Report Lookahead Metrics

**Files:**
- Modify: `minesweeper_ml/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: the three lookahead counters in `MLMinesweeperBot.strategy_stats`.
- Produces evaluation summary keys `lookahead_decisions`, `lookahead_search_nodes`, and `lookahead_budget_exhaustions`.

- [ ] **Step 1: Write failing aggregation and output tests**

Extend `MineCountRequiringBot.strategy_stats` with known lookahead values. Assert `evaluate_ml_bot_games` aggregates them. Extend the printed summary fixture and assert output contains:

```text
4 lookahead decisions, 123 lookahead search nodes, 1 budget exhaustion
```

- [ ] **Step 2: Run CLI tests and confirm failure**

Run:

```powershell
python -m unittest tests.test_cli -v
```

Expected: missing lookahead summary keys or output.

- [ ] **Step 3: Aggregate and print metrics**

Initialize all three summary counters to zero, accumulate them after each game, and add one concise output line:

```python
print(
    f"{summary['lookahead_decisions']} lookahead decisions, "
    f"{summary['lookahead_search_nodes']} lookahead search nodes, "
    f"{summary['lookahead_budget_exhaustions']} budget exhaustions"
)
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
python -m unittest tests.test_cli -v
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit metrics**

```powershell
git add minesweeper_ml/cli.py tests/test_cli.py
git commit -m "Report posterior lookahead metrics"
```

---

### Task 5: Verify And Benchmark

**Files:**
- Modify: `README.md` only if its strategy description lists move-selection behavior.
- Modify: the existing Google Colab notebook cells after the implementation commit is pushed.

**Interfaces:**
- Consumes: completed lookahead implementation and existing box-only evaluation workflow.
- Produces: reproducible tuning and 5,000-board paired benchmark output.

- [ ] **Step 1: Run the complete unit suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: every test passes.

- [ ] **Step 2: Run compile checks**

Run:

```powershell
python -m compileall minesweeper_ml tests
```

Expected: exit code zero.

- [ ] **Step 3: Run a randomized posterior regression audit**

Generate at least 500 small seeded legal constraint systems, compare exact inferred probabilities against brute-force assignment enumeration, and assert agreement within `1e-9`. Include calls both with and without `max_total_search_nodes`.

Expected: 500 audited systems pass without mismatches.

- [ ] **Step 4: Review the final diff**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and no unrelated files staged. Leave `Account Activity.html` and `Account Activity_files/` untouched.

- [ ] **Step 5: Commit any final documentation and push**

```powershell
git add README.md
git commit -m "Document posterior lookahead strategy"
git push origin Main
```

Skip the documentation commit when `README.md` already describes the behavior accurately. Push all implementation commits before updating Colab.

- [ ] **Step 6: Update the box-only Colab evaluation**

Pin the clone cell to the pushed implementation commit. Keep hybrid strategy code removed. Tune `tie_margin` over `[0.0, 0.0025, 0.005, 0.01]` on 500 development boards while retaining `model_prior_strength=0.75`, selecting by wins, average safe moves, then proximity to `0.005`.

- [ ] **Step 7: Run the paired final benchmark**

On 5,000 untouched seeded boards, run both:

- baseline constraint boxes with `lookahead_max_candidates=0`;
- lookahead constraint boxes with the selected `tie_margin`.

Report completed-map wins and rates, baseline-only wins, lookahead-only wins, both-win/both-loss counts, McNemar exact p-value, median uncertain-move latency, constraint and lookahead search nodes, base overflow frequency, fallback frequency, budget exhaustions, and move strategy counts.

Adopt the default lookahead configuration only when lookahead has more paired wins and median uncertain-move latency remains below 100 ms. If it loses the paired comparison, retain the implementation behind `lookahead_max_candidates=0` by default and report the result.
