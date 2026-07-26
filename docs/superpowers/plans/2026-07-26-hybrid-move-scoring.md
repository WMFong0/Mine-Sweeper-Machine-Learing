# Hybrid Move Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the ML bot's win rate by blending model safety with clue-derived safety and using information gain only for near-equal choices.

**Architecture:** Preserve the existing deterministic rule pass and final fallback. Extract the current clue-risk calculation into one reusable helper, then make `MLMinesweeperBot` score every unresolved frontier candidate with a configurable weighted blend. Select among near-equal scores by estimated information gain and deterministic coordinate order.

**Tech Stack:** Python 3.12, standard-library `unittest`, NumPy-compatible prediction models, TensorFlow/Keras model inference.

## Global Constraints

- Every generated, evaluated, and ML-played game still opens `(0, 0)` first.
- Mine placement still excludes exactly `(0, 0)`.
- Deterministic safe deductions always run before model inference.
- The default combined score uses `model_weight=0.6`.
- Candidates within `tie_margin=0.01` of the best combined score use information gain.
- Global mine-density estimation and exact constraint enumeration are out of scope.
- The model architecture and training data are unchanged.

---

### Task 1: Extract Reusable Clue Risks

**Files:**
- Modify: `tests/test_bots.py`
- Modify: `minesweeper_ml/bots.py`

**Interfaces:**
- Consumes: `RuleBasedMinesweeperBot._number_constraints(visible_map, deduced_mines, safe_moves)`
- Produces: `RuleBasedMinesweeperBot._clue_mine_risks(visible_map, deduced_mines, safe_moves) -> dict[Coordinate, float]`
- Preserves: `RuleBasedMinesweeperBot.get_next_move(visible_map) -> Coordinate | None`

- [ ] **Step 1: Add a failing clue-risk test**

Add to `RuleBasedMinesweeperBotTest`:

```python
def test_clue_risks_use_the_most_conservative_visible_constraint(self):
    visible_map = [
        ["-", "-", "-", "-", "-"],
        [0, 1, 0, 2, 0],
    ]
    bot = RuleBasedMinesweeperBot(width=5, height=2)

    risks = bot._clue_mine_risks(visible_map, set(), set())

    self.assertAlmostEqual(1 / 3, risks[(0, 0)])
    self.assertAlmostEqual(2 / 3, risks[(2, 0)])
    self.assertAlmostEqual(2 / 3, risks[(4, 0)])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_bots.RuleBasedMinesweeperBotTest.test_clue_risks_use_the_most_conservative_visible_constraint
```

Expected: `AttributeError` because `_clue_mine_risks` does not exist.

- [ ] **Step 3: Implement the clue-risk helper**

Add to `RuleBasedMinesweeperBot`:

```python
def _clue_mine_risks(
    self,
    visible_map: list[list[Cell]],
    deduced_mines: set[Coordinate],
    safe_moves: set[Coordinate],
) -> dict[Coordinate, float]:
    risks: dict[Coordinate, float] = {}
    for cells, mine_count in self._number_constraints(
        visible_map,
        deduced_mines,
        safe_moves,
    ):
        if not cells:
            continue
        local_risk = max(0.0, min(1.0, mine_count / len(cells)))
        for move in cells:
            risks[move] = max(risks.get(move, 0.0), local_risk)
    return risks
```

Replace the duplicated `risks_by_move` construction in `_lowest_risk_move` with
this helper. Keep the existing fallback key exactly equivalent:

```python
clue_risks = self._clue_mine_risks(
    visible_map,
    deduced_mines,
    safe_moves,
)
return min(
    available_moves,
    key=lambda move: (
        clue_risks.get(move, 1.0),
        move[1],
        move[0],
    ),
)
```

- [ ] **Step 4: Run rule-bot tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_bots.RuleBasedMinesweeperBotTest
```

Expected: all rule-bot tests pass, including the unchanged lowest-risk choice.

- [ ] **Step 5: Commit the focused refactor**

```powershell
git add minesweeper_ml/bots.py tests/test_bots.py
git commit -m "refactor: expose clue-based move risks"
```

If Git still has no author identity, leave the verified changes uncommitted and
report that exact repository configuration blocker without inventing an identity.

---

### Task 2: Blend Model and Clue Safety

**Files:**
- Modify: `tests/test_bots.py`
- Modify: `minesweeper_ml/bots.py`

**Interfaces:**
- Consumes: `_clue_mine_risks(...) -> dict[Coordinate, float]`
- Produces: `MLMinesweeperBot(width, height, ml_model, *, model_weight=0.6, tie_margin=0.01, cnn_input=True)`
- Produces: unresolved candidate tuples `(combined_safety, model_safety, x, y)`

- [ ] **Step 1: Add a reusable score-map test model**

Add near `SpatialPredictionModel` in `tests/test_bots.py`:

```python
class ScoreMapPredictionModel:
    def __init__(self, scores):
        self.scores = scores
        self.last_input = None

    def predict(self, board, verbose=0):
        import numpy as np

        self.last_input = board
        return np.asarray(self.scores, dtype=np.float32).reshape(
            1,
            len(self.scores),
            len(self.scores[0]),
            1,
        )
```

- [ ] **Step 2: Add failing combined-score tests**

Add to `MLMinesweeperBotTest`:

```python
def test_combines_model_confidence_with_clue_safety(self):
    model = ScoreMapPredictionModel(
        [
            [0.80, 0.10, 0.10, 0.90, 0.10],
            [0.10, 0.10, 0.10, 0.10, 0.10],
        ]
    )
    bot = MLMinesweeperBot(width=5, height=2, ml_model=model)
    visible_map = [
        ["-", "-", "-", "-", "-"],
        [0, 1, 0, 2, 0],
    ]

    self.assertEqual((0, 0), bot.get_next_move(visible_map))

def test_equal_clue_risk_follows_the_stronger_model_score(self):
    model = ScoreMapPredictionModel([[0.60, 0.80], [0.10, 0.10]])
    bot = MLMinesweeperBot(width=2, height=2, ml_model=model)
    visible_map = [["-", "-"], [1, 1]]

    self.assertEqual((1, 0), bot.get_next_move(visible_map))

def test_ranks_model_predictions_below_one_half_when_a_guess_is_required(self):
    model = ScoreMapPredictionModel([[0.20, 0.40], [0.10, 0.10]])
    bot = MLMinesweeperBot(width=2, height=2, ml_model=model)
    visible_map = [["-", "-"], [1, 1]]

    self.assertEqual((1, 0), bot.get_next_move(visible_map))

def test_deterministic_safe_move_bypasses_model_prediction(self):
    model = SpatialPredictionModel(best_coordinate=(0, 0), width=4, height=3)
    bot = MLMinesweeperBot(width=4, height=3, ml_model=model)
    visible_map = [
        [0, "-", "-", "-"],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]

    self.assertEqual((3, 0), bot.get_next_move(visible_map))
    self.assertIsNone(model.last_input)
```

- [ ] **Step 3: Run the three tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bots.MLMinesweeperBotTest.test_combines_model_confidence_with_clue_safety `
  tests.test_bots.MLMinesweeperBotTest.test_equal_clue_risk_follows_the_stronger_model_score `
  tests.test_bots.MLMinesweeperBotTest.test_ranks_model_predictions_below_one_half_when_a_guess_is_required `
  tests.test_bots.MLMinesweeperBotTest.test_deterministic_safe_move_bypasses_model_prediction
```

Expected: the combined-score and below-threshold tests fail because the current
bot ranks only model values above `0.5`. The deterministic-bypass test already
passes and locks the behavior that the scoring change must preserve.

- [ ] **Step 4: Add constructor validation tests**

```python
def test_rejects_invalid_scoring_parameters(self):
    model = ScoreMapPredictionModel([[0.5]])

    with self.assertRaises(ValueError):
        MLMinesweeperBot(1, 1, model, model_weight=-0.01)
    with self.assertRaises(ValueError):
        MLMinesweeperBot(1, 1, model, model_weight=1.01)
    with self.assertRaises(ValueError):
        MLMinesweeperBot(1, 1, model, tie_margin=-0.01)
```

- [ ] **Step 5: Run validation test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_bots.MLMinesweeperBotTest.test_rejects_invalid_scoring_parameters
```

Expected: `TypeError` because the constructor does not accept the new keyword
arguments.

- [ ] **Step 6: Implement blended frontier scoring**

Change the constructor to:

```python
def __init__(
    self,
    width: int,
    height: int,
    ml_model: PredictionModel,
    *,
    model_weight: float = 0.6,
    tie_margin: float = 0.01,
    cnn_input: bool = True,
):
    super().__init__(width, height)
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be between zero and one.")
    if tie_margin < 0.0:
        raise ValueError("tie_margin must be non-negative.")
    self.ml_model = ml_model
    self.model_weight = model_weight
    self.tie_margin = tie_margin
    self.cnn_input = cnn_input
```

Replace threshold filtering with a candidate scorer:

```python
def _prediction_candidates(
    self,
    visible_map: list[list[Cell]],
    predictions,
    deduced_mines: set[Coordinate],
    clue_risks: dict[Coordinate, float],
) -> list[tuple[float, float, int, int]]:
    candidates = []
    for index, prediction in enumerate(predictions):
        x = index % self.width
        y = index // self.width
        move = (x, y)
        if visible_map[y][x] != "-" or move in deduced_mines:
            continue
        if not self._has_revealed_neighbor(x, y, visible_map):
            continue

        model_safety = float(prediction)
        clue_safety = 1.0 - clue_risks.get(move, 1.0)
        combined_safety = (
            self.model_weight * model_safety
            + (1.0 - self.model_weight) * clue_safety
        )
        candidates.append((combined_safety, model_safety, x, y))
    return candidates
```

In `get_next_move`, calculate `clue_risks` from the already inferred mines and
pass them into `_prediction_candidates`. For this task, select the highest
combined score with row-major tie order; Task 3 adds information gain.

- [ ] **Step 7: Run all bot tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_bots
```

Expected: all tests pass.

- [ ] **Step 8: Commit blended scoring**

```powershell
git add minesweeper_ml/bots.py tests/test_bots.py
git commit -m "feat: blend model and clue safety"
```

If Git author identity is still unset, retain the verified working-tree changes
and continue without creating a misleading commit.

---

### Task 3: Add Information-Gain Tie-Breaking

**Files:**
- Modify: `tests/test_bots.py`
- Modify: `minesweeper_ml/bots.py`

**Interfaces:**
- Consumes: candidate tuples `(combined_safety, model_safety, x, y)`
- Produces: `MLMinesweeperBot._information_gain(move, visible_map, deduced_mines) -> int`
- Produces: `MLMinesweeperBot._best_candidate(candidates, visible_map, deduced_mines) -> Coordinate`

- [ ] **Step 1: Add failing information-gain tests**

```python
def test_near_equal_scores_prefer_the_move_with_more_hidden_neighbors(self):
    model = ScoreMapPredictionModel(
        [
            [0.81, 0.80, 0.10, 0.10],
            [0.10, 0.10, 0.10, 0.10],
        ]
    )
    bot = MLMinesweeperBot(width=4, height=2, ml_model=model)
    visible_map = [
        ["-", "-", "-", "-"],
        [1, "-", "-", "-"],
    ]

    self.assertEqual((1, 0), bot.get_next_move(visible_map))

def test_materially_safer_score_wins_over_information_gain(self):
    model = ScoreMapPredictionModel(
        [
            [0.90, 0.70, 0.10, 0.10],
            [0.10, 0.10, 0.10, 0.10],
        ]
    )
    bot = MLMinesweeperBot(width=4, height=2, ml_model=model)
    visible_map = [
        ["-", "-", "-", "-"],
        [1, "-", "-", "-"],
    ]

    self.assertEqual((0, 0), bot.get_next_move(visible_map))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_bots.MLMinesweeperBotTest.test_near_equal_scores_prefer_the_move_with_more_hidden_neighbors `
  tests.test_bots.MLMinesweeperBotTest.test_materially_safer_score_wins_over_information_gain
```

Expected: the near-equal test fails because the current highest-score selection
chooses `(0, 0)`.

- [ ] **Step 3: Implement information gain and tie filtering**

Add:

```python
def _information_gain(
    self,
    move: Coordinate,
    visible_map: list[list[Cell]],
    deduced_mines: set[Coordinate],
) -> int:
    x, y = move
    return sum(
        visible_map[neighbor_y][neighbor_x] == "-"
        and (neighbor_x, neighbor_y) not in deduced_mines
        for neighbor_x, neighbor_y in self._neighbors(x, y)
    )
```

Replace `_best_candidate` with:

```python
def _best_candidate(
    self,
    candidates: list[tuple[float, float, int, int]],
    visible_map: list[list[Cell]],
    deduced_mines: set[Coordinate],
) -> Coordinate:
    best_score = max(candidate[0] for candidate in candidates)
    near_best = [
        candidate
        for candidate in candidates
        if best_score - candidate[0] <= self.tie_margin
    ]
    _, _, x, y = min(
        near_best,
        key=lambda candidate: (
            -self._information_gain(
                (candidate[2], candidate[3]),
                visible_map,
                deduced_mines,
            ),
            candidate[3],
            candidate[2],
        ),
    )
    return x, y
```

Pass `visible_map` and `deduced_mines` from `get_next_move`.

- [ ] **Step 4: Run bot tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_bots
```

Expected: all rule and ML strategy tests pass.

- [ ] **Step 5: Commit tie-breaking behavior**

```powershell
git add minesweeper_ml/bots.py tests/test_bots.py
git commit -m "feat: prefer informative near-equal moves"
```

If author identity remains unavailable, leave the tested changes uncommitted.

---

### Task 4: Document and Verify the Strategy

**Files:**
- Modify: `README.md`
- Verify: `minesweeper_ml/bots.py`
- Verify: `tests/test_bots.py`
- Verify: `tests/test_cli.py`

**Interfaces:**
- Consumes: trained Keras safety-map model
- Produces: paired old/new gameplay results on identical seeds

- [ ] **Step 1: Update the README bot-strategy section**

Replace the two-line ML description with:

```markdown
The ML bot uses deterministic safe moves first. For unresolved frontier cells,
it blends model safety with clue-derived safety, then uses expected information
gain only to break near-equal scores. The existing risk fallback is used when no
trained frontier candidate exists.
```

- [ ] **Step 2: Run the full local suite with warnings treated as errors**

Run:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests
```

Expected: every test passes with no warnings.

- [ ] **Step 3: Compile every Python entry point**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall minesweeper_ml tests mine_sweeper_machine_learning.py
```

Expected: exit code `0`.

- [ ] **Step 4: Run a paired Colab evaluation**

Use the existing trained model and the same 1,000 game seeds beginning at
`2042`. Run the legacy ML strategy and the new hybrid scoring strategy on each
mine layout. Record:

```python
{
    "legacy_wins": int,
    "hybrid_wins": int,
    "both_won": int,
    "legacy_only": int,
    "hybrid_only": int,
    "neither_won": int,
    "mcnemar_exact_p": float,
}
```

Define the legacy strategy inside the evaluation notebook so it reproduces the
pre-change decision order without relying on stale imported modules:

```python
class LegacyMLMinesweeperBot(MLMinesweeperBot):
    def get_next_move(self, visible_map):
        predictions = None
        rule_move = self._get_rule_move(visible_map)
        if rule_move is not None:
            return rule_move

        predictions = np.asarray(
            self.ml_model.predict(
                self.preprocess_board_state(visible_map),
                verbose=0,
            )
        ).reshape(-1)
        candidates = []
        for index, prediction in enumerate(predictions):
            x = index % self.width
            y = index // self.width
            if visible_map[y][x] != "-":
                continue
            if (x, y) in self.deduced_mines:
                continue
            if prediction <= 0.5:
                continue
            if not self._has_revealed_neighbor(x, y, visible_map):
                continue
            candidates.append((float(prediction), x, y))
        if candidates:
            _, x, y = max(
                candidates,
                key=lambda candidate: (
                    candidate[0],
                    -candidate[2],
                    -candidate[1],
                ),
            )
            return x, y
        return self._lowest_risk_move(visible_map)
```

Generate the paired seeds exactly once and pass each seed to both bot classes:

```python
paired_rng = random.Random(2042)
paired_seeds = [paired_rng.randrange(2**63) for _ in range(1000)]
```

Expected success criterion: `hybrid_wins > legacy_wins`. Report the exact
difference and paired p-value even if the criterion is not met.

- [ ] **Step 5: Tune only the approved scoring parameters if needed**

If the default does not improve paired wins, evaluate this fixed grid on 300
development seeds that are disjoint from the final 1,000 seeds:

```python
model_weights = [0.4, 0.5, 0.6, 0.7, 0.8]
tie_margins = [0.0, 0.01, 0.02]
```

Create development seeds with `random.Random(3042)` and assert that their seed
set is disjoint from the final seed set before evaluating:

```python
development_rng = random.Random(3042)
development_seeds = [
    development_rng.randrange(2**63)
    for _ in range(300)
]
assert set(development_seeds).isdisjoint(paired_seeds)
```

Select the pair with the most development wins, breaking equal win counts by
fewer losses and then the smaller tie margin. Run the selected pair once on the
untouched final 1,000 seeds and report both development and final results. Do
not change the model, dataset, clue-risk formula, or candidate set during this
tuning step.

- [ ] **Step 6: Commit documentation after verification**

```powershell
git add README.md
git commit -m "docs: explain hybrid move scoring"
```

If Git author identity is not configured, report that commits remain blocked;
do not set repository or global identity without the user's values.
