# Diverse Training Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate varied, reproducible Minesweeper states that always begin at `(0, 0)` and train a spatial model only on meaningful hidden frontier cells.

**Architecture:** Keep mine placement unchanged except for explicitly testing the existing `(0, 0)` exclusion. Replace the single-policy trajectory collector with a seeded policy mixer and one-sample-per-phase reservoir. Convert states into one-hot spatial features, safety maps, and frontier masks; split by game; then train a fully convolutional safety-map model and adapt the ML bot to its output.

**Tech Stack:** Python 3, standard-library `random` and `unittest`, NumPy, scikit-learn-free game-level splitting, TensorFlow/Keras for model construction.

## Global Constraints

- Every generated, evaluated, and ML-played game opens `(0, 0)` first.
- Mine placement excludes exactly `(0, 0)`; every other coordinate remains eligible.
- Ground truth may guide offline exploration but must never be a model input.
- Training and test states must be separated by `game_id`.
- Only hidden frontier cells contribute to model loss and metrics.
- Generation is quiet by default and deterministic when given a seed.

---

### Task 1: Lock the Mine-Placement Invariant

**Files:**
- Modify: `tests/test_game.py`
- Verify: `minesweeper_ml/game.py`

**Interfaces:**
- Consumes: `MinesweeperGame(..., first_safe=True)`
- Produces: regression coverage proving only `(0, 0)` is protected

- [ ] **Step 1: Add the exact invariant test**

```python
def test_first_safe_protects_only_zero_zero(self):
    game = MinesweeperGame(width=2, height=1, mine_count=1, seed=7, first_safe=True)

    self.assertNotIn((0, 0), game.mine_locations)
    self.assertEqual({(1, 0)}, set(game.mine_locations))
```

- [ ] **Step 2: Run the focused test**

Run:

```powershell
python -m unittest tests.test_game
```

Expected: PASS, confirming mine generation itself does not need redesign.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_game.py
git commit -m "test: lock first-click mine invariant"
```

### Task 2: Generate Mixed, Phase-Balanced Trajectories

**Files:**
- Modify: `tests/test_data.py`
- Modify: `minesweeper_ml/data.py`

**Interfaces:**
- Produces: `frontier_mask(board_state) -> list[list[int]]`
- Produces: `run_single_bot_game_and_collect_data(width, height, mine_count, *, seed=None, game_id=0) -> tuple[list[dict], GameState]`
- Produces: `generate_training_data(num_games, width, height, mine_count, *, seed=None, verbose=False) -> list[dict]`

- [ ] **Step 1: Add failing generator tests**

```python
def test_frontier_mask_marks_only_hidden_neighbors_of_revealed_cells(self):
    board = [[1, "-", "-"], ["-", "-", "-"]]
    self.assertEqual([[0, 1, 0], [1, 1, 0]], frontier_mask(board))

def test_seeded_generation_is_reproducible_and_phase_capped(self):
    first = generate_training_data(8, 5, 5, 5, seed=123)
    second = generate_training_data(8, 5, 5, 5, seed=123)
    self.assertEqual(first, second)
    counts = Counter((sample["game_id"], sample["phase"]) for sample in first)
    self.assertTrue(first)
    self.assertTrue(all(count == 1 for count in counts.values()))

def test_generated_games_keep_zero_zero_safe(self):
    samples = generate_training_data(8, 5, 5, 5, seed=321)
    self.assertTrue(all(sample["ground_truth_map"][0][0] != "M" for sample in samples))
```

- [ ] **Step 2: Run tests to verify missing interfaces fail**

Run:

```powershell
python -m unittest tests.test_data
```

Expected: import failures for `frontier_mask` and the new keyword arguments.

- [ ] **Step 3: Implement the seeded policy mixer**

Add constants and helpers in `minesweeper_ml/data.py`:

```python
POLICY_WEIGHTS = {"rule": 0.5, "frontier": 0.3, "global": 0.2}
PHASES = ("early", "middle", "late")

def frontier_mask(board_state):
    height = len(board_state)
    width = len(board_state[0])
    mask = [[0 for _ in range(width)] for _ in range(height)]
    for y, row in enumerate(board_state):
        for x, cell in enumerate(row):
            if cell != "-":
                continue
            if any(
                board_state[ny][nx] != "-"
                for nx, ny in _neighbors(x, y, width, height)
            ):
                mask[y][x] = 1
    return mask
```

Use `random.Random(seed)` for mine seeds, policy selection, candidate selection,
and per-phase reservoir sampling. Open `(0, 0)` before collecting. At each move,
build legal candidates for the rule bot, safe frontier exploration, and safe
global exploration; renormalize `POLICY_WEIGHTS` over available policies.

Calculate phase from revealed safe cells divided by total safe cells:

```python
def _game_phase(game):
    progress = 1.0 - (game.remaining_cells / (game.width * game.height - game.mine_count))
    if progress < 1 / 3:
        return "early"
    if progress < 2 / 3:
        return "middle"
    return "late"
```

Retain one randomly selected sample per reached phase using reservoir sampling.
Each sample contains `game_id`, `phase`, `board_state`, `ground_truth_map`,
`training_mask`, `chosen_move`, and `is_safe`.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m unittest tests.test_data tests.test_game
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add minesweeper_ml/data.py tests/test_data.py tests/test_game.py
git commit -m "feat: diversify generated game states"
```

### Task 3: Build Masked Spatial Tensors and Game-Level Splits

**Files:**
- Modify: `tests/test_data.py`
- Modify: `minesweeper_ml/data.py`
- Modify: `minesweeper_ml/cli.py`

**Interfaces:**
- Produces: `encode_board_features(board_state) -> list[list[list[int]]]`
- Produces: `dataset_to_arrays(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- Produces: `split_dataset_by_game(dataset, test_fraction=0.2, seed=42) -> tuple[list[dict], list[dict]]`
- Produces: `calculate_class_weights(labels, masks) -> dict[int, float]`
- Produces: `build_sample_weights(labels, masks, class_weights) -> np.ndarray`

- [ ] **Step 1: Add failing tensor and split tests**

```python
def test_board_features_are_one_hot_spatial_channels(self):
    features = encode_board_features([["-", 0, 2]])
    self.assertEqual(10, len(features[0][0]))
    self.assertEqual(1, features[0][0][0])
    self.assertEqual(1, features[0][1][1])
    self.assertEqual(1, features[0][2][3])

def test_dataset_arrays_include_spatial_labels_and_masks(self):
    sample = {
        "board_state": [[1, "-"], ["-", "-"]],
        "ground_truth_map": [[1, "M"], [0, 1]],
        "training_mask": [[0, 1], [1, 1]],
    }
    features, labels, masks = dataset_to_arrays([sample])
    self.assertEqual((1, 2, 2, 10), features.shape)
    self.assertEqual((1, 2, 2, 1), labels.shape)
    self.assertEqual((1, 2, 2, 1), masks.shape)

def test_split_dataset_by_game_has_no_game_overlap(self):
    dataset = [{"game_id": game_id} for game_id in range(10) for _ in range(2)]
    train, test = split_dataset_by_game(dataset, seed=7)
    self.assertTrue({s["game_id"] for s in train}.isdisjoint(
        {s["game_id"] for s in test}
    ))
```

- [ ] **Step 2: Run tests and observe missing functions**

Run:

```powershell
python -m unittest tests.test_data
```

Expected: FAIL on missing spatial encoding and split helpers.

- [ ] **Step 3: Implement one-hot features, spatial arrays, and split**

Encode hidden or flagged cells in channel 0 and revealed clues `0..8` in
channels `1..9`. Return float32 arrays shaped `(samples, height, width, channels)`
and label/mask arrays shaped `(samples, height, width, 1)`.

Split unique game IDs with a local seeded RNG, require at least two unique games,
and keep every state for one game on one side.

Calculate safe/mine class weights from masked training labels:

```python
mine_weight = masked_count / (2 * mine_count)
safe_weight = masked_count / (2 * safe_count)
```

Raise `ValueError` if either class is absent. Build `(samples, height, width)`
sample weights by multiplying class weights by the frontier mask.

- [ ] **Step 4: Adapt `train_cnn_bot` to split before array conversion**

Generate states, call `split_dataset_by_game`, convert each side independently,
calculate class weights from training arrays, and pass spatial sample weights to
`model.fit` and `model.evaluate`.

- [ ] **Step 5: Run focused and CLI tests**

Run:

```powershell
python -m unittest tests.test_data tests.test_cli
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add minesweeper_ml/data.py minesweeper_ml/cli.py tests/test_data.py
git commit -m "feat: mask frontier targets by game"
```

### Task 4: Use a Fully Convolutional Safety Map

**Files:**
- Modify: `minesweeper_ml/models.py`
- Modify: `minesweeper_ml/bots.py`
- Modify: `tests/test_bots.py`

**Interfaces:**
- Consumes: one-hot tensors shaped `(batch, height, width, 10)`
- Produces: safety maps shaped `(batch, height, width, 1)`
- Preserves: `MLMinesweeperBot.get_next_move(visible_map) -> Coordinate | None`

- [ ] **Step 1: Add a failing safety-map bot test**

```python
def test_ml_bot_maps_spatial_predictions_to_frontier_coordinates(self):
    model = SpatialPredictionModel(best_coordinate=(1, 1), width=3, height=2)
    bot = MLMinesweeperBot(3, 2, model)
    board = [[1, "-", "-"], ["-", "-", "-"]]

    self.assertEqual((1, 1), bot.get_next_move(board))
    self.assertEqual((1, 2, 3, 10), model.last_input.shape)
```

The fake model returns a nested `(1, 2, 3, 1)` score map with `(1, 1)` highest.

- [ ] **Step 2: Run the bot test and observe shape failure**

Run:

```powershell
python -m unittest tests.test_bots
```

Expected: FAIL because preprocessing still produces one scalar channel and
prediction iteration expects a flat vector.

- [ ] **Step 3: Implement spatial preprocessing and prediction flattening**

Use `encode_board_features`, create a float32 batch, and flatten only the model
output:

```python
predictions = np.asarray(
    self.ml_model.predict(self.preprocess_board_state(visible_map), verbose=0)
).reshape(-1)
```

- [ ] **Step 4: Replace the model architecture**

Build a Keras model with input `(height, width, 10)`, three same-padded
convolutions, and a one-channel sigmoid output:

```python
inputs = keras.Input(shape=(height, width, 10))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
outputs = keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    weighted_metrics=[keras.metrics.BinaryAccuracy(name="masked_accuracy")],
)
```

Remove pooling, flattening, dense output, and the obsolete CNN reshape helper.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m unittest tests.test_bots tests.test_data
```

Expected: PASS without importing TensorFlow.

- [ ] **Step 6: Commit**

```powershell
git add minesweeper_ml/models.py minesweeper_ml/bots.py tests/test_bots.py
git commit -m "feat: predict spatial safety maps"
```

### Task 5: Wire Reproducible Training, Reporting, and Documentation

**Files:**
- Modify: `minesweeper_ml/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Adds CLI option: `--seed`, default `42`
- Adds CLI option: `--eval-games`, full default `100`, smoke default `2`
- Produces: fresh-game win/loss/stopped and safe-move summary

- [ ] **Step 1: Add failing CLI default tests**

```python
def test_training_defaults_include_seed_and_evaluation_games(self):
    args = build_parser().parse_args(["--mode", "train-cnn"])
    self.assertEqual(42, args.seed)
    self.assertEqual(100, args.eval_games)

def test_smoke_evaluates_two_seeded_games(self):
    args = build_parser().parse_args(["--mode", "smoke"])
    self.assertEqual(2, args.eval_games)
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```powershell
python -m unittest tests.test_cli
```

Expected: FAIL because the parser has no seed or evaluation-game options.

- [ ] **Step 3: Add seeded CLI wiring and quiet generation**

Pass `seed` through training generation and game-level splitting. Add a compact
fresh-game evaluator that always constructs games with `first_safe=True`, opens
`(0, 0)`, counts safe moves, and reports aggregate outcomes without per-move
sleep or board output.

- [ ] **Step 4: Update README**

Document the fixed `(0, 0)` rule, mixed policy percentages, phase sampling,
frontier-only masking, game-level split, fully convolutional safety map, and the
new CLI options.

- [ ] **Step 5: Run full verification**

Run:

```powershell
python -m unittest discover -s tests
python -m compileall minesweeper_ml tests mine_sweeper_machine_learning.py
python mine_sweeper_machine_learning.py --help
python -m minesweeper_ml --help
```

Expected: all tests pass, compilation succeeds, and both help commands list
`--seed` and `--eval-games`.

- [ ] **Step 6: Commit**

```powershell
git add minesweeper_ml/cli.py tests/test_cli.py README.md
git commit -m "feat: report seeded bot evaluation"
```
