# Diverse Minesweeper Training Data Design

## Goal

Train the ML bot to estimate whether playable hidden cells are safe from the
visible Minesweeper clues, without learning a fixed trajectory produced by the
data generator.

The model remains a cell-safety estimator. The bot remains responsible for
combining deterministic rules, model scores, and its risk fallback into a move.

## Fixed First-Click Rule

Every generated, evaluated, and ML-played game opens `(0, 0)` first.

Mine placement excludes exactly `(0, 0)`. Every other coordinate, including all
neighbors of `(0, 0)`, remains eligible for a mine. Subject to that one
exclusion, mine layouts continue to be sampled randomly from the configured
board size and mine count.

The generator must support a seed so this behavior can be tested and reproduced.

## Sources of Training Shortcuts

The current pipeline encourages shortcuts because it:

- follows one deterministic rule-bot trajectory after every first click;
- records many highly correlated states from the same game;
- trains against already-revealed cells, whose safe label is trivial;
- randomly splits individual states, allowing one board to appear in both
  training and test data;
- uses pooling and dense layers that can associate outputs with absolute board
  positions.

The fixed `(0, 0)` first click is a game rule, not a defect. The redesign removes
the other shortcut sources while preserving that rule.

## Data Generation

Each game uses a new random mine layout, opens `(0, 0)`, and then explores valid
board states with a weighted move-policy mixer:

- 50% rule-based move: use the existing rule and risk bot for realistic play;
- 30% safe frontier exploration: choose a ground-truth-safe hidden cell adjacent
  to at least one revealed cell;
- 20% safe global exploration: choose any ground-truth-safe hidden cell.

Ground truth is permitted only inside the offline generator to diversify
reachable successful-game states. It is never passed to the playing bot or
encoded as an input feature.

Before each move, the generator finds which of the three policies have a legal
candidate, renormalizes the configured weights across those policies, and then
draws one. A game ends normally on win, mine hit, or when no policy has a move.

The generator assigns every state to one reveal-progress phase:

- early: less than one third of safe cells revealed;
- middle: at least one third and less than two thirds revealed;
- late: at least two thirds revealed.

One state is selected at random from each phase reached by a game. This caps
correlation at three samples per game and prevents long late-game trajectories
from dominating the dataset.

Each retained sample contains:

- `game_id`;
- `phase`;
- visible `board_state`;
- `ground_truth_map`;
- a hidden-frontier training mask.

Move choice and final game result may be retained as diagnostic metadata, but
they are not prediction labels.

## Model Inputs and Targets

The board is encoded as per-cell feature channels:

- one hidden-cell channel;
- one channel for each revealed clue value from 0 through 8.

Only hidden frontier cells contribute to training loss. A frontier cell is
hidden and adjacent to at least one revealed clue. Revealed cells and hidden
cells with no visible evidence are masked out.

The target remains a safety probability:

- `1` for a safe frontier cell;
- `0` for a mined frontier cell.

The masked binary cross-entropy loss balances safe and mined frontier labels so
that predicting every hidden cell as safe is not rewarded by class imbalance.
Class weights are calculated from masked cells in the training games only. Each
class receives half of the total weighted contribution. Generation fails with a
clear error if the resulting training set contains no masked example of either
class.

## Model Architecture

Replace the pooled, flattened CNN with a fully convolutional safety-map model:

- same-padded convolutional layers preserve board dimensions;
- shared convolution kernels apply the same clue logic at every coordinate;
- a final one-channel sigmoid layer emits one safety score per board cell;
- no pooling, flattening, or dense output layer is used.

The ML bot flattens the returned safety map only when mapping scores back to
coordinates. It continues to use deterministic safe moves first, then model
scores for hidden frontier candidates, then the existing rule-based risk
fallback.

## Dataset Split and Evaluation

Training and test sets are split by `game_id`, never by individual state. No
mine layout or trajectory can therefore occur in both sets.

Model metrics are calculated only on masked frontier cells. End-to-end evaluation
also runs the hybrid bot on fresh seeded games that all begin at `(0, 0)` and
reports:

- games won;
- games lost by mine hit;
- games stopped without a move;
- average number of safe moves before termination.

These gameplay measurements are more meaningful than full-board accuracy.

## Validation and Errors

Generation rejects non-positive game counts and invalid board or mine counts.
It skips states with no frontier training cells. Seeded generation must be
deterministic, including mine layouts, policy choices, and phase sampling.

Progress output is optional and disabled by default so tests and library callers
remain quiet.

## Tests

Focused tests will verify:

- `(0, 0)` is never mined across generated layouts;
- a `2x1` board with one mine places that mine at `(1, 0)`, proving no cell other
  than `(0, 0)` is protected;
- every generated game opens `(0, 0)` first;
- a fixed seed reproduces the same samples;
- generated samples cover more than one post-click trajectory and game phase;
- frontier masks include only hidden cells adjacent to revealed clues;
- revealed and non-frontier cells do not contribute to labels or loss;
- train and test game identifiers are disjoint;
- one game contributes at most one sample per phase;
- the ML bot maps the convolutional safety map back to the correct coordinates.

Existing game, rule-bot, CLI, and encoding tests remain in place.

## Scope

This change updates the mine-layout invariant tests, offline data generation,
training-data conversion, CNN model, training split, bot prediction-shape
handling, CLI reporting, and README. It does not introduce reinforcement
learning, alter Minesweeper reveal rules, or protect an area larger than the
single first-click cell.
