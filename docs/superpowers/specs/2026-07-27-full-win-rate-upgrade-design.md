# Full Win-Rate Upgrade Design

**Date:** 2026-07-27

## Objective

Raise completed-map win rate beyond the accepted posterior-lookahead result of
4087/5000 (81.74%) without inspecting the hidden mine map during move
selection. The work combines exact correlated lookahead, on-policy training
data, dihedral symmetry, a candidate-value head, and a CNN architecture
tournament.

## Solver Inference

`constraints.py` will expose a reusable `ConstraintInference` object containing
the enumerated legal assignments for every solved constraint box, their
CNN-weighted masses, the global mine-count normalizer, and the unconstrained
cell count. It will continue to expose the existing marginal-probability API.

For candidate `c`, exact clue outcomes are computed by filtering legal
assignments to `c = safe`, coupling every box and the unconstrained region to the
remaining global mine count, and accumulating the joint mine count among hidden
neighbors of `c`. This replaces the independent Poisson-binomial approximation.
Unconstrained neighbors are integrated combinatorially. The same conditioned
assignment weights provide posterior marginals for each possible revealed clue,
so lookahead does not re-enumerate the board for every outcome.

Overflow, inconsistency, or total-node exhaustion remains deterministic. The bot
falls back to the accepted marginal or heuristic strategy instead of using a
partial exact distribution.

## On-Policy Dataset

The generator accepts a bot factory and plays every post-`(0,0)` move through
that bot. Move selection never reads `mine_map`, `mine_locations`, or a
ground-truth-safe candidate list.

Immediately before each non-deduction decision, the generator records:

- encoded visible state and frontier mask;
- the bot's posterior mine probabilities and near-best candidate set;
- chosen move, probability gap, game phase, and eventual match result;
- safety labels created from the mine map only after move selection;
- optional candidate-value labels from counterfactual continuations.

Samples from lost games and close-probability decisions receive configurable
sample multipliers. Collection limits are stratified by phase to avoid long
games dominating the dataset.

## Candidate Value

The multitask model has a shared spatial trunk and two sigmoid heads:

- `safety`: probability that each cell is safe;
- `value`: probability of eventually completing the board after opening that
  cell, conditioned on the cell being safe.

At an uncertain state, up to `value_rollout_candidates` near-best candidates
are evaluated. Unsafe candidates are masked out. For each safe candidate, the
generator clones the real game state, opens the candidate, and finishes with the
constraint-box policy while disabling value tie-breaking. The value label is
one only when that continuation reaches `WON`. Hidden state is used solely for
the retrospective label and rollout environment, never to select the original
trajectory move.

At inference, posterior mine risk remains the primary objective. Candidate
value breaks ties only inside the configured posterior risk margin; exact
lookahead and deterministic row-major order remain lower-priority tie-breakers.

## Symmetry

Square boards use all eight elements of the D4 group. Rectangular boards use
the four transforms that preserve shape. Training augmentation transforms
features, safety labels, masks, value labels, and value masks together.

Inference sends all transformed boards through the model in one batch,
inverse-transforms each output, and averages the aligned maps. This removes
orientation-specific priors while keeping one model call per board state.

## Architecture Tournament

Two named trunks are supported:

- `local_3x3`: the existing 32/64/32 stack;
- `paper_5x5`: wider 5x5 spatial layers with a larger channel budget.

Both use identical heads, losses, augmentation, datasets, epochs, and seeds.
Selection uses development completed-map wins, then average safe moves, then
median uncertain-move latency, then `local_3x3`.

## Compatibility

- Existing single-output `.keras` models remain valid.
- `build_cnn_model()` preserves the current single-head default.
- New multitask models identify outputs by `safety` and `value`.
- Existing data conversion and CLI calls continue to work.
- `(0,0)` remains the first move and the only mandatory mine-free generation
  cell.

## Evaluation Gate

Tune only on 500 development boards. Compare the accepted strategy and each
candidate on 5000 untouched paired boards. Report completed-map win rate,
McNemar exact p-value, average safe moves, median and p95 uncertain-move
latency, constraint fallback frequency, and move-source counts.

The new default requires at least +1 percentage point with exact `p < 0.05` and
median uncertain-move latency below 100 ms. Components may ship behind explicit
configuration when correct and tested even if the combined checkpoint does not
pass the default-selection gate.
