# Hybrid Move Scoring Design

## Goal

Improve fresh-game win rate by changing how the ML bot chooses an uncertain
move. Preserve the trained safety model and the existing deterministic
Minesweeper deductions.

The measured baseline is 668 ML-hybrid wins versus 669 rule/risk wins on the
same 1,000 seeded boards. The model has strong held-out frontier classification
metrics, but its current move strategy follows the single highest model score
without considering clue-derived risk.

## Considered Approaches

### Global density fallback

Use the known total mine count to estimate the risk of cells outside the
frontier. A prototype improved the rule bot from 669 to 737 wins on the seeded
benchmark, but this makes a classical probability fallback responsible for the
gain rather than improving ML decision-making. It is excluded from this change.

### Exact constraint solver

Enumerate satisfying mine assignments and use exact marginal probabilities.
This can improve deductions and guesses, but it would make the constraint solver
the primary decision engine. It is excluded from this change.

### Combined ML and clue scoring

Keep the model central while preventing its confidence from overruling visible
clue evidence. This is the selected approach.

## Decision Flow

For every move after the required `(0, 0)` opening:

1. Apply deterministic clue and subset deductions.
2. If a guaranteed safe move exists, open it without consulting the model.
3. Otherwise, collect hidden frontier cells that are not deduced mines.
4. Predict model safety for every candidate.
5. Estimate clue safety from the visible numbered constraints.
6. Combine the two safety values into one score.
7. Among candidates whose scores are within the tie margin of the best score,
   prefer the move with the greatest information-gain estimate.
8. Break any remaining tie by row and column for reproducibility.
9. If no trained frontier candidate exists, use the existing fallback.

The bot recalculates deductions and scores after every opened cell.

## Combined Score

For a candidate cell:

```text
combined_safety =
    model_weight * model_safety
    + (1 - model_weight) * clue_safety
```

`clue_safety` is `1 - clue_mine_risk`. The clue mine risk retains the current
conservative interpretation: when multiple visible constraints contain a cell,
use the largest local mine fraction.

The initial default `model_weight` is `0.6`. It keeps the model as the majority
signal while allowing strong clue evidence to reverse a close or overconfident
model ranking. The constructor validates that the weight is between zero and
one so seeded evaluation can compare alternative values without code changes.

The old `prediction_threshold` no longer gates uncertain frontier choices.
When the bot must guess, relative ranking remains useful even if every predicted
safety value is below `0.5`.

## Information Gain

Information gain is a tie-breaker, not a reason to take materially greater
mine risk. Candidates within `0.01` combined-safety points of the best score are
considered tied.

The estimate is the number of adjacent hidden, non-deduced cells. Revealing a
safe candidate with more hidden neighbors is expected to expose a clue that
constrains more future moves. Equal estimates use row-major order.

The tie margin is configurable and validated as non-negative.

## Code Structure

`RuleBasedMinesweeperBot` will expose a focused clue-risk helper used by both
the fallback and the ML bot. This removes duplicate risk calculations without
changing the rule bot's existing behavior.

`MLMinesweeperBot` will:

- accept `model_weight` and `tie_margin`;
- score all legal frontier candidates;
- combine model and clue safety;
- apply the information-gain tie-break;
- retain deterministic rules and the existing final fallback.

The model architecture, training data, game rules, and fixed first click are
unchanged.

## Validation

Unit tests will verify:

- clue evidence can overturn a modestly higher model score;
- equal clue risks continue to follow the stronger model prediction;
- predictions below `0.5` are still ranked when a guess is required;
- near-equal combined scores prefer greater information gain;
- constructor weights and tie margins reject invalid values;
- deterministic safe moves still bypass model prediction.

The complete local suite must pass. Gameplay evaluation will compare the old
and new strategies on identical seeded boards using the trained Colab model.
The change succeeds only if the new strategy improves paired win rate rather
than merely preserving frontier accuracy.

## Scope

This change modifies uncertain-move selection in `minesweeper_ml/bots.py` and
its focused tests. It may add evaluation wiring needed to compare strategy
weights. It does not retrain the model, change mine placement, protect cells
other than `(0, 0)`, introduce a global-density solver, or add exact constraint
enumeration.
