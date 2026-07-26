# Posterior Lookahead Strategy Design

## Goal

Raise the constraint-box bot's completed-map win rate without changing the CNN,
training data, first click, mine placement, or access to game ground truth.

The strategy must remain deterministic for a fixed board and model, keep median
uncertain-move latency below 100 ms, and fall back to the current posterior
selection whenever lookahead cannot finish safely.

## Selected Approach

Use a dynamic risk margin followed by bounded one-step constraint lookahead.

Immediate posterior mine probability remains the primary decision criterion.
Only cells close enough to the safest posterior are eligible for lookahead. The
eligible risk margin shrinks as the number of hidden cells decreases:

```text
effective_margin =
    tie_margin * clamp(hidden_cells / board_cells, 0.25, 1.0)
```

This permits information-seeking choices early in a game while becoming more
strict about immediate survival in the endgame.

At most four candidates are evaluated, ordered by mine probability and then
row-major position.

## One-Step Lookahead

For each shortlisted candidate:

1. Add a constraint declaring the candidate safe.
2. Re-run constraint inference to obtain neighbor probabilities conditioned on
   that safe click.
3. Build a Poisson-binomial approximation for the possible mine count among
   the candidate's still-hidden neighbors.
4. For each outcome with meaningful probability, add the corresponding visible
   clue constraint and run inference again.
5. Measure newly forced safe/mine cells and posterior entropy reduction.
6. Normalize over consistent outcomes and calculate the expected values.

The candidate score is ordered by:

1. expected newly forced cells;
2. expected posterior entropy reduction;
3. probability of a zero-region reveal;
4. current adjacent-hidden information gain;
5. row-major order.

The lookahead never reads `mine_locations`, `ground_truth_map`, or any future
game state. It uses only the visible board, configured mine total, CNN output,
deduced mines, and hypothetical constraints.

## Budgets And Fallbacks

Lookahead receives a deterministic shared budget of 100,000 constraint search
nodes per move. Each conditional inference consumes from that budget.

If a hypothesis is inconsistent, it contributes no outcome mass. If inference
overflows, exhausts the shared budget, or produces no valid outcomes, that
candidate falls back to the existing adjacent-hidden information score. If all
candidates fall back, behavior remains deterministic and row-major stable.

Existing per-box enumeration limits and oversized-box fallback behavior remain
unchanged.

## Interfaces And Metrics

Add these optional `MLMinesweeperBot` parameters:

```python
lookahead_max_candidates: int = 4
lookahead_max_nodes: int = 100_000
lookahead_min_outcome_probability: float = 1e-6
```

Zero candidates disables posterior lookahead without changing compatibility.

Extend strategy statistics with:

- `lookahead_decisions`
- `lookahead_search_nodes`
- `lookahead_budget_exhaustions`

CLI and Colab evaluation summaries report these values.

## Testing

Unit tests cover:

- the effective risk margin shrinking toward the endgame;
- cells outside the dynamic margin never being selected;
- lookahead preferring a near-equal move with more expected deductions;
- impossible clue outcomes being ignored;
- search-budget exhaustion returning a deterministic fallback;
- no lookahead when the candidate limit is zero;
- constructor validation and metric aggregation.

The existing constraint, first-click, mine-placement, full unit, compile, and
randomized posterior checks continue to pass.

## Evaluation

Use 500 development boards to tune `tie_margin` over
`[0.0, 0.0025, 0.005, 0.01]`, retaining the selected CNN prior strength from
the box-only run. Selection is by wins, average safe moves, then proximity to
`0.005`.

Compare the current box strategy and lookahead strategy on 5,000 untouched
paired boards. Report win rate, paired outcomes, McNemar exact p-value, median
uncertain-move latency, search budget use, overflow/fallback frequency, and
lookahead decision count.

Adopt lookahead only if it increases paired wins without exceeding the 100 ms
latency requirement. A statistically significant improvement is preferred but
not assumed.
