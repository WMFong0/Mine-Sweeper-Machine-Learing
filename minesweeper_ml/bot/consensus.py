from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from minesweeper_ml.game import Coordinate


def average_algorithm_ranks(
    candidates: Sequence[Coordinate],
    signals: Sequence[Mapping[Coordinate, float]],
) -> dict[Coordinate, float]:
    ordered_candidates = tuple(dict.fromkeys(candidates))
    if not ordered_candidates:
        return {}
    if any(
        any(candidate not in signal for candidate in ordered_candidates)
        for signal in signals
    ):
        raise ValueError("every signal must score every candidate.")

    totals = dict.fromkeys(ordered_candidates, 0.0)
    participating = 0
    denominator = max(1, len(ordered_candidates) - 1)
    for signal in signals:
        ranked = sorted(
            ordered_candidates,
            key=lambda candidate: (
                -float(signal[candidate]),
                candidate[1],
                candidate[0],
            ),
        )
        if math.isclose(
            float(signal[ranked[0]]),
            float(signal[ranked[-1]]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            continue
        participating += 1
        start = 0
        while start < len(ranked):
            end = start
            while (
                end + 1 < len(ranked)
                and math.isclose(
                    float(signal[ranked[start]]),
                    float(signal[ranked[end + 1]]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                end += 1
            score = 1.0 - ((start + end) / 2.0) / denominator
            for candidate in ranked[start : end + 1]:
                totals[candidate] += score
            start = end + 1
    return (
        {
            candidate: total / participating
            for candidate, total in totals.items()
        }
        if participating
        else totals
    )
