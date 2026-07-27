from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Callable, Sequence

from minesweeper_ml.game import GameState, MinesweeperGame


def exact_mcnemar_p_value(
    baseline_only: int,
    candidate_only: int,
) -> float:
    if baseline_only < 0 or candidate_only < 0:
        raise ValueError("discordant counts must be non-negative.")
    discordant = baseline_only + candidate_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, successes)
        for successes in range(
            min(baseline_only, candidate_only) + 1
        )
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_win_statistics(
    baseline_wins: Sequence[bool],
    candidate_wins: Sequence[bool],
) -> dict[str, float | int]:
    if len(baseline_wins) != len(candidate_wins) or not baseline_wins:
        raise ValueError("paired results must be non-empty and equally sized.")
    baseline_only = sum(
        baseline and not candidate
        for baseline, candidate in zip(
            baseline_wins,
            candidate_wins,
        )
    )
    candidate_only = sum(
        candidate and not baseline
        for baseline, candidate in zip(
            baseline_wins,
            candidate_wins,
        )
    )
    game_count = len(baseline_wins)
    return {
        "games": game_count,
        "baseline_wins": sum(baseline_wins),
        "candidate_wins": sum(candidate_wins),
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "baseline_win_rate": sum(baseline_wins) / game_count,
        "candidate_win_rate": sum(candidate_wins) / game_count,
        "win_rate_delta": (
            sum(candidate_wins) - sum(baseline_wins)
        )
        / game_count,
        "mcnemar_exact_p": exact_mcnemar_p_value(
            baseline_only,
            candidate_only,
        ),
    }


def evaluate_bot_factory(
    bot_factory: Callable[[int, int, int], object],
    *,
    width: int,
    height: int,
    mine_count: int,
    layouts: Sequence[Sequence[tuple[int, int]]],
) -> tuple[dict, list[bool]]:
    strategies = {
        "deduction": 0,
        "box": 0,
        "unconstrained": 0,
        "fallback": 0,
    }
    wins = []
    safe_moves = []
    uncertain_latencies = []
    constraint_overflows = 0
    for mine_locations in layouts:
        game = MinesweeperGame(
            width,
            height,
            mine_count,
            mine_locations=list(mine_locations),
        )
        bot = bot_factory(width, height, mine_count)
        game.open_cell(0, 0)
        opened_safe_moves = 1
        while game.state == GameState.IN_PROGRESS:
            started = time.perf_counter()
            move = bot.get_next_move(game.visible_map)
            elapsed = (time.perf_counter() - started) * 1000
            if getattr(bot, "last_move_strategy", None) not in {
                None,
                "deduction",
            }:
                uncertain_latencies.append(elapsed)
            if move is None:
                game.state = GameState.STOPPED
                break
            if game.open_cell(*move):
                opened_safe_moves += 1
        stats = getattr(bot, "strategy_stats", {})
        for strategy in strategies:
            strategies[strategy] += int(stats.get(strategy, 0))
        constraint_overflows += int(
            stats.get("constraint_overflows", 0)
        )
        wins.append(game.state == GameState.WON)
        safe_moves.append(opened_safe_moves)
    uncertain_moves = sum(
        strategies[strategy]
        for strategy in ("box", "unconstrained", "fallback")
    )
    return (
        {
            "won": sum(wins),
            "lost": sum(not won for won in wins),
            "win_rate": sum(wins) / len(wins),
            "average_safe_moves": statistics.fmean(safe_moves),
            "move_strategies": strategies,
            "constraint_overflows": constraint_overflows,
            "fallback_move_rate": (
                strategies["fallback"] / uncertain_moves
                if uncertain_moves
                else 0.0
            ),
            "median_uncertain_move_ms": (
                statistics.median(uncertain_latencies)
                if uncertain_latencies
                else 0.0
            ),
            "p95_uncertain_move_ms": _percentile(
                uncertain_latencies,
                0.95,
            ),
        },
        wins,
    )


def compare_bot_factories(
    baseline_factory: Callable[[int, int, int], object],
    candidate_factory: Callable[[int, int, int], object],
    *,
    width: int = 10,
    height: int = 10,
    mine_count: int = 15,
    num_games: int = 5_000,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    layouts = [
        tuple(
            MinesweeperGame(
                width,
                height,
                mine_count,
                seed=rng.randrange(2**63),
                first_safe=True,
            ).mine_locations
        )
        for _ in range(num_games)
    ]
    baseline, baseline_wins = evaluate_bot_factory(
        baseline_factory,
        width=width,
        height=height,
        mine_count=mine_count,
        layouts=layouts,
    )
    candidate, candidate_wins = evaluate_bot_factory(
        candidate_factory,
        width=width,
        height=height,
        mine_count=mine_count,
        layouts=layouts,
    )
    paired = paired_win_statistics(
        baseline_wins,
        candidate_wins,
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired": paired,
        "accepted": (
            paired["win_rate_delta"] >= 0.01
            and paired["mcnemar_exact_p"] < 0.05
            and candidate["median_uncertain_move_ms"] < 100.0
        ),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return (
        ordered[lower]
        if lower == upper
        else ordered[lower]
        + (ordered[upper] - ordered[lower])
        * (position - lower)
    )
