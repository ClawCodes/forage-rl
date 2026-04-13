#!/usr/bin/env python3
"""Batch wrapper: runs the forage CLI for every agent pair, maze, and observability."""

from __future__ import annotations

import subprocess
import sys
from itertools import permutations

AGENTS = ["mbrl", "q_learning"]
MAZES = ["simple_one_way", "full_one_way"]
NUM_RUNS = 100
NUM_EPISODES = 6


def run_experiment(source: str, compare_to: str, maze: str, pomdp: bool) -> None:
    cmd = [
        "forage",
        "--source",
        source,
        "--compare-to",
        compare_to,
        "--num-runs",
        str(NUM_RUNS),
        "--num-episodes",
        str(NUM_EPISODES),
        "--num-datasets",
        str(NUM_RUNS),
        "--maze",
        maze,
        "--quiet",
        "--yes",
    ]
    if pomdp:
        cmd.append("--pomdp")

    label = f"source={source} compare-to={compare_to} maze={maze} pomdp={pomdp}"
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAILED] {label} — exit code {result.returncode}", file=sys.stderr)


def main() -> None:
    pairs = list(permutations(AGENTS, 2))
    total = len(pairs) * len(MAZES) * 2
    print(f"Running {total} experiment combinations...")

    for source, compare_to in pairs:
        for maze in MAZES:
            for pomdp in [False, True]:
                run_experiment(source, compare_to, maze, pomdp)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
