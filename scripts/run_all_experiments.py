#!/usr/bin/env python3
"""Batch wrapper: runs forage CLI for every agent pair × maze × observability."""

import subprocess
import sys
from itertools import permutations

AGENTS = ["mbrl", "q_learning"]
# MAZES = ["simple", "simple_one_way", "full", "full_one_way"]
MAZES = ["simple_one_way", "full_one_way"]
NUM_RUNS = 100
NUM_EPISODES = 6


def run_experiment(source, compare_to, maze, pomdp):
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
        "--maze",
        maze,
        "--quiet",
    ]
    if pomdp:
        cmd.append("--pomdp")

    label = f"source={source} compare-to={compare_to} maze={maze} pomdp={pomdp}"
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}")

    # Pipe "y" for each agent's overwrite prompt (one per agent in the pair)
    result = subprocess.run(cmd, input="y\ny\n", text=True)
    if result.returncode != 0:
        print(f"[FAILED] {label} — exit code {result.returncode}", file=sys.stderr)


def main():
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
