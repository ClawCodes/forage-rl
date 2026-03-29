"""Pipeline entry point: generate trajectories → run inference → plot results."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from forage_rl.agents.registry import Agent, registered_agents
from forage_rl.config import DefaultParams
from forage_rl.experiments.generate_trajectories import generate_trajectories
from forage_rl.experiments.model_inference import run_inference_experiment
from forage_rl.utils.io import get_run_count
from forage_rl.visualization.plots import (
    plot_aggregate_comparison,
    plot_aggregate_trajectory_stats,
)


def _confirm_overwrite(agent: Agent, maze_name: str, observable: bool) -> bool:
    n = get_run_count(agent, maze_name, observable)
    if n == 0:
        return True
    ans = input(
        f"Found {n} existing trajectory file(s) for '{agent.value}'. Overwrite? [y/N] "
    )
    return ans.strip().lower() == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: generate trajectories → inference → plots"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source agent name (e.g. q_learning, mbrl)",
    )
    parser.add_argument(
        "--compare-to",
        nargs="+",
        default=["all"],
        help="Evaluator agent name(s), or 'all'",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DefaultParams.NUM_TRAINING_RUNS,
        help="Number of independent trajectory files to generate",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.NUM_TRAINING_EPISODES,
        help="Episodes per run",
    )
    parser.add_argument(
        "--maze",
        default="simple",
        help="Built-in maze spec name (e.g. simple, full)",
    )
    parser.add_argument(
        "--pomdp",
        action="store_true",
        help="Use partially observable maze (PO); default is fully observable (FO)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run output")

    args = parser.parse_args()
    verbose = not args.quiet
    maze_name = args.maze
    observable = not args.pomdp

    # Step 1 — Validate agents
    valid_agents = registered_agents()
    valid_names = [a.value for a in valid_agents]

    try:
        source = Agent(args.source)
    except ValueError:
        print(f"Unknown agent: '{args.source}'. Valid options: {valid_names}")
        sys.exit(1)

    if args.compare_to == ["all"]:
        compare_to = valid_agents
    else:
        compare_to = []
        for name in args.compare_to:
            try:
                compare_to.append(Agent(name))
            except ValueError:
                print(f"Unknown agent: '{name}'. Valid options: {valid_names}")
                sys.exit(1)

    # Step 2 — Overwrite confirmation
    all_agents = list(dict.fromkeys([source] + compare_to))
    agents_to_generate = [
        a for a in all_agents if _confirm_overwrite(a, maze_name, observable)
    ]

    # Step 3 — Generate trajectories in parallel
    if agents_to_generate:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    generate_trajectories,
                    a,
                    maze_name,
                    args.num_runs,
                    args.num_episodes,
                    observable,
                    False,
                ): a
                for a in agents_to_generate
            }
            for future in as_completed(futures):
                agent = futures[future]
                future.result()  # re-raise any exception
                if verbose:
                    print(f"[done] {agent.value} trajectories generated.")

    # Step 4 — Run inference
    evaluators = list(dict.fromkeys([source] + compare_to))
    run_inference_experiment(
        source_agents=[source],
        compare_to=evaluators,
        maze_name=maze_name,
        observable=observable,
        verbose=verbose,
    )

    # Step 5 — Plot trajectory stats
    plot_aggregate_trajectory_stats(source, maze_name, observable, save=True)
    for agent in compare_to:
        plot_aggregate_trajectory_stats(agent, maze_name, observable, save=True)

    # Step 6 — Plot comparison
    plot_aggregate_comparison(
        source, compare_to, maze_name, observable, save=True, show=False
    )
