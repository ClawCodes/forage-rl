"""Run model inference experiment comparing registered agents."""

import argparse
from typing import Optional

import numpy as np

from forage_rl import Trajectory
from forage_rl.agents.registry import Agent
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.agents import get_agent, registered_agents
from forage_rl.utils import load_trajectories, save_logprobs, get_run_count
from forage_rl.config import ensure_directories


def evaluate_trajectory(
    trajectory: Trajectory,
    maze_name: str = "simple",
    agents: list[Agent] | None = None,
    observable: bool = True,
) -> dict[Agent, np.ndarray]:
    """Evaluate a trajectory under each specified agent model.

    Args:
        trajectory: Instance of Trajectory
        maze_name: Built-in maze spec name used to initialise evaluator agents
        agents: Agent names to evaluate with; defaults to all registered agents
        observable: True for fully observable (FO), False for partially observable (PO)

    Returns:
        Dictionary mapping agent name to array of per-transition log-likelihoods
    """
    if agents is None:
        agents = registered_agents()

    maze_spec = load_builtin_maze_spec(maze_name)
    maze_cls = Maze if observable else MazePOMDP
    results = {}
    for agent_name in agents:
        agent = get_agent(agent_name, maze_cls(maze_spec))
        results[agent_name] = np.array(agent.simulate(trajectory))
    return results


def run_inference_experiment(
    source_agents: list[Agent] | None = None,
    compare_to: list[Agent] | None = None,
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    verbose: bool = True,
):
    """Run the full model inference experiment.

    For each trajectory file from each source agent, evaluate log-likelihood
    under each evaluator agent.

    Args:
        source_agents: Agents whose saved trajectories to evaluate; defaults to all registered
        compare_to: Agents to evaluate trajectories with; defaults to all registered
        maze_name: Built-in maze spec name used for both loading trajectories and evaluating
        num_datasets: Number of trajectory files to process per source agent
        observable: True for fully observable (FO), False for partially observable (PO)
        verbose: Whether to print progress
    """
    if source_agents is None:
        source_agents = registered_agents()
    if compare_to is None:
        compare_to = registered_agents()

    ensure_directories()

    for source in source_agents:
        n = min(
            num_datasets or get_run_count(source, maze_name, observable),
            get_run_count(source, maze_name, observable),
        )
        if n == 0:
            print(
                f"No trajectory files for {source.value}. Run generate_trajectories.py first."
            )
            continue

        print(f"\nEvaluating {source.value}-generated trajectories...")
        for i in range(n):
            trajectory = load_trajectories(source, i, maze_name, observable)

            if verbose:
                print(f"\n  Dataset {i + 1}/{n} (source: {source.value})")

            results = evaluate_trajectory(trajectory, maze_name, compare_to, observable)

            for evaluator, log_liks in results.items():
                save_logprobs(
                    np.cumsum(log_liks), source, evaluator, i, maze_name, observable
                )
                if verbose:
                    print(
                        f"  [{evaluator}] total log-likelihood: {np.sum(log_liks):.2f}"
                    )

    print("\nInference experiment complete!")


def main():
    parser = argparse.ArgumentParser(description="Run model inference experiment")
    parser.add_argument(
        "--source-agents",
        nargs="+",
        default=["all"],
        help="Source agent name(s) whose trajectories to evaluate, or 'all'",
    )
    parser.add_argument(
        "--compare-to",
        nargs="+",
        default=["all"],
        help="Evaluator agent name(s) to compare against, or 'all'",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=None,
        help="Number of datasets to process per source agent (default: all available)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--maze",
        default="simple",
        help="Built-in maze spec name (e.g. simple, full)",
    )
    parser.add_argument(
        "--pomdp",
        action="store_true",
        help="Use partially observable trajectories (PO); default is fully observable (FO)",
    )

    args = parser.parse_args()

    source_agents = (
        registered_agents() if args.source_agents == ["all"] else args.source_agents
    )
    compare_to = registered_agents() if args.compare_to == ["all"] else args.compare_to

    run_inference_experiment(
        source_agents=source_agents,
        compare_to=compare_to,
        maze_name=args.maze,
        num_datasets=args.num_datasets,
        observable=not args.pomdp,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
