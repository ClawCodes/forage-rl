"""Run model inference experiment comparing registered agents."""

import argparse
from typing import Optional

import numpy as np

from forage_rl import Trajectory
from forage_rl.environments import SimpleMaze
from forage_rl.agents import get_agent, registered_agents
from forage_rl.utils import load_trajectories, save_logprobs, get_run_count
from forage_rl.config import ensure_directories


def evaluate_trajectory(
    trajectory: Trajectory,
    agents: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate a trajectory under each specified agent model.

    Args:
        trajectory: Instance of Trajectory
        agents: Agent names to evaluate with; defaults to all registered agents

    Returns:
        Dictionary mapping agent name to array of per-transition log-likelihoods
    """
    if agents is None:
        agents = registered_agents()

    results = {}
    for agent_name in agents:
        agent = get_agent(agent_name, SimpleMaze())
        results[agent_name] = np.array(agent.simulate(trajectory))
    return results


def run_inference_experiment(
    source_agents: list[str] | None = None,
    compare_to: list[str] | None = None,
    num_datasets: Optional[int] = None,
    verbose: bool = True,
):
    """Run the full model inference experiment.

    For each trajectory file from each source agent, evaluate log-likelihood
    under each evaluator agent.

    Args:
        source_agents: Agents whose saved trajectories to evaluate; defaults to all registered
        compare_to: Agents to evaluate trajectories with; defaults to all registered
        num_datasets: Number of trajectory files to process per source agent
        verbose: Whether to print progress
    """
    if source_agents is None:
        source_agents = registered_agents()
    if compare_to is None:
        compare_to = registered_agents()

    ensure_directories()

    for source in source_agents:
        n = min(num_datasets or get_run_count(source), get_run_count(source))
        if n == 0:
            print(
                f"No trajectory files for {source}. Run generate_trajectories.py first."
            )
            continue

        print(f"\nEvaluating {source}-generated trajectories...")
        for i in range(n):
            trajectory = load_trajectories(source, i)

            if verbose:
                print(f"\n  Dataset {i + 1}/{n} (source: {source})")

            results = evaluate_trajectory(trajectory, compare_to)

            for evaluator, log_liks in results.items():
                label = f"source_{source}_eval_{evaluator}"
                save_logprobs(np.cumsum(log_liks), label, i)
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

    args = parser.parse_args()

    source_agents = (
        registered_agents() if args.source_agents == ["all"] else args.source_agents
    )
    compare_to = registered_agents() if args.compare_to == ["all"] else args.compare_to

    run_inference_experiment(
        source_agents=source_agents,
        compare_to=compare_to,
        num_datasets=args.num_datasets,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
