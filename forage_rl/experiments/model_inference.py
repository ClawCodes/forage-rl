"""Run model inference experiment comparing MBRL vs Q-learning."""

import argparse
from typing import Optional

import numpy as np

from forage_rl import Trajectory
from forage_rl.environments import SimpleMaze
from forage_rl.agents import MBRL, QLearningTime
from forage_rl.utils import load_trajectories, save_logprobs, get_run_count
from forage_rl.config import DefaultParams, ensure_directories


def evaluate_trajectory(
    trajectory: Trajectory,
    gamma: float = DefaultParams.GAMMA,
    alpha: float = DefaultParams.ALPHA,
) -> dict:
    """Evaluate a trajectory under both MBRL and Q-learning models.

    Args:
        trajectory: Instance of Trajectory
        gamma: Discount factor for MBRL
        alpha: Learning rate for Q-learning

    Returns:
        Dictionary with log-likelihoods under each model
    """
    maze = SimpleMaze()

    # Evaluate under MBRL
    mbrl = MBRL(maze, num_episodes=DefaultParams.NUM_EPISODES, gamma=gamma)
    mb_log_likelihoods = mbrl.simulate_model_based_rl(trajectory)

    # Evaluate under Q-learning
    maze = SimpleMaze()  # Fresh maze
    qlearning = QLearningTime(
        maze, num_episodes=DefaultParams.NUM_EPISODES, alpha=alpha
    )
    ql_log_likelihoods = qlearning.simulate_q_learning(trajectory)

    return {
        "mbrl": np.array(mb_log_likelihoods),
        "qlearning": np.array(ql_log_likelihoods),
    }


def run_inference_experiment(num_datasets: Optional[int] = None, verbose: bool = True):
    """Run the full model inference experiment.

    For each trajectory file from both algorithms, evaluate log-likelihood
    under both MBRL and Q-learning models.

    Args:
        num_datasets: Number of trajectory files to process
        verbose: Whether to print progress
    """
    ensure_directories()

    num_datasets = num_datasets or min(
        get_run_count("mbrl"),
        get_run_count("q_learning"),
        DefaultParams.NUM_TRAINING_RUNS,
    )

    if num_datasets == 0:
        print("No trajectory files found. Run generate_trajectories.py first.")
        return

    # Process MBRL-generated trajectories
    print("\nEvaluating MBRL-generated trajectories...")
    for i in range(num_datasets):
        transitions = load_trajectories("mbrl", i)

        if verbose:
            print(f"\nDataset {i + 1}/{num_datasets} (MBRL source)")

        results = evaluate_trajectory(transitions)

        mb_cumsum = np.cumsum(results["mbrl"])
        ql_cumsum = np.cumsum(results["qlearning"])

        if verbose:
            print(f"  MBRL total log-likelihood: {np.sum(results['mbrl']):.2f}")
            print(
                f"  Q-learning total log-likelihood: {np.sum(results['qlearning']):.2f}"
            )

        # Save cumulative log-likelihoods
        save_logprobs(mb_cumsum, "mbrl_true", i)
        save_logprobs(ql_cumsum, "ql_false", i)

    # Process Q-learning-generated trajectories
    print("\nEvaluating Q-learning-generated trajectories...")
    for i in range(num_datasets):
        transitions = load_trajectories("q_learning", i)

        if verbose:
            print(f"\nDataset {i + 1}/{num_datasets} (Q-learning source)")

        results = evaluate_trajectory(transitions)

        mb_cumsum = np.cumsum(results["mbrl"])
        ql_cumsum = np.cumsum(results["qlearning"])

        if verbose:
            print(f"  MBRL total log-likelihood: {np.sum(results['mbrl']):.2f}")
            print(
                f"  Q-learning total log-likelihood: {np.sum(results['qlearning']):.2f}"
            )

        # Save cumulative log-likelihoods
        save_logprobs(mb_cumsum, "mbrl_false", i)
        save_logprobs(ql_cumsum, "ql_true", i)

    print("\nInference experiment complete!")


def main():
    parser = argparse.ArgumentParser(description="Run model inference experiment")
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=None,
        help="Number of datasets to process (default: all available)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    run_inference_experiment(
        num_datasets=args.num_datasets,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
