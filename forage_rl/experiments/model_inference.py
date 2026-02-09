"""Run model inference experiment comparing MBRL vs Q-learning."""

import argparse
from typing import Optional

import numpy as np

from forage_rl import TimedTransition, Trajectory
from forage_rl.environments import Maze
from forage_rl.agents import MBRL, QLearningTime
from forage_rl.utils import derive_seed, get_run_count, load_trajectories, save_logprobs
from forage_rl.config import DefaultParams, ensure_directories


def _resolve_dataset_count(requested_num_datasets: Optional[int]) -> int:
    available_num_datasets = min(
        get_run_count("mbrl"),
        get_run_count("q_learning"),
        DefaultParams.NUM_TRAINING_RUNS,
    )

    if requested_num_datasets is None:
        return available_num_datasets

    if requested_num_datasets < 0:
        raise ValueError(
            f"--num-datasets must be >= 0, got {requested_num_datasets}"
        )

    if requested_num_datasets > available_num_datasets:
        print(
            "Warning: requested "
            f"{requested_num_datasets} datasets but only "
            f"{available_num_datasets} are available; "
            f"clamping to {available_num_datasets}."
        )
    return min(requested_num_datasets, available_num_datasets)


def evaluate_trajectory(
    timed_trajectory: Trajectory[TimedTransition],
    gamma: float = DefaultParams.GAMMA,
    alpha: float = DefaultParams.ALPHA,
    maze_config: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Evaluate a trajectory under both MBRL and Q-learning models.

    Args:
        timed_trajectory: Timed trajectory to score under both models.
        gamma: Discount factor for MBRL
        alpha: Learning rate for Q-learning

    Returns:
        Dictionary with log-likelihoods under each model
    """
    mbrl_env_seed = derive_seed(seed, 0)
    mbrl_agent_seed = derive_seed(seed, 1)
    maze = Maze(spec_path=maze_config, seed=mbrl_env_seed)

    # Evaluate under MBRL
    mbrl = MBRL(
        maze,
        num_episodes=DefaultParams.NUM_EPISODES,
        gamma=gamma,
        seed=mbrl_agent_seed,
    )
    mb_log_likelihoods = mbrl.simulate_model_based_rl(timed_trajectory)

    # Evaluate under Q-learning
    ql_env_seed = derive_seed(seed, 2)
    ql_agent_seed = derive_seed(seed, 3)
    maze = Maze(spec_path=maze_config, seed=ql_env_seed)  # Fresh maze
    qlearning = QLearningTime(
        maze,
        num_episodes=DefaultParams.NUM_EPISODES,
        alpha=alpha,
        seed=ql_agent_seed,
    )
    ql_log_likelihoods = qlearning.simulate_q_learning(timed_trajectory)

    return {
        "mbrl": np.array(mb_log_likelihoods),
        "qlearning": np.array(ql_log_likelihoods),
    }


def run_inference_experiment(
    num_datasets: Optional[int] = None,
    maze_config: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """Run the full model inference experiment.

    For each trajectory file from both algorithms, evaluate log-likelihood
    under both MBRL and Q-learning models.

    Args:
        num_datasets: Number of trajectory files to process
        verbose: Whether to print progress
    """
    ensure_directories()

    dataset_count = _resolve_dataset_count(num_datasets)

    if dataset_count == 0:
        print("No trajectory files found. Run generate_trajectories.py first.")
        return

    # Process MBRL-generated trajectories
    print("\nEvaluating MBRL-generated trajectories...")
    for run_idx in range(dataset_count):
        timed_trajectory = load_trajectories("mbrl", run_idx)

        if verbose:
            print(f"\nDataset {run_idx + 1}/{dataset_count} (MBRL source)")

        run_seed = derive_seed(seed, 10, run_idx)
        results = evaluate_trajectory(
            timed_trajectory, maze_config=maze_config, seed=run_seed
        )

        mb_cumsum = np.cumsum(results["mbrl"])
        ql_cumsum = np.cumsum(results["qlearning"])

        if verbose:
            print(f"  MBRL total log-likelihood: {np.sum(results['mbrl']):.2f}")
            print(
                f"  Q-learning total log-likelihood: {np.sum(results['qlearning']):.2f}"
            )

        # Save cumulative log-likelihoods
        save_logprobs(mb_cumsum, "mbrl_true", run_idx)
        save_logprobs(ql_cumsum, "ql_false", run_idx)

    # Process Q-learning-generated trajectories
    print("\nEvaluating Q-learning-generated trajectories...")
    for run_idx in range(dataset_count):
        timed_trajectory = load_trajectories("q_learning", run_idx)

        if verbose:
            print(f"\nDataset {run_idx + 1}/{dataset_count} (Q-learning source)")

        run_seed = derive_seed(seed, 11, run_idx)
        results = evaluate_trajectory(
            timed_trajectory, maze_config=maze_config, seed=run_seed
        )

        mb_cumsum = np.cumsum(results["mbrl"])
        ql_cumsum = np.cumsum(results["qlearning"])

        if verbose:
            print(f"  MBRL total log-likelihood: {np.sum(results['mbrl']):.2f}")
            print(
                f"  Q-learning total log-likelihood: {np.sum(results['qlearning']):.2f}"
            )

        # Save cumulative log-likelihoods
        save_logprobs(mb_cumsum, "mbrl_false", run_idx)
        save_logprobs(ql_cumsum, "ql_true", run_idx)

    print("\nInference experiment complete!")


def main():
    """Parse CLI args and run cross-model log-likelihood inference."""
    parser = argparse.ArgumentParser(description="Run model inference experiment")
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=None,
        help="Number of datasets to process (default: all available)",
    )
    parser.add_argument(
        "--maze-config",
        type=str,
        default=None,
        help="Path to a maze TOML specification (default: built-in simple maze)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for environment stochasticity",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    run_inference_experiment(
        num_datasets=args.num_datasets,
        maze_config=args.maze_config,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
