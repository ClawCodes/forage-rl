"""Generate training trajectories from MBRL and Q-learning agents."""

import argparse

from forage_rl.environments import SimpleMaze
from forage_rl.agents import MBRL, QLearningTime
from forage_rl.utils import save_trajectories
from forage_rl.config import DefaultParams, ensure_directories


def generate_mbrl_trajectories(
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    gamma: float = DefaultParams.GAMMA,
    verbose: bool = True,
):
    """Generate trajectories from MBRL agent.

    Args:
        num_runs: Number of independent training runs
        num_episodes: Episodes per run
        gamma: Discount factor
        verbose: Whether to print progress
    """
    ensure_directories()

    for i in range(num_runs):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"MBRL Run {i + 1}/{num_runs}")
            print(f"{'=' * 50}")

        maze = SimpleMaze()
        agent = MBRL(maze, num_episodes=num_episodes, gamma=gamma)
        transitions = agent.train(verbose=False)

        filepath = save_trajectories(transitions, "mbrl", i)
        if verbose:
            print(f"Saved {len(transitions)} transitions to {filepath}")


def generate_qlearning_trajectories(
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    alpha: float = DefaultParams.ALPHA,
    verbose: bool = True,
):
    """Generate trajectories from Q-learning agent.

    Args:
        num_runs: Number of independent training runs
        num_episodes: Episodes per run
        alpha: Learning rate
        verbose: Whether to print progress
    """
    ensure_directories()

    for i in range(num_runs):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Q-Learning Run {i + 1}/{num_runs}")
            print(f"{'=' * 50}")

        maze = SimpleMaze()
        agent = QLearningTime(maze, num_episodes=num_episodes, alpha=alpha)
        transitions = agent.train(verbose=False)

        filepath = save_trajectories(transitions, "q_learning", i)
        if verbose:
            print(f"Saved {len(transitions)} transitions to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate training trajectories")
    parser.add_argument(
        "--algo",
        choices=["mbrl", "qlearning", "both"],
        default="both",
        help="Which algorithm(s) to run",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DefaultParams.NUM_TRAINING_RUNS,
        help="Number of independent runs",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.NUM_TRAINING_EPISODES,
        help="Episodes per run",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.algo in ["mbrl", "both"]:
        print("\nGenerating MBRL trajectories...")
        generate_mbrl_trajectories(
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            verbose=verbose,
        )

    if args.algo in ["qlearning", "both"]:
        print("\nGenerating Q-learning trajectories...")
        generate_qlearning_trajectories(
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            verbose=verbose,
        )

    print("\nTrajectory generation complete!")


if __name__ == "__main__":
    main()
