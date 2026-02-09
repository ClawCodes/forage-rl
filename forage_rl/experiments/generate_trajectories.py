"""Generate training trajectories from MBRL and Q-learning agents."""

import argparse
from typing import Optional

from forage_rl.environments import Maze
from forage_rl.agents import MBRL, QLearningTime
from forage_rl.utils import derive_seed, save_trajectories
from forage_rl.config import DefaultParams, ensure_directories


def generate_mbrl_trajectories(
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    gamma: float = DefaultParams.GAMMA,
    maze_config: Optional[str] = None,
    seed: Optional[int] = None,
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

    for run_idx in range(num_runs):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"MBRL Run {run_idx + 1}/{num_runs}")
            print(f"{'=' * 50}")

        env_seed = derive_seed(seed, 0, run_idx)
        agent_seed = derive_seed(seed, 1, run_idx)
        maze = Maze(spec_path=maze_config, seed=env_seed)
        agent = MBRL(maze, num_episodes=num_episodes, gamma=gamma, seed=agent_seed)
        timed_trajectory = agent.train(verbose=False)

        filepath = save_trajectories(timed_trajectory, "mbrl", run_idx)
        if verbose:
            print(f"Saved {len(timed_trajectory)} transitions to {filepath}")


def generate_qlearning_trajectories(
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    alpha: float = DefaultParams.ALPHA,
    maze_config: Optional[str] = None,
    seed: Optional[int] = None,
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

    for run_idx in range(num_runs):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Q-Learning Run {run_idx + 1}/{num_runs}")
            print(f"{'=' * 50}")

        env_seed = derive_seed(seed, 2, run_idx)
        agent_seed = derive_seed(seed, 3, run_idx)
        maze = Maze(spec_path=maze_config, seed=env_seed)
        agent = QLearningTime(
            maze,
            num_episodes=num_episodes,
            alpha=alpha,
            seed=agent_seed,
        )
        timed_trajectory = agent.train(verbose=False)

        filepath = save_trajectories(timed_trajectory, "q_learning", run_idx)
        if verbose:
            print(f"Saved {len(timed_trajectory)} transitions to {filepath}")


def main():
    """Parse CLI args and generate trajectories for selected algorithms."""
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
    verbose = not args.quiet

    if args.algo in ["mbrl", "both"]:
        print("\nGenerating MBRL trajectories...")
        generate_mbrl_trajectories(
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            maze_config=args.maze_config,
            seed=args.seed,
            verbose=verbose,
        )

    if args.algo in ["qlearning", "both"]:
        print("\nGenerating Q-learning trajectories...")
        generate_qlearning_trajectories(
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            maze_config=args.maze_config,
            seed=args.seed,
            verbose=verbose,
        )

    print("\nTrajectory generation complete!")


if __name__ == "__main__":
    main()
