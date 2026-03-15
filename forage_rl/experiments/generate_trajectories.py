"""Generate training trajectories for registered agents."""

import argparse

from forage_rl.agents.registry import Agent
from forage_rl.environments import Maze, load_builtin_maze_spec
from forage_rl.agents import get_agent, registered_agents
from forage_rl.utils import save_trajectories
from forage_rl.config import DefaultParams, ensure_directories


def generate_trajectories(
    agent_type: Agent,
    maze_name: str = "simple",
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    verbose: bool = True,
):
    """Generate trajectories from a registered agent.

    Args:
        agent_type: Name of the agent in the registry
        maze_name: Built-in maze spec name (e.g. "simple", "full")
        num_runs: Number of independent training runs
        num_episodes: Episodes per run
        verbose: Whether to print progress
    """
    ensure_directories()
    maze_spec = load_builtin_maze_spec(maze_name)

    for i in range(num_runs):
        if verbose:
            print(f"\n{'=' * 50}\n{agent_type} Run {i + 1}/{num_runs}\n{'=' * 50}")

        maze = Maze(maze_spec)
        agent = get_agent(agent_type, maze, num_episodes=num_episodes)
        transitions = agent.train(verbose=False)

        filepath = save_trajectories(transitions, agent_type, i, maze_name)
        if verbose:
            print(f"Saved {len(transitions)} transitions to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate training trajectories")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Agent name(s) to run, or 'all' for every registered agent",
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
    parser.add_argument(
        "--maze",
        default="simple",
        help="Built-in maze spec name (e.g. simple, full)",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    agents = registered_agents() if args.agents == ["all"] else args.agents

    for agent_type in agents:
        if verbose:
            print(f"\nGenerating {agent_type} trajectories...")
        generate_trajectories(
            agent_type=agent_type,
            maze_name=args.maze,
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            verbose=verbose,
        )

    print("\nTrajectory generation complete!")


if __name__ == "__main__":
    main()
