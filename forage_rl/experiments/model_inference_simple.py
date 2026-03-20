"""Simple model inference test on single trajectories."""

import numpy as np

from forage_rl.agents import MBRL, QLearningTime
from forage_rl.agents.registry import Agent
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.utils import load_trajectories, get_run_count
from forage_rl.config import DefaultParams


def run_simple_inference(
    maze_name: str = "simple",
    mbrl_file_id: int = 0,
    qlearning_file_id: int = 0,
    observable: bool = True,
):
    """Run simple inference test on single trajectory files.

    Args:
        maze_name: Built-in maze spec name (e.g. 'simple', 'full')
        mbrl_file_id: ID of MBRL trajectory file to use
        qlearning_file_id: ID of Q-learning trajectory file to use
        observable: True for fully observable (FO), False for partially observable (PO)
    """
    # Check if trajectory files exist
    mbrl_count = get_run_count(Agent.MBRL, maze_name, observable)
    ql_count = get_run_count(Agent.QLearning, maze_name, observable)

    if mbrl_count == 0 or ql_count == 0:
        print("No trajectory files found. Run generate_trajectories.py first.")
        return

    maze_spec = load_builtin_maze_spec(maze_name)
    maze_cls = Maze if observable else MazePOMDP

    # Test on MBRL-generated trajectory
    print("=" * 60)
    print("Evaluating transitions from MBRL simulation")
    print("=" * 60)

    transitions = load_trajectories(Agent.MBRL, mbrl_file_id, maze_name, observable)

    maze = maze_cls(maze_spec)
    mbrl = MBRL(
        maze, num_episodes=DefaultParams.NUM_EPISODES, gamma=DefaultParams.GAMMA
    )
    mb_log_likelihood = mbrl.simulate(transitions)
    mb_total = np.sum(mb_log_likelihood)
    print(f"MBRL log-likelihood: {mb_total:.4f}")

    maze.reset()
    qlearning = QLearningTime(
        maze, num_episodes=DefaultParams.NUM_EPISODES, alpha=DefaultParams.ALPHA
    )
    ql_log_likelihood = qlearning.simulate(transitions)
    ql_total = np.sum(ql_log_likelihood)
    print(f"Q-learning log-likelihood: {ql_total:.4f}")

    if mb_total > ql_total:
        print("Result: MBRL explains the data better")
    else:
        print("Result: Q-learning explains the data better")

    # Test on Q-learning-generated trajectory
    print("\n" + "=" * 60)
    print("Evaluating transitions from Q-learning simulation")
    print("=" * 60)

    transitions = load_trajectories(
        Agent.QLearning, qlearning_file_id, maze_name, observable
    )

    maze = maze_cls(maze_spec)
    mbrl = MBRL(
        maze, num_episodes=DefaultParams.NUM_EPISODES, gamma=DefaultParams.GAMMA
    )
    mb_log_likelihood = mbrl.simulate(transitions)
    mb_total = np.sum(mb_log_likelihood)
    print(f"MBRL log-likelihood: {mb_total:.4f}")

    maze = maze_cls(maze_spec)
    qlearning = QLearningTime(
        maze, num_episodes=DefaultParams.NUM_EPISODES, alpha=DefaultParams.ALPHA
    )
    ql_log_likelihood = qlearning.simulate(transitions)
    ql_total = np.sum(ql_log_likelihood)
    print(f"Q-learning log-likelihood: {ql_total:.4f}")

    if ql_total > mb_total:
        print("Result: Q-learning explains the data better")
    else:
        print("Result: MBRL explains the data better")


if __name__ == "__main__":
    run_simple_inference()
