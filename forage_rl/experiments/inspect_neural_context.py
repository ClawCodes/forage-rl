"""Inspect the per-step neural context encoding for one saved episode."""

from __future__ import annotations

import argparse

import numpy as np

from forage_rl.agents import get_agent
from forage_rl.agents.registry import Agent
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.utils import load_run_dataset


def inspect_neural_context(
    *,
    agent_type: Agent,
    maze_name: str,
    run_id: int,
    episode_index: int,
    observable: bool = True,
    device: str = "auto",
) -> list[dict[str, object]]:
    """Load one saved episode and return the neural context trace."""
    if agent_type not in {Agent.DQN, Agent.DRQN}:
        raise ValueError(
            f"Context inspection only supports neural agents, got {agent_type.value}."
        )

    run_dataset = load_run_dataset(agent_type, run_id, maze_name, observable)
    trajectory = run_dataset.trajectories[episode_index]
    maze_spec = load_builtin_maze_spec(maze_name)
    maze_cls = Maze if observable else MazePOMDP
    agent = get_agent(
        agent_type,
        maze_cls(maze_spec),
        num_episodes=1,
        device=device,
        seed=DefaultParams.FRESH_EVALUATOR_SEED,
    )
    return agent.context_trace(trajectory)


def _format_feature(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values.tolist()) + "]"


def format_context_rows(rows: list[dict[str, object]], steps: int | None = None) -> str:
    """Format context-trace rows as a plain-text table."""
    limited_rows = rows if steps is None else rows[:steps]
    headers = [
        "step",
        "state",
        "time",
        "prev_action",
        "prev_reward",
        "action",
        "reward",
        "next_state",
        "feature",
    ]
    table_rows = [
        [
            str(row["step_index"]),
            str(row["state"]),
            str(row["time_spent"]),
            "start" if row["prev_action"] is None else str(row["prev_action"]),
            f"{float(row['prev_reward']):.3f}",
            str(row["action"]),
            f"{float(row['reward']):.3f}",
            str(row["next_state"]),
            _format_feature(np.asarray(row["encoded_feature"], dtype=float)),
        ]
        for row in limited_rows
    ]
    widths = [
        max(len(header), *(len(values[index]) for values in table_rows))
        for index, header in enumerate(headers)
    ]
    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator_line = "  ".join("-" * width for width in widths)
    body_lines = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in table_rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the neural-context encoding for one saved episode",
    )
    parser.add_argument(
        "--agent",
        choices=[Agent.DQN.value, Agent.DRQN.value],
        required=True,
        help="Neural agent whose saved run dataset should be inspected",
    )
    parser.add_argument(
        "--maze",
        required=True,
        help="Built-in maze spec name, e.g. simple or full",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Saved run dataset id to inspect",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Zero-based episode index inside the saved run dataset",
    )
    parser.add_argument(
        "--pomdp",
        action="store_true",
        help="Inspect the partially observable setting; default is fully observable",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional maximum number of steps to print",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for agent instantiation: auto, cpu, cuda, or mps",
    )

    args = parser.parse_args()
    rows = inspect_neural_context(
        agent_type=Agent(args.agent),
        maze_name=args.maze,
        run_id=args.run_id,
        episode_index=args.episode_index,
        observable=not args.pomdp,
        device=args.device,
    )
    print(format_context_rows(rows, steps=args.steps))


if __name__ == "__main__":
    main()
