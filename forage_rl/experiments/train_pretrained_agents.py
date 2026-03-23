"""Train canonical pretrained checkpoints for neural evaluators."""

from __future__ import annotations

import argparse
import json

from forage_rl.agents import get_agent
from forage_rl.agents.registry import Agent
from forage_rl.config import DefaultParams, ensure_directories
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.experiments.parallel import is_neural_agent
from forage_rl.utils.io import checkpoint_metadata_path, checkpoint_path


def _parse_agents(values: list[str]) -> list[Agent]:
    if values == ["all"]:
        return [Agent.DQN, Agent.DRQN]
    return [Agent(value) for value in values]


def train_pretrained_agents(
    agent_types: list[Agent] | None = None,
    maze_name: str = "simple",
    num_episodes: int = DefaultParams.NUM_EPISODES * 5,
    observable: bool = True,
    device: str = "auto",
    seed: int = DefaultParams.FRESH_EVALUATOR_SEED,
    verbose: bool = True,
) -> None:
    """Train and save canonical final checkpoints for neural agents."""
    ensure_directories()
    agent_types = [Agent.DQN, Agent.DRQN] if agent_types is None else agent_types
    maze_spec = load_builtin_maze_spec(maze_name)
    maze_cls = Maze if observable else MazePOMDP

    for agent_type in agent_types:
        if not is_neural_agent(agent_type):
            raise ValueError(
                f"Pretraining only supports neural agents, got {agent_type.value}."
            )

        maze = maze_cls(maze_spec, seed=seed)
        agent = get_agent(
            agent_type,
            maze,
            num_episodes=num_episodes,
            device=device,
            seed=seed,
        )
        run_dataset = agent.train(verbose=verbose)
        ckpt_path = checkpoint_path(agent_type, maze_name, observable)
        agent.save_checkpoint(ckpt_path)

        metadata = {
            "agent": agent_type.value,
            "maze_name": maze_name,
            "observable": observable,
            "seed": seed,
            "device": agent.device,
            "num_episodes": num_episodes,
            "num_transitions": run_dataset.num_transitions(),
            **agent.feature_schema_metadata(),
            "hyperparameters": {
                "alpha": agent.alpha,
                "gamma": agent.gamma,
                "beta": agent.beta,
                "learning_rate": agent.learning_rate,
                "batch_size": agent.batch_size,
                "replay_capacity": agent.replay_capacity,
                "target_update_interval": agent.target_update_interval,
                "gradient_clip": agent.gradient_clip,
            },
        }
        metadata_path = checkpoint_metadata_path(agent_type, maze_name, observable)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if verbose:
            print(f"Saved {agent_type.value} checkpoint to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train canonical pretrained DQN/DRQN checkpoints"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Neural agent name(s) to pretrain, or 'all' for dqn and drqn",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.NUM_EPISODES * 5,
        help="Number of training episodes for each pretrained agent",
    )
    parser.add_argument(
        "--maze",
        default="simple",
        help="Built-in maze spec name (e.g. simple, full)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for neural agents: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DefaultParams.FRESH_EVALUATOR_SEED,
        help="Deterministic seed for pretraining",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--pomdp",
        action="store_true",
        help="Use partially observable maze (PO); default is fully observable (FO)",
    )

    args = parser.parse_args()
    train_pretrained_agents(
        agent_types=_parse_agents(args.agents),
        maze_name=args.maze,
        num_episodes=args.num_episodes,
        observable=not args.pomdp,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
