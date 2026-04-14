"""Train canonical pretrained checkpoints for neural evaluators."""

from __future__ import annotations

import argparse
import json

from forage_rl.agents import get_agent
from forage_rl.agents.registry import (
    Agent,
    NEURAL_CONTEXT_MODES,
    NeuralContextMode,
    canonical_agent,
    neural_agents,
)
from forage_rl.config import DefaultParams, ensure_directories
from forage_rl.environments import (
    Maze,
    load_builtin_maze_spec,
    resolve_effective_horizon,
)
from forage_rl.experiments.parallel import is_neural_agent
from forage_rl.utils.io import checkpoint_metadata_path, checkpoint_path


def _parse_agents(values: list[str]) -> list[Agent]:
    if values == ["all"]:
        return neural_agents()
    return [Agent(value) for value in values]


def train_pretrained_agents(
    agent_types: list[Agent] | None = None,
    maze_name: str = "simple",
    num_episodes: int = DefaultParams.NUM_EPISODES * 5,
    observable: bool = True,
    device: str = "auto",
    seed: int = DefaultParams.FRESH_EVALUATOR_SEED,
    verbose: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> None:
    """Train and save canonical final checkpoints for neural agents."""
    ensure_directories()
    agent_types = neural_agents() if agent_types is None else agent_types
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    maze_spec = load_builtin_maze_spec(maze_name)

    for agent_type in agent_types:
        if not is_neural_agent(agent_type):
            raise ValueError(
                f"Pretraining only supports neural agents, got {agent_type.value}."
            )

        maze = Maze(maze_spec, seed=seed, horizon=resolved_horizon, observable=observable)
        agent = get_agent(
            agent_type,
            maze,
            num_episodes=num_episodes,
            device=device,
            seed=seed,
            context_mode=context_mode,
        )
        run_dataset = agent.train(verbose=verbose)
        ckpt_path = checkpoint_path(
            agent_type,
            maze_name,
            observable,
            context_mode=context_mode,
            horizon=resolved_horizon,
        )
        agent.save_checkpoint(ckpt_path)

        metadata = {
            "agent": canonical_agent(agent_type).value,
            "maze_name": maze_name,
            "observable": observable,
            "horizon": resolved_horizon,
            "seed": seed,
            "device": agent.device,
            "context_mode": context_mode,
            "num_episodes": num_episodes,
            "num_transitions": run_dataset.num_transitions(),
            **agent.feature_schema_metadata(),
            "hyperparameters": {
                "context_mode": agent.context_mode,
                "gamma": agent.gamma,
                "beta": agent.beta,
                "learning_rate": agent.learning_rate,
                "batch_size": agent.batch_size,
                "replay_capacity": agent.replay_capacity,
                "target_update_interval": agent.target_update_interval,
                "gradient_clip": agent.gradient_clip,
                **(
                    {
                        "sequence_length": agent.sequence_length,
                        "burn_in": agent.burn_in,
                    }
                    if hasattr(agent, "sequence_length")
                    else {}
                ),
            },
        }
        metadata_path = checkpoint_metadata_path(
            agent_type,
            maze_name,
            observable,
            context_mode=context_mode,
            horizon=resolved_horizon,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if verbose:
            print(
                f"Saved {canonical_agent(agent_type).value} checkpoint to {ckpt_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train canonical pretrained neural-agent checkpoints"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Neural agent name(s) to pretrain, or 'all' for dqn elman gru lstm.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.NUM_EPISODES * 5,
        help="Number of training episodes for each pretrained agent",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional episode-length override; default uses the built-in maze horizon.",
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
        "--context-mode",
        choices=list(NEURAL_CONTEXT_MODES),
        default="legacy_context",
        help="Neural input context mode to use while training checkpoints.",
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
        context_mode=args.context_mode,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
