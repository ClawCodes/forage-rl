"""Regenerate trajectory, inference, and figure artifacts across settings."""

from __future__ import annotations

import argparse

from forage_rl.agents.registry import Agent, EvaluatorSpec
from forage_rl.config import DefaultParams
from forage_rl.experiments.generate_trajectories import run_generation_experiment
from forage_rl.experiments.model_inference import run_inference_experiment
from forage_rl.experiments.train_pretrained_agents import train_pretrained_agents
from forage_rl.visualization.plots import (
    plot_aggregate_comparison,
    plot_aggregate_trajectory_stats,
    plot_episode_return_comparison,
)


def _default_sources() -> list[Agent]:
    return [Agent.MBRL, Agent.QLearning, Agent.DQN, Agent.DRQN]


def _default_evaluators() -> list[Agent | EvaluatorSpec]:
    return [
        Agent.MBRL,
        Agent.QLearning,
        EvaluatorSpec(agent=Agent.DQN, mode="fresh"),
        EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        EvaluatorSpec(agent=Agent.DRQN, mode="fresh"),
        EvaluatorSpec(agent=Agent.DRQN, mode="pretrained"),
    ]


_SUPPORTED_SETTINGS: tuple[tuple[str, bool], ...] = (
    ("simple", True),
    ("full", True),
    ("full", False),
)


def _selected_settings(
    mazes: list[str] | None = None,
    observability: str = "all",
    *,
    verbose: bool = False,
) -> list[tuple[str, bool]]:
    requested_mazes = ["simple", "full"] if mazes is None else mazes
    selected = [
        (maze_name, observable)
        for maze_name, observable in _SUPPORTED_SETTINGS
        if maze_name in requested_mazes
        and (
            observability == "all"
            or (observability == "fo" and observable)
            or (observability == "po" and not observable)
        )
    ]
    if "simple" in requested_mazes and observability in {"all", "po"} and verbose:
        print(
            "Skipping simple/PO in regenerate_artifacts because it is redundant "
            "with simple/FO for the current artifact set."
        )
    return selected


def regenerate_artifacts(
    mazes: list[str] | None = None,
    observability: str = "all",
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    num_datasets: int | None = None,
    workers: int | None = None,
    device: str = "auto",
    seed: int = 0,
    train_pretrained: bool = False,
    skip_generation: bool = False,
    skip_inference: bool = False,
    skip_figures: bool = False,
    verbose: bool = True,
) -> None:
    """Rebuild trajectory datasets, inference outputs, and figures."""
    settings = _selected_settings(mazes, observability, verbose=verbose)
    source_agents = _default_sources()
    evaluators = _default_evaluators()
    resolved_num_datasets = num_runs if num_datasets is None else num_datasets

    for maze_name, observable in settings:
        obs_tag = "FO" if observable else "PO"
        if verbose:
            print(f"\n=== Setting: maze={maze_name}, observable={obs_tag} ===")

        if train_pretrained:
            train_pretrained_agents(
                agent_types=[Agent.DQN, Agent.DRQN],
                maze_name=maze_name,
                observable=observable,
                num_episodes=DefaultParams.NUM_EPISODES * 5,
                device=device,
                seed=seed,
                verbose=verbose,
            )

        if not skip_generation:
            run_generation_experiment(
                agent_types=source_agents,
                maze_name=maze_name,
                num_runs=num_runs,
                num_episodes=num_episodes,
                observable=observable,
                verbose=verbose,
                workers=workers,
                base_seed=seed,
                device=device,
            )

        if not skip_inference:
            run_inference_experiment(
                source_agents=source_agents,
                compare_to=evaluators,
                maze_name=maze_name,
                num_datasets=resolved_num_datasets,
                observable=observable,
                verbose=verbose,
                workers=workers,
                device=device,
                base_seed=seed,
            )

        if not skip_figures:
            plot_episode_return_comparison(
                maze_name=maze_name,
                observable=observable,
                save=True,
                show=False,
            )
            for source in source_agents:
                plot_aggregate_trajectory_stats(
                    source,
                    maze_name=maze_name,
                    observable=observable,
                    save=True,
                    show=False,
                )
                plot_aggregate_comparison(
                    source,
                    evaluators,
                    maze_name=maze_name,
                    observable=observable,
                    save=True,
                    show=False,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate trajectory, inference, and figure artifacts."
    )
    parser.add_argument(
        "--mazes",
        nargs="+",
        default=["simple", "full"],
        help="Built-in maze spec names to regenerate.",
    )
    parser.add_argument(
        "--observability",
        choices=["all", "fo", "po"],
        default="all",
        help="Which observability settings to include.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DefaultParams.NUM_TRAINING_RUNS,
        help="Number of independent run datasets to generate per source agent.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.NUM_TRAINING_EPISODES,
        help="Episodes per generated run dataset.",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=None,
        help="Number of saved run datasets to evaluate per source agent.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count override for generation and inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for neural agents and evaluators.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic base seed for generation and inference.",
    )
    parser.add_argument(
        "--train-pretrained",
        action="store_true",
        help="Refresh canonical DQN/DRQN checkpoints before inference.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip trajectory generation and reuse saved run datasets.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference and reuse saved log-likelihood artifacts.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure rendering.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    args = parser.parse_args()
    regenerate_artifacts(
        mazes=args.mazes,
        observability=args.observability,
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        num_datasets=args.num_datasets,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        train_pretrained=args.train_pretrained,
        skip_generation=args.skip_generation,
        skip_inference=args.skip_inference,
        skip_figures=args.skip_figures,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
