"""Compatibility CLI for the legacy forage pipeline entrypoint."""

from __future__ import annotations

import argparse
import sys

from forage_rl.agents.registry import Agent, NEURAL_CONTEXT_MODES, registered_agents
from forage_rl.config import DefaultParams
from forage_rl.experiments.generate_trajectories import run_generation_experiment
from forage_rl.experiments.model_inference import (
    _parse_evaluators,
    run_inference_experiment,
)
from forage_rl.utils import get_run_count
from forage_rl.visualization.plots import (
    plot_aggregate_comparison,
    plot_aggregate_trajectory_stats,
)


def _confirm_overwrite(agent: Agent, maze_name: str, observable: bool) -> bool:
    run_count = get_run_count(agent, maze_name, observable)
    if run_count == 0:
        return True

    answer = input(
        f"Found {run_count} existing run dataset(s) for '{agent.value}'. "
        "Overwrite? [y/N] "
    )
    return answer.strip().lower() == "y"


def _parse_agents(values: list[str]) -> list[Agent]:
    if values == ["all"]:
        return registered_agents()
    return [Agent(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: generate trajectories, run inference, and plot."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source agent name (e.g. q_learning, mbrl, dqn, lstm)",
    )
    parser.add_argument(
        "--compare-to",
        nargs="+",
        default=["all"],
        help=(
            "Evaluator agent name(s), evaluator specs like dqn:pretrained, or 'all'."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DefaultParams.NUM_RUN_DATASETS,
        help="Number of independent run datasets to generate",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DefaultParams.TRAINING_EPISODES,
        help="Episodes per run dataset",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=None,
        help="Number of saved run datasets to use during inference and plotting",
    )
    parser.add_argument(
        "--maze",
        default="simple",
        help="Built-in maze spec name (e.g. simple, full)",
    )
    parser.add_argument(
        "--pomdp",
        action="store_true",
        help="Use partially observable maze (PO); default is fully observable (FO)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for neural agents/evaluators: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--context-mode",
        choices=list(NEURAL_CONTEXT_MODES),
        default="legacy_context",
        help="Neural input context mode for source/evaluator neural policies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DefaultParams.DEFAULT_SEED,
        help="Deterministic base seed for reproducible generation and evaluation",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional episode-length override; default uses the built-in maze horizon.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Overwrite existing run datasets without prompting",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run output")

    args = parser.parse_args()
    verbose = not args.quiet
    observable = not args.pomdp

    valid_agents = registered_agents()
    valid_names = [agent.value for agent in valid_agents]

    try:
        source = Agent(args.source)
    except ValueError:
        print(f"Unknown agent: '{args.source}'. Valid options: {valid_names}")
        sys.exit(1)

    try:
        compare_to = _parse_evaluators(args.compare_to, args.context_mode)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if args.compare_to == ["all"]:
        compare_to_agents = valid_agents
    else:
        try:
            compare_to_agents = _parse_agents(
                [value.split(":", maxsplit=1)[0] for value in args.compare_to]
            )
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)

    all_agents = list(dict.fromkeys([source] + compare_to_agents))
    if args.yes:
        agents_to_generate = all_agents
    else:
        agents_to_generate = [
            agent
            for agent in all_agents
            if _confirm_overwrite(agent, args.maze, observable)
        ]

    if agents_to_generate:
        run_generation_experiment(
            agent_types=agents_to_generate,
            maze_name=args.maze,
            num_runs=args.num_runs,
            num_episodes=args.num_episodes,
            observable=observable,
            verbose=verbose,
            workers=args.workers,
            base_seed=args.seed,
            device=args.device,
            context_mode=args.context_mode,
            horizon=args.horizon,
        )

    run_inference_experiment(
        source_agents=[source],
        compare_to=compare_to,
        maze_name=args.maze,
        num_datasets=args.num_datasets or args.num_runs,
        observable=observable,
        verbose=verbose,
        workers=args.workers,
        device=args.device,
        base_seed=args.seed,
        source_context_mode=args.context_mode,
        horizon=args.horizon,
    )

    plot_policies = list(dict.fromkeys([source] + compare_to_agents))
    for policy in plot_policies:
        plot_aggregate_trajectory_stats(
            policy,
            maze_name=args.maze,
            observable=observable,
            cohort_policies=plot_policies,
            save=True,
            show=False,
            horizon=args.horizon,
        )

    plot_aggregate_comparison(
        source,
        compare_to,
        maze_name=args.maze,
        num_datasets=args.num_datasets or args.num_runs,
        observable=observable,
        save=True,
        show=False,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
