"""Regenerate trajectory, inference, and figure artifacts across settings."""

from __future__ import annotations

import argparse

from forage_rl.agents.registry import (
    PolicySpec,
    neural_agents,
)
from forage_rl.config import DefaultParams
from forage_rl.experiments.artifact_scenarios import (
    ArtifactScenario,
    EvaluatorMode,
    benchmark_suite_scenarios,
    default_evaluators,
    default_sources,
    default_scenarios,
    filter_evaluators,
    neural_context_evaluators,
    neural_context_policies,
    reward_timing_evaluators,
    reward_timing_benchmark_scenarios,
    selected_settings,
)
from forage_rl.experiments.generate_trajectories import run_generation_experiment
from forage_rl.experiments.model_inference import run_inference_experiment
from forage_rl.experiments.reward_timing_benchmark import (
    analyze_reward_timing_benchmark,
)
from forage_rl.experiments.train_pretrained_agents import train_pretrained_agents
from forage_rl.visualization.plots import (
    plot_aggregate_comparison,
    plot_aggregate_trajectory_stats,
    plot_episode_return_comparison,
)

# Re-export the extracted scenario helpers to keep the existing module-level API stable.
_default_sources = default_sources
_default_evaluators = default_evaluators
_neural_context_policies = neural_context_policies
_neural_context_evaluators = neural_context_evaluators
_reward_timing_evaluators = reward_timing_evaluators
_selected_settings = selected_settings
_default_scenarios = default_scenarios
_reward_timing_benchmark_scenarios = reward_timing_benchmark_scenarios
_benchmark_suite_scenarios = benchmark_suite_scenarios
_filter_evaluators = filter_evaluators


def regenerate_artifacts(
    mazes: list[str] | None = None,
    observability: str = "all",
    num_runs: int = DefaultParams.NUM_TRAINING_RUNS,
    num_episodes: int = DefaultParams.NUM_TRAINING_EPISODES,
    num_datasets: int | None = None,
    workers: int | None = None,
    device: str = "auto",
    seed: int = 0,
    horizon: int | None = None,
    train_pretrained: bool = False,
    skip_generation: bool = False,
    skip_inference: bool = False,
    skip_figures: bool = False,
    reward_timing_benchmark: bool = False,
    benchmark_suite: bool = False,
    evaluator_mode: EvaluatorMode = "all",
    verbose: bool = True,
) -> None:
    """Rebuild trajectory datasets, inference outputs, and figures."""
    if sum((reward_timing_benchmark, benchmark_suite)) > 1:
        raise ValueError(
            "benchmark_suite and reward_timing_benchmark are mutually exclusive presets."
        )
    if evaluator_mode == "fresh" and train_pretrained:
        raise ValueError(
            "Fresh-only artifact regeneration does not use pretrained checkpoints; "
            "drop --train-pretrained or use --evaluator-mode all/pretrained."
        )

    if benchmark_suite:
        scenarios = benchmark_suite_scenarios()
        if verbose:
            print(
                "Using benchmark suite preset: full baseline and full context."
            )
    elif reward_timing_benchmark:
        scenarios = reward_timing_benchmark_scenarios()
        if verbose:
            print(
                "Using reward timing benchmark preset: maze=full, observable=PO, "
                "comparing prev_reward and prev_reward_time DQN/ELMAN/GRU/LSTM variants."
            )
    else:
        scenarios = default_scenarios(
            mazes,
            observability,
            verbose=verbose,
        )
    resolved_num_datasets = num_runs if num_datasets is None else num_datasets

    for scenario in scenarios:
        maze_name = scenario.maze_name
        observable = scenario.observable
        filtered_evaluators = filter_evaluators(
            scenario.evaluators,
            evaluator_mode,
        )
        obs_tag = "FO" if observable else "PO"
        if verbose:
            print(f"\n=== Setting: maze={maze_name}, observable={obs_tag} ===")

        if train_pretrained:
            for context_mode in scenario.train_context_modes:
                train_pretrained_agents(
                    agent_types=neural_agents(),
                    maze_name=maze_name,
                    observable=observable,
                    num_episodes=DefaultParams.NUM_EPISODES * 5,
                    device=device,
                    seed=seed,
                    verbose=verbose,
                    context_mode=context_mode,
                    horizon=horizon,
                )

        if not skip_generation:
            for context_mode in scenario.generation_context_modes:
                run_generation_experiment(
                    agent_types=list(scenario.source_agents),
                    maze_name=maze_name,
                    num_runs=num_runs,
                    num_episodes=num_episodes,
                    observable=observable,
                    verbose=verbose,
                    workers=workers,
                    base_seed=seed,
                    device=device,
                    context_mode=context_mode,
                    horizon=horizon,
                )

        if not skip_inference:
            for context_mode in scenario.inference_context_modes:
                run_inference_experiment(
                    source_agents=list(scenario.source_agents),
                    compare_to=list(filtered_evaluators),
                    maze_name=maze_name,
                    num_datasets=resolved_num_datasets,
                    observable=observable,
                    verbose=verbose,
                    workers=workers,
                    device=device,
                    base_seed=seed,
                    source_context_mode=context_mode,
                    horizon=horizon,
                )

        if not skip_figures:
            plot_episode_return_comparison(
                maze_name=maze_name,
                observable=observable,
                agents=list(scenario.figure_policies),
                save=True,
                show=False,
                filename_suffix=scenario.filename_suffix,
                benchmark_label=scenario.benchmark_label,
                benchmark_note=scenario.benchmark_note,
                horizon=horizon,
            )
            for source in scenario.figure_policies:
                plot_aggregate_trajectory_stats(
                    source,
                    maze_name=maze_name,
                    observable=observable,
                    cohort_policies=list(scenario.figure_policies),
                    save=True,
                    show=False,
                    filename_suffix=scenario.filename_suffix,
                    benchmark_label=scenario.benchmark_label,
                    benchmark_note=scenario.benchmark_note,
                    horizon=horizon,
                )
                plot_aggregate_comparison(
                    source,
                    list(filtered_evaluators),
                    maze_name=maze_name,
                    observable=observable,
                    save=True,
                    show=False,
                    filename_suffix=scenario.filename_suffix,
                    benchmark_label=scenario.benchmark_label,
                    benchmark_note=scenario.benchmark_note,
                    horizon=horizon,
                )
            if scenario.filename_suffix == "reward_timing_benchmark":
                analyze_reward_timing_benchmark(
                    maze_name=maze_name,
                    observable=observable,
                    policies=[
                        policy
                        for policy in scenario.figure_policies
                        if isinstance(policy, PolicySpec)
                    ],
                    num_datasets=resolved_num_datasets,
                    filename_suffix=scenario.filename_suffix,
                    horizon=horizon,
                    verbose=verbose,
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
        "--horizon",
        type=int,
        default=None,
        help="Optional episode-length override; default uses the built-in maze horizon.",
    )
    parser.add_argument(
        "--train-pretrained",
        action="store_true",
        help="Refresh canonical neural checkpoints before inference.",
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
    parser.add_argument(
        "--reward-timing-benchmark",
        action="store_true",
        help=(
            "Run the dedicated reward timing benchmark on full/PO with "
            "prev_reward and prev_reward_time DQN/ELMAN/GRU/LSTM variants."
        ),
    )
    parser.add_argument(
        "--benchmark-suite",
        action="store_true",
        help=(
            "Run the recommended benchmark suite on full/FO and full/PO."
        ),
    )
    parser.add_argument(
        "--evaluator-mode",
        choices=("all", "fresh", "pretrained"),
        default="all",
        help=(
            "Filter neural evaluator modes during inference and aggregate "
            "comparison plots. Tabular evaluators are always kept."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    args = parser.parse_args()
    if sum((args.reward_timing_benchmark, args.benchmark_suite)) > 1:
        parser.error(
            "--benchmark-suite and --reward-timing-benchmark are mutually exclusive."
        )
    if args.evaluator_mode == "fresh" and args.train_pretrained:
        parser.error(
            "--train-pretrained cannot be used with --evaluator-mode fresh "
            "because fresh-only runs do not load pretrained checkpoints."
        )
    regenerate_artifacts(
        mazes=args.mazes,
        observability=args.observability,
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        num_datasets=args.num_datasets,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        horizon=args.horizon,
        train_pretrained=args.train_pretrained,
        skip_generation=args.skip_generation,
        skip_inference=args.skip_inference,
        skip_figures=args.skip_figures,
        reward_timing_benchmark=args.reward_timing_benchmark,
        benchmark_suite=args.benchmark_suite,
        evaluator_mode=args.evaluator_mode,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
