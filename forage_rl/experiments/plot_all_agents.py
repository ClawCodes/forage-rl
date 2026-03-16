"""Render the default all-agents plot bundle."""

import argparse
import matplotlib.pyplot as plt

from forage_rl.agents import registered_agents
from forage_rl.config import ensure_directories
from forage_rl.environments import (
    EnvironmentTarget,
    build_environment,
    normalize_environment_targets,
)
from forage_rl.experiments import scheduling
from forage_rl.utils import get_trajectory_run_ids, load_trajectories
from forage_rl.visualization import (
    plot_cumulative_sum_accuracy,
    plot_model_comparison,
    plot_model_accuracies_from_trajectory_type,
    plot_pairwise_accuracy_matrix,
    plot_pairwise_cumulative_accuracy,
    plot_pairwise_logprob_gap_matrix,
)
from forage_rl.visualization.plots import plot_mean_trajectory_stats


def render_all_agent_plots(
    env_target: EnvironmentTarget,
    agents: list[str],
    num_runs: int | None = None,
    show: bool = False,
    verbose: bool = True,
) -> None:
    """Render the default summary and per-source plot bundle."""
    agents = scheduling.validate_registered_agent_names(agents, "agents")
    scheduling.validate_optional_positive_int(num_runs, "num_runs")
    agents, unsupported_agents = scheduling.split_supported_agent_names(
        agents,
        env_target,
    )
    if unsupported_agents and verbose:
        print(
            scheduling.format_unsupported_agents_message(
                unsupported_agents,
                env_target,
            )
        )
    if not agents:
        return
    ensure_directories()
    maze = build_environment(env_target)

    if len(agents) >= 2:
        fig = plot_pairwise_accuracy_matrix(
            agents=agents,
            num_datasets=num_runs,
            env_key=env_target.key,
            env_label=env_target.label,
            save=True,
            show=show,
            verbose=verbose,
        )
        if fig is not None and not show:
            plt.close(fig)
        fig = plot_pairwise_logprob_gap_matrix(
            agents=agents,
            num_datasets=num_runs,
            env_key=env_target.key,
            env_label=env_target.label,
            save=True,
            show=show,
            verbose=verbose,
        )
        if fig is not None and not show:
            plt.close(fig)
        fig = plot_model_comparison(
            agents=agents,
            num_datasets=num_runs,
            env_key=env_target.key,
            env_label=env_target.label,
            save=True,
            show=show,
            verbose=verbose,
        )
        if fig is not None and not show:
            plt.close(fig)
        fig = plot_cumulative_sum_accuracy(
            agents=agents,
            num_datasets=num_runs,
            env_key=env_target.key,
            env_label=env_target.label,
            save=True,
            show=show,
            verbose=verbose,
        )
        if fig is not None and not show:
            plt.close(fig)

    for source in agents:
        compare_to = [agent for agent in agents if agent != source]
        source_run_ids = get_trajectory_run_ids(source, env_key=env_target.key)
        if num_runs is not None:
            source_run_ids = source_run_ids[:num_runs]
        source_runs = len(source_run_ids)

        if verbose:
            print(
                f"Rendering plots for {source} on {env_target.label} "
                f"using {source_runs} runs"
            )

        if source_runs == 0:
            continue

        fig = plot_model_accuracies_from_trajectory_type(
            source=source,
            compare_to=compare_to,
            num_datasets=source_runs,
            env_key=env_target.key,
            env_label=env_target.label,
            save=True,
            show=show,
            verbose=verbose,
        )
        if fig is not None and not show:
            plt.close(fig)

        trajectories = [
            load_trajectories(source, run_id, env_key=env_target.key)
            for run_id in source_run_ids
        ]
        fig = plot_mean_trajectory_stats(
            trajectories=trajectories,
            maze=maze,
            source=source,
            compare_to=compare_to,
            num_datasets=source_runs,
            env_key=env_target.key,
            env_label=env_target.label,
            save=True,
            show=show,
            verbose=verbose,
        )
        if fig is not None and not show:
            plt.close(fig)

        for evaluator in compare_to:
            fig = plot_pairwise_cumulative_accuracy(
                source=source,
                evaluator=evaluator,
                num_datasets=source_runs,
                env_key=env_target.key,
                env_label=env_target.label,
                save=True,
                show=show,
                verbose=verbose,
            )
            if fig is not None and not show:
                plt.close(fig)


def main() -> None:
    """Parse CLI args and render the all-agent plot bundle."""
    parser = argparse.ArgumentParser(
        description="Render all-agent model comparison plots"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Agent name(s) to include, or 'all' for every registered agent",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment target(s): simple:full, full:full, full:pomdp",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Optional cap on the number of saved runs to include per source",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display figures interactively"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()
    agents = scheduling.normalize_agent_names(
        registered_agents() if args.agents == ["all"] else args.agents
    )
    env_targets = normalize_environment_targets(args.envs)
    for env_target in env_targets:
        render_all_agent_plots(
            env_target=env_target,
            agents=agents,
            num_runs=args.num_runs,
            show=args.show,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
