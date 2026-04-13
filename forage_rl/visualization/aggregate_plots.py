"""Aggregate plotting implementations extracted from the main plots module."""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_mean_cumulative_reward_impl(
    reward_sequences: list[np.ndarray],
    *,
    save: bool,
    show: bool,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    deps["draw_mean_cumulative_reward"](ax, reward_sequences)
    if save:
        deps["ensure_directories"]()
        filepath = deps["figures_dir"] / "mean_cumulative_reward.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_modal_residency_impl(
    state_sequences: list[np.ndarray],
    maze,
    *,
    save: bool,
    show: bool,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    deps["draw_residency_fractions"](ax, state_sequences, maze)
    if save:
        deps["ensure_directories"]()
        filepath = deps["figures_dir"] / "modal_residency.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_mean_trajectory_stats_impl(
    reward_sequences: list[np.ndarray],
    state_sequences: list[np.ndarray],
    maze,
    source,
    *,
    save: bool,
    show: bool,
    run_count: int | None,
    episodes_per_run: str | int | None,
    transitions_per_run: str | int | None,
    plotted_episodes: int | None,
    matched_run_ids,
    excluded_cohorts,
    setting_note: str | None,
    filename_suffix: str | None,
    benchmark_label: str | None,
    benchmark_note: str | None,
    timing_summary,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    if timing_summary is None:
        fig, (ax_reward, ax_residency) = plt.subplots(
            1, 2, figsize=(14, 5), constrained_layout=True
        )
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
        ax_reward, ax_residency, ax_leave, ax_timing = axes.flat
    axis_label = deps["aggregate_trajectory_axis_label"]()
    resolved_plotted_episodes = (
        plotted_episodes if plotted_episodes is not None else len(reward_sequences)
    )
    effective_setting_note = setting_note
    if effective_setting_note is None:
        effective_setting_note = deps["setting_note"](
            maze.maze_spec.maze.name,
            not deps["is_pomdp"](maze),
        )

    if not reward_sequences or not state_sequences:
        ax_reward.text(0.5, 0.5, "No episode data", ha="center", va="center")
        ax_residency.text(0.5, 0.5, "No episode data", ha="center", va="center")
        if timing_summary is not None:
            ax_leave.text(0.5, 0.5, "No episode data", ha="center", va="center")
            ax_timing.text(0.5, 0.5, "No episode data", ha="center", va="center")
        fig.suptitle(
            deps["benchmark_title"](
                benchmark_label,
                "Aggregate Trajectory Overview\n"
                f"{deps['compact_source_policy_line'](source)}",
            ),
            fontsize=16,
            fontweight="bold",
        )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    deps["draw_mean_cumulative_reward"](
        ax_reward,
        reward_sequences,
        axis_label=axis_label,
        sample_label="episodes",
        single_label="Episode",
    )
    deps["draw_cumulative_residency_shares"](
        ax_residency,
        state_sequences,
        maze,
        axis_label=axis_label,
        sample_label="Episodes",
        single_label="episode",
    )
    if timing_summary is not None:
        deps["draw_patch_leave_probability_by_time"](ax_leave, timing_summary)
        deps["draw_patch_timing_summary"](ax_timing, timing_summary)

    title_prefix = (
        "Trajectory Overview"
        if resolved_plotted_episodes == 1
        else "Aggregate Trajectory Overview"
    )
    title = f"{title_prefix}\n{deps['compact_source_policy_line'](source)}"
    fig.suptitle(
        deps["benchmark_title"](benchmark_label, title),
        fontsize=16,
        fontweight="bold",
    )

    notes: list[str] = []
    notes.append(deps["trajectory_source_policy_description"](source))
    notes.append(
        "Aggregate trajectory figures summarize source policies only; "
        "fresh/pretrained evaluator modes apply to likelihood comparison plots."
    )
    notes.append(f"horizon={maze.horizon}")
    if run_count is not None:
        episodes_label = (
            str(episodes_per_run) if episodes_per_run is not None else "unknown"
        )
        notes.append(
            deps["trajectory_sample_metadata_note"](
                run_count,
                episodes_label,
                resolved_plotted_episodes,
            )
        )
        low_sample_note = deps["low_sample_note"](run_count, resolved_plotted_episodes)
        if low_sample_note is not None:
            notes.append(low_sample_note)
        coarse_note = deps["cumulative_residency_coarse_sample_note"](
            resolved_plotted_episodes
        )
        if coarse_note is not None:
            notes.append(coarse_note)
    if matched_run_ids:
        notes.append(
            f"matched_run_ids={deps['format_run_id_label'](list(matched_run_ids))}"
        )
    if excluded_cohorts:
        notes.append(f"excluded_cohorts={len(excluded_cohorts)}")
    if benchmark_note is not None:
        notes.append(benchmark_note)
    if effective_setting_note is not None:
        notes.append(effective_setting_note)
    notes.append(deps["cumulative_residency_interpretation_note"]())
    if timing_summary is not None:
        notes.append(
            "Bottom-left panel shows leave probability conditioned on time spent in patch."
        )
        notes.append(
            "Bottom-right panel shows signed deviation from the MVT-optimal dwell and normalized leave-time AUC."
        )
        if getattr(timing_summary, "uses_hidden_state_inference", False):
            notes.append(
                "Patch timing in PO settings uses hidden-state reconstruction from observed rewards and patch transitions."
            )
    deps["add_figure_notes"](fig, notes)

    if save:
        deps["ensure_directories"]()
        obs_tag = "PO" if deps["is_pomdp"](maze) else "FO"
        horizon_suffix = deps["figure_horizon_suffix"](
            maze.maze_spec.maze.name,
            maze.horizon,
        )
        filename = (
            f"mean_trajectory_stats_{deps['policy_artifact_label'](source)}_"
            f"{maze.maze_spec.maze.name}_{obs_tag}{horizon_suffix}"
        )
        if filename_suffix is not None:
            filename = f"{filename}_{filename_suffix}"
        filepath = deps["figures_dir"] / f"{filename}.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_aggregate_trajectory_stats_impl(
    source,
    *,
    maze_name: str,
    observable: bool,
    cohort_policies,
    run_ids,
    save: bool,
    show: bool,
    filename_suffix: str | None,
    benchmark_label: str | None,
        benchmark_note: str | None,
        horizon: int | None,
        deps: dict[str, Any],
):
    resolved_horizon = deps["resolve_effective_horizon"](maze_name, horizon)
    cohort = deps["resolve_aggregate_trajectory_cohort"](
        source,
        maze_name,
        observable,
        cohort_policies=cohort_policies,
        run_ids=run_ids,
        horizon=horizon,
    )
    if cohort.horizon != resolved_horizon:
        raise ValueError(
            "Aggregate trajectory plotting requires one exact horizon; "
            f"resolved horizon={resolved_horizon}, cohort horizon={cohort.horizon}."
        )
    selected_run_ids = list(cohort.run_ids)
    run_datasets = [
        (
            deps["load_run_dataset_for_policy"](
                source,
                run_id,
                maze_name,
                observable,
            )
            if horizon is None
            else deps["load_run_dataset_for_policy"](
                source,
                run_id,
                maze_name,
                observable,
                horizon=horizon,
            )
        )
        for run_id in selected_run_ids
    ]
    maze = deps["maze_from_builtin_maze_spec"](
        maze_name,
        observable,
        horizon=cohort.horizon,
    )
    reward_sequences, state_sequences = deps["episode_level_sequences"](
        run_datasets,
        selected_run_ids,
        cohort.horizon,
        expected_episodes_per_run=cohort.episodes_per_run,
        expected_transitions_per_run=cohort.transitions_per_run,
    )
    return deps["plot_mean_trajectory_stats"](
        reward_sequences,
        state_sequences,
        maze,
        source,
        save=save,
        show=show,
        run_count=len(selected_run_ids),
        episodes_per_run=cohort.episodes_per_run,
        transitions_per_run=cohort.transitions_per_run,
        plotted_episodes=cohort.plotted_episodes,
        matched_run_ids=selected_run_ids,
        excluded_cohorts=cohort.excluded_cohorts,
        setting_note=deps["setting_note"](maze_name, observable),
        filename_suffix=filename_suffix,
        benchmark_label=benchmark_label,
        benchmark_note=benchmark_note,
        timing_trajectories=[
            trajectory
            for run_dataset in run_datasets
            for trajectory in run_dataset.trajectories
        ],
    )


def plot_episode_return_comparison_impl(
    *,
    maze_name: str,
    observable: bool,
    agents,
    save: bool,
    show: bool,
    filename_suffix: str | None,
    benchmark_label: str | None,
    benchmark_note: str | None,
    horizon: int | None,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    resolved_horizon = deps["resolve_effective_horizon"](maze_name, horizon)
    selected_agents = deps["registered_agents"]() if agents is None else agents
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    plotted = False
    run_counts: list[int] = []
    episode_counts: list[int] = []
    plotted_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, str, dict[str, object]]] = []
    include_context = deps["include_context_labels"](policies=selected_agents)
    for agent in selected_agents:
        run_ids = deps["list_run_ids_for_policy"](
            agent,
            maze_name,
            observable,
            horizon=resolved_horizon,
        )
        if not run_ids:
            continue

        run_datasets = []
        for run_id in run_ids:
            metadata = deps["load_run_dataset_metadata_for_policy"](
                agent,
                run_id,
                maze_name,
                observable,
                horizon=resolved_horizon,
            )
            metadata_horizon = int(metadata["horizon"])
            if metadata_horizon != resolved_horizon:
                raise ValueError(
                    "Episode-return plotting requires one exact horizon; "
                    f"run_id={run_id} has horizon={metadata_horizon}, expected {resolved_horizon}."
                )
            run_datasets.append(
                deps["load_run_dataset_for_policy"](
                    agent,
                    run_id,
                    maze_name,
                    observable,
                    horizon=resolved_horizon,
                )
            )
        episode_returns = [
            np.array(
                [
                    sum(transition.reward for transition in trajectory.transitions)
                    for trajectory in run_dataset
                ],
                dtype=float,
            )
            for run_dataset in run_datasets
        ]
        min_len = min(len(sequence) for sequence in episode_returns)
        arr = np.array([sequence[:min_len] for sequence in episode_returns])
        mean = arr.mean(axis=0)
        x = np.arange(1, min_len + 1)
        std = arr.std(axis=0) if len(episode_returns) > 1 else None

        plotted_entries.append(
            (
                x,
                mean,
                std,
                deps["policy_display_label"](agent, include_context=include_context),
                deps["policy_line_style"](agent),
            )
        )

        plotted = True
        run_counts.append(len(run_ids))
        episode_counts.append(min_len)

    single_episode_mode = plotted and all(count == 1 for count in episode_counts)

    if not plotted:
        ax.text(0.5, 0.5, "No data for selected agents", ha="center", va="center")
        title = (
            f"Episode Return by Training Episode ({maze_name}, "
            f"{'FO' if observable else 'PO'}, runs=0, episodes_per_run=0)"
        )
    else:
        if single_episode_mode:
            for x, mean, std, label, style in plotted_entries:
                yerr = None if std is None else np.array([std[0]])
                ax.errorbar(
                    [float(x[0])],
                    [float(mean[0])],
                    yerr=yerr,
                    fmt="o",
                    linestyle="none",
                    capsize=4,
                    markersize=8,
                    color=style["color"],
                    alpha=style.get("alpha", 1.0),
                    label=label,
                )
            ax.set_xticks([1])
            ax.set_xlim(0.75, 1.25)
            title = (
                "Episode-1 Return by Agent "
                f"({maze_name}, {'FO' if observable else 'PO'}, "
                f"runs={deps['count_label'](run_counts)}, episodes_per_run={deps['count_label'](episode_counts)})"
            )
        else:
            for x, mean, std, label, style in plotted_entries:
                ax.plot(x, mean, label=label, **style)
                if std is not None:
                    ax.fill_between(
                        x,
                        mean - std,
                        mean + std,
                        color=style["color"],
                        alpha=0.18,
                    )
        ax.set_xlabel("Episode Within Run", fontsize=12)
        ax.set_ylabel("Episode Return", fontsize=12)
        ax.legend(fontsize=10)
        if not single_episode_mode:
            title = (
                "Episode Return by Training Episode "
                f"({maze_name}, {'FO' if observable else 'PO'}, "
                f"runs={deps['count_label'](run_counts)}, episodes_per_run={deps['count_label'](episode_counts)})"
            )
    ax.set_title(deps["benchmark_title"](benchmark_label, title), fontsize=14)

    notes: list[str] = [f"horizon={resolved_horizon}"]
    if benchmark_note is not None:
        notes.append(benchmark_note)
    setting_note = deps["setting_note"](maze_name, observable)
    if setting_note is not None:
        notes.append(setting_note)
    if single_episode_mode:
        notes.append(
            "Single-episode diagnostic only: one return point per agent; "
            "do not interpret this figure as a learning curve."
        )
    if run_counts and max(run_counts) < 2:
        notes.append(f"Low sample size (runs={deps['count_label'](run_counts)})")
    deps["add_figure_notes"](fig, notes)

    if save:
        deps["ensure_directories"]()
        filename = (
            "episode_return_comparison_"
            f"{maze_name}_{'FO' if observable else 'PO'}"
            f"{deps['figure_horizon_suffix'](maze_name, resolved_horizon)}"
        )
        if filename_suffix is not None:
            filename = f"{filename}_{filename_suffix}"
        filepath = deps["figures_dir"] / f"{filename}.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_aggregate_comparison_impl(
    source,
    compare_to,
    *,
    maze_name: str,
    num_datasets: int | None,
    observable: bool,
    save: bool,
    show: bool,
    filename_suffix: str | None,
    benchmark_label: str | None,
    benchmark_note: str | None,
    horizon: int | None,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    fig, (ax_bars, ax_lines) = plt.subplots(
        1, 2, figsize=(18, 6), constrained_layout=True
    )
    resolved_horizon = deps["resolve_effective_horizon"](maze_name, horizon)

    run_count = deps["draw_model_accuracies"](
        ax_bars,
        source,
        compare_to,
        maze_name,
        num_datasets,
        observable,
        horizon=horizon,
    )
    deps["draw_running_win_rate"](
        ax_lines,
        source,
        compare_to,
        maze_name,
        num_datasets,
        observable,
        horizon=horizon,
    )

    diagnostic_title = (
        "Source-Likelihood Diagnostic on Saved Source Trajectories"
        if filename_suffix is not None
        else "Source-Centric Model Comparison on Saved Source Trajectories"
    )
    title = deps["benchmark_title"](
        benchmark_label,
        f"{diagnostic_title}\n{deps['source_policy_description'](source)}",
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    notes: list[str] = [f"horizon={resolved_horizon}"]
    if benchmark_note is not None:
        notes.append(benchmark_note)
    setting_note = deps["setting_note"](maze_name, observable)
    if setting_note is not None:
        notes.append(setting_note)
    notes.append(deps["source_lead_note"])
    if run_count < 2:
        notes.append(f"Low sample size (runs={run_count})")
    deps["add_figure_notes"](fig, notes)

    if save:
        deps["ensure_directories"]()
        filepath = deps["figures_dir"] / deps["aggregate_comparison_filename"](
            source,
            compare_to,
            maze_name,
            observable,
            horizon=horizon,
            filename_suffix=filename_suffix,
        )
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
