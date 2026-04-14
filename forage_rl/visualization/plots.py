"""Visualization functions for model comparison and trajectory summaries."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl import RunDataset, Trajectory
from forage_rl.agents.registry import Agent, EvaluatorSpec, PolicySpec
from forage_rl.analysis.patch_timing import (
    aggregate_curves,
    extract_decision_rows,
    infer_hidden_states_for_trajectory,
    leave_probability_curve,
    oracle_optimal_dwell_by_state,
    oracle_residency_deviation_by_patch,
    normalized_curve_auc,
    observation_group_patch_labels,
)
from forage_rl.config import FIGURES_DIR, ensure_directories
from forage_rl.environments import resolve_effective_horizon
from forage_rl.environments.maze import (
    Maze,
    MazeMDP,
    MazePOMDP,
    maze_from_builtin_maze_spec,
)
from forage_rl.utils import (
    list_run_dataset_run_ids,
    load_logprobs,
    load_run_dataset,
)

PolicyInput = Agent | PolicySpec | str
EvaluatorInput = Agent | EvaluatorSpec


def _normalize_policy(policy: PolicyInput) -> PolicySpec:
    if isinstance(policy, str):
        raise ValueError(
            "String policy labels are supported only for recovery plotting helpers."
        )
    if isinstance(policy, PolicySpec):
        return policy
    return PolicySpec(agent=policy)


def _normalize_evaluator(evaluator: EvaluatorInput) -> EvaluatorSpec:
    if isinstance(evaluator, EvaluatorSpec):
        return evaluator
    return EvaluatorSpec(agent=evaluator, mode="fresh")


def _policy_label(policy: PolicyInput) -> str:
    if isinstance(policy, str):
        return policy
    return _normalize_policy(policy).display_label


def _policy_artifact_label(policy: PolicyInput) -> str:
    if isinstance(policy, str):
        return policy.lower().replace(" ", "_")
    return _normalize_policy(policy).artifact_label


def _evaluator_label(evaluator: EvaluatorInput) -> str:
    spec = _normalize_evaluator(evaluator)
    return spec.label


def _load_policy_run_ids(
    policy: PolicyInput,
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> list[int]:
    spec = _normalize_policy(policy)
    return list_run_dataset_run_ids(
        spec.agent,
        maze_name,
        observable,
        context_mode=spec.context_mode,
        horizon=horizon,
    )


def _load_policy_run_dataset(
    policy: PolicyInput,
    run_id: int,
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> RunDataset:
    spec = _normalize_policy(policy)
    return load_run_dataset(
        spec.agent,
        run_id,
        maze_name,
        observable,
        context_mode=spec.context_mode,
        horizon=horizon,
    )


def _flatten_run_dataset(run_dataset: RunDataset) -> Trajectory:
    return Trajectory(transitions=list(run_dataset.iter_transitions()))


def _common_run_ids(
    policies: list[PolicyInput],
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> list[int]:
    if not policies:
        return []

    run_id_sets = [
        set(_load_policy_run_ids(policy, maze_name, observable, horizon=horizon))
        for policy in policies
    ]
    if not run_id_sets:
        return []
    return sorted(set.intersection(*run_id_sets))


def _obs_tag(observable: bool) -> str:
    return "FO" if observable else "PO"


def _figure_suffix(maze_name: str, horizon: int | None) -> str:
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    return "" if horizon is None else f"_h{resolved_horizon}"


def _figure_title(base: str, benchmark_label: str | None = None) -> str:
    return f"{benchmark_label}: {base}" if benchmark_label else base


def _recovery_title_context(
    maze_name: str,
    observable: bool,
    condition_label: str | None,
) -> str:
    parts = [maze_name, _obs_tag(observable)]
    if condition_label:
        parts.append(condition_label)
    return ", ".join(parts)


def _finalize_figure(
    fig,
    *,
    save: bool,
    show: bool,
    filepath,
):
    if save:
        ensure_directories()
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_cumulative_accuracy(
    ax: plt.Axes,
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    horizon: int | None = None,
) -> None:
    """Draw per-evaluator win-rate-over-time lines onto ax."""
    source_spec = _normalize_policy(source)
    comparisons = [
        _normalize_evaluator(evaluator)
        for evaluator in compare_to
        if _evaluator_label(evaluator)
        != EvaluatorSpec(
            agent=source_spec.agent,
            mode="fresh",
            context_mode=source_spec.context_mode,
        ).label
    ]
    if not comparisons:
        ax.text(
            0.5, 0.5, "No comparisons", ha="center", va="center", transform=ax.transAxes
        )
        return

    run_ids = _load_policy_run_ids(source_spec, maze_name, observable, horizon=horizon)
    if num_datasets is not None:
        run_ids = run_ids[:num_datasets]
    if not run_ids:
        ax.text(
            0.5,
            0.5,
            f"No data for '{_policy_label(source_spec)}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    for evaluator in comparisons:
        accuracies = []
        for run_id in run_ids:
            source_cumsum = load_logprobs(
                source_spec.agent,
                EvaluatorSpec(
                    agent=source_spec.agent,
                    mode="fresh",
                    context_mode=source_spec.context_mode,
                ),
                run_id,
                maze_name,
                observable,
                source_context_mode=source_spec.context_mode,
                horizon=horizon,
            )
            eval_cumsum = load_logprobs(
                source_spec.agent,
                evaluator,
                run_id,
                maze_name,
                observable,
                source_context_mode=source_spec.context_mode,
                horizon=horizon,
            )
            accuracy = np.where(
                np.isclose(source_cumsum, eval_cumsum),
                0.5,
                (source_cumsum > eval_cumsum).astype(float),
            )
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies, axis=0)
        ax.plot(avg_accuracy, linewidth=3, label=evaluator.label)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Number of Observed Transitions", fontsize=16)
    ax.set_ylabel("Prediction Accuracy", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)


def _draw_model_accuracies(
    ax: plt.Axes,
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    horizon: int | None = None,
) -> None:
    """Draw paired win-rate bars for source vs each evaluator onto ax."""
    source_spec = _normalize_policy(source)
    comparisons = [
        _normalize_evaluator(evaluator)
        for evaluator in compare_to
        if _evaluator_label(evaluator)
        != EvaluatorSpec(
            agent=source_spec.agent,
            mode="fresh",
            context_mode=source_spec.context_mode,
        ).label
    ]
    if not comparisons:
        ax.text(
            0.5, 0.5, "No comparisons", ha="center", va="center", transform=ax.transAxes
        )
        return

    run_ids = _load_policy_run_ids(source_spec, maze_name, observable, horizon=horizon)
    if num_datasets is not None:
        run_ids = run_ids[:num_datasets]
    if not run_ids:
        ax.text(
            0.5,
            0.5,
            f"No data for '{_policy_label(source_spec)}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    source_wins = {evaluator.label: 0 for evaluator in comparisons}
    source_self = EvaluatorSpec(
        agent=source_spec.agent,
        mode="fresh",
        context_mode=source_spec.context_mode,
    )
    for run_id in run_ids:
        source_final = load_logprobs(
            source_spec.agent,
            source_self,
            run_id,
            maze_name,
            observable,
            source_context_mode=source_spec.context_mode,
            horizon=horizon,
        )[-1]
        for evaluator in comparisons:
            eval_final = load_logprobs(
                source_spec.agent,
                evaluator,
                run_id,
                maze_name,
                observable,
                source_context_mode=source_spec.context_mode,
                horizon=horizon,
            )[-1]
            if source_final > eval_final:
                source_wins[evaluator.label] += 1

    n = len(comparisons)
    x = np.arange(n)
    width = 0.35
    eval_colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    source_rates = [
        source_wins[evaluator.label] / len(run_ids) for evaluator in comparisons
    ]
    eval_rates = [1 - rate for rate in source_rates]

    ax.bar(
        x - width / 2,
        source_rates,
        width,
        label=_policy_label(source_spec),
        color="#3498db",
    )
    for index, (evaluator, rate) in enumerate(zip(comparisons, eval_rates)):
        ax.bar(
            x[index] + width / 2,
            rate,
            width,
            label=evaluator.label
            if index == 0
            or evaluator.label not in [e.label for e in comparisons[:index]]
            else "_nolegend_",
            color=eval_colors[index % len(eval_colors)],
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_xlabel("Comparison", fontsize=12)
    ax.set_title(
        f"Model Accuracy on '{_policy_label(source_spec)}'-Generated Trajectories",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{_policy_artifact_label(source_spec)} vs {evaluator.label}"
            for evaluator in comparisons
        ],
        ha="right",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (0.50)")
    ax.legend()

    for index, (source_rate, eval_rate) in enumerate(zip(source_rates, eval_rates)):
        ax.text(
            x[index] - width / 2,
            min(source_rate + 0.02, 0.97),
            f"{source_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            x[index] + width / 2,
            min(eval_rate + 0.02, 0.97),
            f"{eval_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _draw_mean_cumulative_reward(ax: plt.Axes, trajectories: list[Trajectory]) -> None:
    cumsums = [
        np.cumsum([transition.reward for transition in trajectory.transitions])
        for trajectory in trajectories
    ]
    min_len = min(len(cumsum) for cumsum in cumsums)
    arr = np.array([cumsum[:min_len] for cumsum in cumsums])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x = np.arange(min_len)
    ax.plot(x, mean, linewidth=2, color="#2ecc71", label="Mean")
    ax.fill_between(
        x, mean - std, mean + std, alpha=0.3, color="#2ecc71", label="±1 SD"
    )
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title(f"Mean Cumulative Reward (n={len(trajectories)})", fontsize=14)
    ax.legend(fontsize=10)


def _draw_modal_residency(
    ax: plt.Axes, trajectories: list[Trajectory], maze: Maze
) -> None:
    state_sequences = [
        [transition.state for transition in trajectory.transitions]
        for trajectory in trajectories
    ]
    min_len = min(len(sequence) for sequence in state_sequences)
    arr = np.array([sequence[:min_len] for sequence in state_sequences])

    if isinstance(maze, MazePOMDP):
        obs_map = maze._state_to_observation_group
        obs_arr = np.vectorize(obs_map.__getitem__)(arr)
        n_bins = maze.num_observations
        modal_values: list[int] = []
        frequencies: list[float] = []
        for step in range(min_len):
            counts = np.bincount(obs_arr[:, step], minlength=n_bins)
            modal = int(np.argmax(counts))
            modal_values.append(modal)
            frequencies.append(counts[modal] / len(trajectories))
        y_labels = maze.maze_spec.observation_labels
    else:
        n_bins = maze.num_states
        modal_values = []
        frequencies = []
        for step in range(min_len):
            counts = np.bincount(arr[:, step], minlength=n_bins)
            modal = int(np.argmax(counts))
            modal_values.append(modal)
            frequencies.append(counts[modal] / len(trajectories))
        y_labels = maze.state_labels or [f"State {state}" for state in range(n_bins)]

    ax.scatter(
        range(min_len),
        modal_values,
        s=np.array(frequencies) * 40,
        alpha=0.7,
        color="#3498db",
    )
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title(f"Modal Residency Location (n={len(trajectories)})", fontsize=14)


def plot_mean_trajectory_stats(
    trajectories: list[Trajectory],
    maze: MazeMDP,
    source: PolicyInput,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
):
    fig, (ax_reward, ax_residency) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )
    _draw_mean_cumulative_reward(ax_reward, trajectories)
    _draw_modal_residency(ax_residency, trajectories, maze)
    fig.suptitle(
        _figure_title(
            f"Average Trajectory Overview: '{_policy_label(source)}' (n={len(trajectories)})",
            benchmark_label,
        ),
        fontsize=16,
        fontweight="bold",
    )

    filename = f"mean_trajectory_stats_{_policy_artifact_label(source)}_{maze.maze_spec.maze.name}_{_obs_tag(not isinstance(maze, MazePOMDP))}"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def plot_aggregate_trajectory_stats(
    source: PolicyInput,
    maze_name: str = "simple",
    observable: bool = True,
    cohort_policies: list[PolicyInput] | None = None,
    run_ids: list[int] | None = None,
    save: bool = True,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    benchmark_note: str | None = None,
    horizon: int | None = None,
) -> None:
    del benchmark_note
    source_spec = _normalize_policy(source)
    selected_run_ids = (
        run_ids
        if run_ids is not None
        else _common_run_ids(
            [source_spec, *(cohort_policies or [])],
            maze_name,
            observable,
            horizon=horizon,
        )
    )
    if not selected_run_ids:
        selected_run_ids = _load_policy_run_ids(
            source_spec, maze_name, observable, horizon=horizon
        )

    trajectories = [
        _flatten_run_dataset(
            _load_policy_run_dataset(
                source_spec,
                run_id,
                maze_name,
                observable,
                horizon=horizon,
            )
        )
        for run_id in selected_run_ids
    ]
    maze = maze_from_builtin_maze_spec(
        maze_name,
        observable,
        horizon=resolve_effective_horizon(maze_name, horizon),
    )
    plot_mean_trajectory_stats(
        trajectories,
        maze,
        source_spec,
        save=save,
        show=show,
        filename_suffix=filename_suffix,
        benchmark_label=benchmark_label,
    )


def plot_episode_return_comparison(
    maze_name: str,
    observable: bool,
    agents: list[PolicyInput] | None = None,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    benchmark_note: str | None = None,
    horizon: int | None = None,
):
    del benchmark_note
    policies = [_normalize_policy(agent) for agent in (agents or list(Agent))]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    plotted_any = False
    for policy in policies:
        run_ids = _load_policy_run_ids(policy, maze_name, observable, horizon=horizon)
        if not run_ids:
            continue
        episode_return_sequences = []
        for run_id in run_ids:
            run_dataset = _load_policy_run_dataset(
                policy,
                run_id,
                maze_name,
                observable,
                horizon=horizon,
            )
            episode_returns = np.array(
                [
                    sum(transition.reward for transition in trajectory.transitions)
                    for trajectory in run_dataset
                ],
                dtype=float,
            )
            episode_return_sequences.append(episode_returns)

        min_len = min(len(sequence) for sequence in episode_return_sequences)
        arr = np.array([sequence[:min_len] for sequence in episode_return_sequences])
        x = np.arange(1, min_len + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x, mean, linewidth=2, label=_policy_label(policy))
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No run datasets available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Episode Within Run", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title(
        _figure_title(
            f"Episode Return Comparison ({maze_name}, {_obs_tag(observable)})",
            benchmark_label,
        ),
        fontsize=14,
    )
    if plotted_any:
        ax.legend(fontsize=10)

    filename = f"episode_return_comparison_{maze_name}_{_obs_tag(observable)}{_figure_suffix(maze_name, horizon)}"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def _extract_run_decision_rows(
    run_dataset: RunDataset,
    *,
    maze_name: str,
    maze: Maze,
    observable: bool,
) -> list:
    if observable:
        patch_labels = {
            int(state_spec.id): state_spec.label for state_spec in maze.maze_spec.states
        }
    else:
        patch_labels = observation_group_patch_labels(maze_name)

    rows = []
    for trajectory in run_dataset:
        resolved_states = (
            None
            if observable
            else infer_hidden_states_for_trajectory(trajectory, maze=maze)
        )
        rows.extend(
            extract_decision_rows(
                trajectory,
                patch_labels=patch_labels,
                resolved_states=resolved_states,
            )
        )
    return rows


def plot_patch_timing_summary(
    source: PolicyInput,
    maze_name: str = "simple",
    observable: bool = True,
    run_ids: list[int] | None = None,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    horizon: int | None = None,
):
    source_spec = _normalize_policy(source)
    selected_run_ids = (
        run_ids
        if run_ids is not None
        else _load_policy_run_ids(source_spec, maze_name, observable, horizon=horizon)
    )
    if not selected_run_ids:
        return None

    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    maze = maze_from_builtin_maze_spec(
        maze_name,
        observable,
        horizon=resolved_horizon,
    )
    leave_action = maze.action_labels.index("leave")
    optimal_dwell_by_state = oracle_optimal_dwell_by_state(
        maze_name=maze_name,
        horizon=resolved_horizon,
    )
    patch_order = list(
        maze.maze_spec.observation_labels or ["Upper Patch", "Lower Patch"]
    )
    deviation_offset = resolved_horizon
    curve_width = resolved_horizon * 2 + 1

    deviations_by_patch = {patch_label: [] for patch_label in patch_order}
    curves_by_patch = {patch_label: [] for patch_label in patch_order}

    for run_id in selected_run_ids:
        run_dataset = _load_policy_run_dataset(
            source_spec,
            run_id,
            maze_name,
            observable,
            horizon=horizon,
        )
        rows = _extract_run_decision_rows(
            run_dataset,
            maze_name=maze_name,
            maze=maze,
            observable=observable,
        )
        run_deviations = oracle_residency_deviation_by_patch(
            rows,
            leave_action=leave_action,
            optimal_dwell_by_state=optimal_dwell_by_state,
        )
        for patch_label, deviations in run_deviations.items():
            deviations_by_patch.setdefault(patch_label, []).extend(deviations)
            curve = leave_probability_curve(
                rows,
                value_getter=lambda row, optimal=optimal_dwell_by_state, offset=deviation_offset: (
                    row.time_spent + 1 - optimal[row.state] + offset
                ),
                leave_action=leave_action,
                max_value=curve_width,
                patch_label=patch_label,
            )
            if np.any(np.isfinite(curve)):
                curves_by_patch.setdefault(patch_label, []).append(curve)

    fig, (ax_deviation, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(15, 5),
        constrained_layout=True,
    )

    plotted_deviations = [
        deviations_by_patch.get(patch_label, []) for patch_label in patch_order
    ]
    if any(plotted_deviations):
        boxplot = ax_deviation.boxplot(
            plotted_deviations,
            tick_labels=patch_order,
            patch_artist=True,
        )
        for patch, patch_artist in zip(
            boxplot["boxes"], ["#3498db", "#e67e22"], strict=False
        ):
            patch.set_facecolor(patch_artist)
            patch.set_alpha(0.45)
        ax_deviation.axhline(0.0, color="gray", linestyle="--", alpha=0.8)
    else:
        ax_deviation.text(
            0.5,
            0.5,
            "No leave decisions available",
            ha="center",
            va="center",
            transform=ax_deviation.transAxes,
        )
    ax_deviation.set_ylabel("Actual Dwell - Oracle Optimal Dwell", fontsize=12)
    ax_deviation.set_title("Patch Residency Deviation vs Oracle", fontsize=14)

    color_by_patch = {
        "Upper Patch": "#3498db",
        "Lower Patch": "#e67e22",
    }
    plotted_curves = False
    for patch_label in patch_order:
        patch_curves = curves_by_patch.get(patch_label, [])
        if not patch_curves:
            continue
        summary = aggregate_curves(patch_curves)
        x = summary.x - deviation_offset
        auc = normalized_curve_auc(summary.mean)
        color = color_by_patch.get(patch_label, "#34495e")
        ax_curve.plot(
            x,
            summary.mean,
            linewidth=2,
            color=color,
            label=f"{patch_label} (AUC={auc:.2f})",
        )
        ax_curve.fill_between(
            x,
            summary.mean - summary.std,
            summary.mean + summary.std,
            alpha=0.2,
            color=color,
        )
        plotted_curves = True

    if not plotted_curves:
        ax_curve.text(
            0.5,
            0.5,
            "No leave-probability curves available",
            ha="center",
            va="center",
            transform=ax_curve.transAxes,
        )
    else:
        ax_curve.legend(fontsize=10)
    ax_curve.axvline(0.0, color="gray", linestyle="--", alpha=0.8)
    ax_curve.set_ylim(0.0, 1.0)
    ax_curve.set_xlabel("Dwell Deviation From Oracle Optimal", fontsize=12)
    ax_curve.set_ylabel("Leave Probability", fontsize=12)
    ax_curve.set_title("Leave Probability vs Oracle Deviation", fontsize=14)

    fig.suptitle(
        _figure_title(
            f"Patch Timing Summary: '{_policy_label(source_spec)}' ({maze_name}, {_obs_tag(observable)})",
            benchmark_label,
        ),
        fontsize=16,
        fontweight="bold",
    )

    filename = (
        f"patch_timing_{_policy_artifact_label(source_spec)}_"
        f"{maze_name}_{_obs_tag(observable)}{_figure_suffix(maze_name, horizon)}"
    )
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def plot_aggregate_comparison(
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    benchmark_note: str | None = None,
    horizon: int | None = None,
):
    del benchmark_note
    fig, (ax_bars, ax_lines) = plt.subplots(
        1, 2, figsize=(16, 6), constrained_layout=True
    )
    _draw_model_accuracies(
        ax_bars,
        source,
        compare_to,
        maze_name,
        num_datasets,
        observable,
        horizon=horizon,
    )
    _draw_cumulative_accuracy(
        ax_lines,
        source,
        compare_to,
        maze_name,
        num_datasets,
        observable,
        horizon=horizon,
    )

    fig.suptitle(
        _figure_title(
            f"Model Comparison: '{_policy_label(source)}' vs evaluators",
            benchmark_label,
        ),
        fontsize=16,
        fontweight="bold",
    )

    comparisons = (
        "_".join([_evaluator_label(evaluator) for evaluator in compare_to]) or "none"
    )
    filename = (
        f"agg_compare_{_policy_artifact_label(source)}_to_{comparisons}_"
        f"{maze_name}_{_obs_tag(observable)}{_figure_suffix(maze_name, horizon)}"
    )
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def _policy_series_items(
    data_by_policy: dict[PolicyInput, list[np.ndarray] | list[float]],
) -> list[tuple[str, list[np.ndarray] | list[float]]]:
    items = [(_policy_label(policy), values) for policy, values in data_by_policy.items()]
    return sorted(items, key=lambda item: item[0])


def plot_recovery_curve_comparison(
    curves_by_policy: dict[PolicyInput, list[np.ndarray]],
    *,
    maze_name: str = "simple",
    observable: bool = True,
    perturbation_label: str = "decay_swap",
    condition_label: str | None = None,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
):
    """Plot mean absolute recovery curves with run-level variability."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    plotted_any = False
    for label, curves in _policy_series_items(curves_by_policy):
        if not curves:
            continue
        summary = aggregate_curves(curves)
        x = summary.x + 1
        ax.plot(x, summary.mean, linewidth=2, label=label)
        ax.fill_between(
            x,
            summary.mean - summary.std,
            summary.mean + summary.std,
            alpha=0.2,
        )
        plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No recovery curves available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.legend(fontsize=10)

    ax.set_xlabel("Post-Perturbation Episode", fontsize=12)
    ax.set_ylabel("Mean Absolute Dwell Deviation", fontsize=12)
    ax.set_title(
        _figure_title(
            f"Recovery Curves ({_recovery_title_context(maze_name, observable, condition_label)})",
            benchmark_label,
        ),
        fontsize=14,
    )

    filename = f"recovery_curve_comparison_{maze_name}_{_obs_tag(observable)}_{perturbation_label}"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def plot_signed_recovery_curve_comparison(
    curves_by_policy: dict[PolicyInput, list[np.ndarray]],
    *,
    maze_name: str = "simple",
    observable: bool = True,
    perturbation_label: str = "decay_swap",
    condition_label: str | None = None,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
):
    """Plot mean signed recovery curves to distinguish under- and over-stay."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    plotted_any = False
    for label, curves in _policy_series_items(curves_by_policy):
        if not curves:
            continue
        summary = aggregate_curves(curves)
        x = summary.x + 1
        ax.plot(x, summary.mean, linewidth=2, label=label)
        ax.fill_between(
            x,
            summary.mean - summary.std,
            summary.mean + summary.std,
            alpha=0.2,
        )
        plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No signed recovery curves available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.legend(fontsize=10)

    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.8)
    ax.set_xlabel("Post-Perturbation Episode", fontsize=12)
    ax.set_ylabel("Mean Signed Dwell Deviation", fontsize=12)
    ax.set_title(
        _figure_title(
            f"Signed Recovery Curves ({_recovery_title_context(maze_name, observable, condition_label)})",
            benchmark_label,
        ),
        fontsize=14,
    )

    filename = f"signed_recovery_curve_comparison_{maze_name}_{_obs_tag(observable)}_{perturbation_label}"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def plot_recovery_auc_comparison(
    aucs_by_policy: dict[PolicyInput, list[float]],
    *,
    maze_name: str = "simple",
    observable: bool = True,
    perturbation_label: str = "decay_swap",
    condition_label: str | None = None,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
):
    """Plot mean recovery AUC by policy with run-level standard deviation."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for label, values in _policy_series_items(aucs_by_policy):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        labels.append(label)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=0)))

    if not labels:
        ax.text(
            0.5,
            0.5,
            "No recovery AUC values available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        positions = np.arange(len(labels))
        ax.bar(positions, means, yerr=stds, alpha=0.7, color="#2c7fb8", capsize=4)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")

    ax.set_ylabel("Recovery AUC", fontsize=12)
    ax.set_title(
        _figure_title(
            f"Recovery AUC Comparison ({_recovery_title_context(maze_name, observable, condition_label)})",
            benchmark_label,
        ),
        fontsize=14,
    )

    filename = f"recovery_auc_comparison_{maze_name}_{_obs_tag(observable)}_{perturbation_label}"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig
