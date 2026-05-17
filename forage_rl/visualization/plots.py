"""Visualization functions for model comparison and trajectory summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from forage_rl.types import RunDataset, Trajectory
from forage_rl.agents.registry import Agent, EvaluatorSpec, PolicySpec
from forage_rl.analysis.patch_timing import (
    aggregate_curves,
    extract_decision_rows,
    infer_hidden_states_for_trajectory,
    oracle_optimal_dwell_by_state,
    oracle_residency_deviation_by_patch,
    observation_group_patch_labels,
    state_patch_labels,
)
from forage_rl.config import FIGURES_DIR, ensure_output_directories
from forage_rl.environments import resolve_effective_horizon
from forage_rl.environments.maze import Maze, maze_from_builtin_maze_spec
from forage_rl.utils import (
    list_run_dataset_run_ids,
    load_logprobs,
    load_run_dataset,
)

PolicyInput = Agent | PolicySpec | str
EvaluatorInput = Agent | EvaluatorSpec


def _normalize_policy(policy: PolicyInput) -> PolicySpec:
    if isinstance(policy, PolicySpec):
        return policy
    if isinstance(policy, Agent):
        return PolicySpec(agent=policy)
    if isinstance(policy, str):
        raise ValueError(
            "String policy labels are supported only for recovery plotting helpers."
        )
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
        ensure_output_directories()
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
        ax.plot(avg_accuracy, linewidth=3, label=f"{source.value} vs. {evaluator.label}")

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

    if not maze.observable:
        n_bins = maze.num_observations
        y_labels = maze.maze_spec.observation_labels
    else:
        n_bins = maze.num_states
        y_labels = maze.state_labels or [f"State {state}" for state in range(n_bins)]

    modal_values: list[int] = []
    frequencies: list[float] = []
    for step in range(min_len):
        counts = np.bincount(arr[:, step], minlength=n_bins)
        modal = int(np.argmax(counts))
        modal_values.append(modal)
        frequencies.append(counts[modal] / len(trajectories))

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
    maze: Maze,
    source: PolicyInput,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    filepath: Path | None = None,
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

    if filepath is None:
        filename = f"mean_trajectory_stats_{_policy_artifact_label(source)}_{maze.maze_spec.maze.name}_{_obs_tag(maze.observable)}"
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
    filepath: Path | None = None,
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
        filepath=filepath,
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
    filepath: Path | None = None,
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

    if filepath is None:
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
        patch_labels = state_patch_labels(maze_name)
    else:
        patch_labels = observation_group_patch_labels(maze_name)
    inference_maze = (
        maze
        if observable
        else maze_from_builtin_maze_spec(maze_name, True, horizon=maze.horizon)
    )

    rows = []
    for trajectory in run_dataset:
        resolved_states = (
            None
            if observable
            else infer_hidden_states_for_trajectory(trajectory, maze=inference_maze)
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
    filepath: Path | None = None,
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
    patch_label_by_state = state_patch_labels(maze_name)
    patch_order: list[str] = []
    seen_patch_labels: set[str] = set()
    for state_spec in maze.maze_spec.states:
        if int(state_spec.id) not in optimal_dwell_by_state:
            continue
        patch_label = patch_label_by_state[int(state_spec.id)]
        if patch_label in seen_patch_labels:
            continue
        patch_order.append(patch_label)
        seen_patch_labels.add(patch_label)
    deviations_by_patch = {patch_label: [] for patch_label in patch_order}

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
        rows = [row for row in rows if row.state in optimal_dwell_by_state]
        if not rows:
            continue
        run_deviations = oracle_residency_deviation_by_patch(
            rows,
            leave_action=leave_action,
            optimal_dwell_by_state=optimal_dwell_by_state,
        )
        for patch_label, deviations in run_deviations.items():
            deviations_by_patch.setdefault(patch_label, []).extend(deviations)
    fig, ax_deviation = plt.subplots(figsize=(8, 5), constrained_layout=True)

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

    fig.suptitle(
        _figure_title(
            f"Patch Timing Summary: '{_policy_label(source_spec)}' ({maze_name}, {_obs_tag(observable)})",
            benchmark_label,
        ),
        fontsize=16,
        fontweight="bold",
    )

    if filepath is None:
        filename = (
            f"patch_timing_{_policy_artifact_label(source_spec)}_"
            f"{maze_name}_{_obs_tag(observable)}{_figure_suffix(maze_name, horizon)}"
        )
        if filename_suffix is not None:
            filename = f"{filename}_{filename_suffix}"
        filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _draw_cumulative_reward_single(ax: plt.Axes, trajectory: Trajectory) -> None:
    """Draw cumulative reward over transitions with an EMA trend onto ax."""
    rewards = np.array([t.reward for t in trajectory.transitions], dtype=float)
    cumsum = np.cumsum(rewards)
    x = np.arange(len(rewards))
    ax.plot(x, cumsum, linewidth=1.5, color="#2ecc71", label="Cumulative reward")
    window = max(1, len(rewards) // 10)
    alpha = 2.0 / (window + 1)
    trend = _ema(rewards, alpha)
    ax2 = ax.twinx()
    ax2.plot(x, trend, linewidth=2, color="#e67e22", alpha=0.8, label=f"EMA reward (w={window})")
    ax2.set_ylabel("EMA Reward", fontsize=11, color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title(f"Learning Curve (N={len(rewards)} transitions)", fontsize=14)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)


def _draw_residency_lines(
    ax: plt.Axes,
    trajectory: Trajectory,
    maze: Maze,
    window: int | None = None,
) -> None:
    """Draw rolling-window residency fraction lines onto ax (one line per state/obs group).

    For each transition t, plots the fraction of the preceding `window` transitions
    spent in each state. Shows how time allocation shifts over training.
    """
    if not maze.observable:
        n_bins = maze.num_observations
        y_labels = list(maze.maze_spec.observation_labels)
    else:
        n_bins = maze.num_states
        y_labels = maze.state_labels or [f"State {s}" for s in range(n_bins)]
    states = np.array([t.state for t in trajectory.transitions])

    n = len(states)
    w = window if window is not None else max(1, n // 10)

    # One-hot encode, then compute rolling fraction via cumsum.
    # Using cumsum avoids the zero-padding distortion of np.convolve(mode="same"),
    # which artificially ramps fractions up at the start and down at the end.
    one_hot = np.zeros((n, n_bins), dtype=float)
    one_hot[np.arange(n), states] = 1.0

    padded = np.zeros((n + 1, n_bins), dtype=float)
    padded[1:] = np.cumsum(one_hot, axis=0)

    lo = np.maximum(0, np.arange(n) - w + 1)  # trailing window start (clamped)
    hi = np.arange(1, n + 1)                   # trailing window end (exclusive)
    window_sizes = (hi - lo)[:, None]           # true number of real observations

    rolling = (padded[hi] - padded[lo]) / window_sizes

    x = np.arange(n)
    colors = ["#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12"]
    for s in range(n_bins):
        label = y_labels[s] if s < len(y_labels) else f"State {s}"
        ax.plot(x, rolling[:, s], linewidth=1.5, color=colors[s % len(colors)], label=label)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel(f"Fraction of transitions (w={w})", fontsize=12)
    ax.set_title(f"Rolling Residency (N={n} transitions)", fontsize=14)
    ax.legend(fontsize=10)


def _draw_residency_scatter(ax: plt.Axes, trajectory: Trajectory, maze: Maze) -> None:
    """Draw raw residency scatter (state at each transition step) onto ax."""
    if not maze.observable:
        obs_map = maze._state_to_observation_group
        states = [obs_map[t.state] for t in trajectory.transitions]
        n_bins = maze.num_observations
        y_labels = list(maze.maze_spec.observation_labels)
    else:
        states = [t.state for t in trajectory.transitions]
        n_bins = maze.num_states
        y_labels = maze.state_labels or [f"State {s}" for s in range(n_bins)]

    x = np.arange(len(states))
    rewards = np.array([t.reward for t in trajectory.transitions], dtype=float)
    # Size points by absolute reward magnitude to highlight high-reward visits
    sizes = 4 + np.abs(rewards) * 20
    ax.scatter(x, states, s=sizes, alpha=0.4, color="#3498db")
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title(f"Raw Residency (N={len(states)} transitions)", fontsize=14)


def plot_single_run_stats(
    source: PolicyInput,
    maze_name: str = "simple",
    observable: bool = True,
    run_id: int | None = None,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    horizon: int | None = None,
    filepath: Path | None = None,
) -> plt.Figure | None:
    """Plot cumulative reward and raw residency scatter for one saved training run.

    Contrasts with plot_mean_trajectory_stats which averages across many runs.
    """
    source_spec = _normalize_policy(source)
    available_ids = _load_policy_run_ids(source_spec, maze_name, observable, horizon=horizon)
    if not available_ids:
        print(f"No run datasets found for '{_policy_label(source_spec)}' on {maze_name}.")
        return None

    selected_id = run_id if run_id is not None else available_ids[0]
    run_dataset = _load_policy_run_dataset(
        source_spec, selected_id, maze_name, observable, horizon=horizon
    )
    trajectory = _flatten_run_dataset(run_dataset)

    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    maze = maze_from_builtin_maze_spec(maze_name, observable, horizon=resolved_horizon)

    fig, (ax_reward, ax_residency) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    _draw_cumulative_reward_single(ax_reward, trajectory)
    _draw_residency_lines(ax_residency, trajectory, maze)

    fig.suptitle(
        _figure_title(
            f"Single-Run Overview: '{_policy_label(source_spec)}' "
            f"(run {selected_id}, {maze_name}, {_obs_tag(observable)})",
            benchmark_label,
        ),
        fontsize=16,
        fontweight="bold",
    )

    if filepath is None:
        filename = (
            f"single_run_{_policy_artifact_label(source_spec)}_"
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
    filepath: Path | None = None,
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

    if filepath is None:
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


def _curve_matrix(curves: list[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(curve) for curve in curves)
    matrix = np.full((len(curves), max_len), np.nan, dtype=float)
    for row_index, curve in enumerate(curves):
        arr = np.asarray(curve, dtype=float)
        matrix[row_index, : arr.shape[0]] = arr
    return matrix


def _finite_support_counts(curves: list[np.ndarray]) -> np.ndarray:
    matrix = _curve_matrix(curves)
    if matrix.size == 0:
        return np.array([], dtype=int)
    return np.sum(np.isfinite(matrix), axis=0, dtype=int)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return arr.copy()

    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(np.where(finite_mask, arr, 0.0), kernel, mode="same")
    denominator = np.convolve(finite_mask.astype(float), kernel, mode="same")
    smoothed = arr.copy()
    valid = denominator > 0
    smoothed[valid] = numerator[valid] / denominator[valid]
    return smoothed


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
    x_label: str = "Post-Perturbation Episode",
    filepath: Path | None = None,
):
    """Plot mean absolute recovery curves with run-level variability."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    plotted_any = False
    for label, curves in _policy_series_items(curves_by_policy):
        if not curves:
            continue
        summary = aggregate_curves(curves)
        if summary.x.size == 0 or not np.any(np.isfinite(summary.mean)):
            continue
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

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Mean Absolute Dwell Deviation", fontsize=12)
    ax.set_title(
        _figure_title(
            f"Recovery Curves ({_recovery_title_context(maze_name, observable, condition_label)})",
            benchmark_label,
        ),
        fontsize=14,
    )

    if filepath is None:
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
    x_label: str = "Post-Perturbation Episode",
    filepath: Path | None = None,
):
    """Plot mean signed recovery curves to distinguish under- and over-stay."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    plotted_any = False
    for label, curves in _policy_series_items(curves_by_policy):
        if not curves:
            continue
        summary = aggregate_curves(curves)
        if summary.x.size == 0 or not np.any(np.isfinite(summary.mean)):
            continue
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
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Mean Signed Dwell Deviation", fontsize=12)
    ax.set_title(
        _figure_title(
            f"Signed Recovery Curves ({_recovery_title_context(maze_name, observable, condition_label)})",
            benchmark_label,
        ),
        fontsize=14,
    )

    if filepath is None:
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
    """Plot mean recovery AUC by policy."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    labels: list[str] = []
    means: list[float] = []
    for label, values in _policy_series_items(aucs_by_policy):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        labels.append(label)
        means.append(float(np.mean(arr)))

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
        ax.bar(positions, means, alpha=0.7, color="#2c7fb8")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")

    ax.set_ylabel("Recovery AUC", fontsize=12)
    title = _figure_title("Recovery AUC", benchmark_label)
    context = _recovery_title_context(maze_name, observable, condition_label)
    ax.set_title(f"{title}\n{context}", fontsize=13)

    filename = f"recovery_auc_comparison_{maze_name}_{_obs_tag(observable)}_{perturbation_label}"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def plot_boundary_window_recovery_comparison(
    curves_by_policy: dict[PolicyInput, list[np.ndarray]],
    *,
    boundary_window: int | None = None,
    boundary_window_before: int | None = None,
    boundary_window_after: int | None = None,
    maze_name: str = "simple",
    observable: bool = True,
    perturbation_label: str = "decay_swap",
    condition_label: str | None = None,
    save: bool = False,
    show: bool = True,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    smoothing_window: int = 7,
    filepath: Path | None = None,
    metric: str = "cumulative",
):
    """Plot boundary-window dwell deviation before vs after the perturbation boundary."""
    del smoothing_window
    if metric not in {"cumulative", "average"}:
        raise ValueError(
            f"metric must be 'cumulative' or 'average', got {metric!r}"
        )
    resolved_before, resolved_after = _resolve_boundary_plot_windows(
        boundary_window=boundary_window,
        boundary_window_before=boundary_window_before,
        boundary_window_after=boundary_window_after,
    )
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    labels: list[str] = []
    before_totals: list[float] = []
    after_totals: list[float] = []
    for label, curves in _policy_series_items(curves_by_policy):
        if not curves:
            continue
        phase_values = np.array(
            [
                _boundary_window_phase_values(
                    curve,
                    boundary_window_before=resolved_before,
                    boundary_window_after=resolved_after,
                    metric=metric,
                )
                for curve in curves
            ],
            dtype=float,
        )
        before_values = phase_values[:, 0]
        before_values = before_values[np.isfinite(before_values)]
        after_values = phase_values[:, 1]
        after_values = after_values[np.isfinite(after_values)]
        if before_values.size == 0 and after_values.size == 0:
            continue
        labels.append(label)
        before_totals.append(
            float(np.mean(before_values)) if before_values.size > 0 else float("nan")
        )
        after_totals.append(
            float(np.mean(after_values)) if after_values.size > 0 else float("nan")
        )

    if not labels:
        ax.text(
            0.5,
            0.5,
            f"No {metric} boundary-window dwell deviations available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        positions = np.arange(len(labels), dtype=float)
        width = 0.35
        before_totals_arr = np.asarray(before_totals, dtype=float)
        after_totals_arr = np.asarray(after_totals, dtype=float)

        ax.bar(
            positions - width / 2,
            before_totals_arr,
            width,
            color="#4c78a8",
            alpha=0.85,
            label=f"{metric.title()} {resolved_before} Steps Before",
        )
        ax.bar(
            positions + width / 2,
            after_totals_arr,
            width,
            color="#f28e2b",
            alpha=0.85,
            label=f"{metric.title()} {resolved_after} Steps After",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend(fontsize=10)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.8)
    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel(f"{metric.title()} Dwell Deviation", fontsize=12)
    title = _figure_title(
        f"{metric.title()} Dwell Deviation Before vs After Perturbation",
        benchmark_label,
    )
    context = _recovery_title_context(maze_name, observable, condition_label)
    ax.set_title(f"{title}\n{context}", fontsize=13)

    if filepath is None:
        filename = f"boundary_window_recovery_comparison_{maze_name}_{_obs_tag(observable)}_{perturbation_label}"
        if filename_suffix is not None:
            filename = f"{filename}_{filename_suffix}"
        filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def _boundary_window_phase_values(
    curve: np.ndarray,
    *,
    boundary_window_before: int,
    boundary_window_after: int,
    metric: str,
) -> tuple[float, float]:
    """Return finite deviation aggregates before and after the perturbation step."""
    arr = np.asarray(curve, dtype=float)
    expected_length = boundary_window_before + boundary_window_after + 1
    if arr.shape[0] != expected_length:
        raise ValueError(
            "Boundary-window curves must all have length "
            "boundary_window_before + boundary_window_after + 1."
        )

    before_values = arr[:boundary_window_before]
    after_values = arr[boundary_window_before + 1 :]
    if metric == "cumulative":
        return _finite_sum(before_values), _finite_sum(after_values)
    return _finite_mean(before_values), _finite_mean(after_values)


def _finite_sum(values: np.ndarray) -> float:
    """Return the sum of finite values or NaN when no finite values exist."""
    arr = np.asarray(values, dtype=float)
    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.sum(finite_values))


def _finite_mean(values: np.ndarray) -> float:
    """Return the mean of finite values or NaN when no finite values exist."""
    arr = np.asarray(values, dtype=float)
    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def _resolve_boundary_plot_windows(
    *,
    boundary_window: int | None = None,
    boundary_window_before: int | None = None,
    boundary_window_after: int | None = None,
) -> tuple[int, int]:
    if boundary_window_before is None and boundary_window_after is None:
        if boundary_window is None:
            raise ValueError(
                "Specify boundary_window or both boundary_window_before/boundary_window_after."
            )
        if boundary_window <= 0:
            raise ValueError(
                f"boundary_window must be > 0, got {boundary_window}"
            )
        return boundary_window, boundary_window

    if boundary_window is not None:
        raise ValueError(
            "Specify either boundary_window or boundary_window_before/boundary_window_after, not both."
        )
    if boundary_window_before is None or boundary_window_after is None:
        raise ValueError(
            "boundary_window_before and boundary_window_after must be provided together."
        )
    if boundary_window_before <= 0 or boundary_window_after <= 0:
        raise ValueError(
            "boundary_window_before and boundary_window_after must both be > 0, "
            f"got {boundary_window_before} and {boundary_window_after}."
        )
    return boundary_window_before, boundary_window_after


def plot_visit_index_recovery_comparison(
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
    """Plot post-perturbation recovery against complete visit index."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    plotted_any = False
    for label, curves in _policy_series_items(curves_by_policy):
        if not curves:
            continue
        summary = aggregate_curves(curves)
        if summary.x.size == 0 or not np.any(np.isfinite(summary.mean)):
            continue
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
            "No visit-index recovery curves available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.legend(fontsize=10)

    ax.set_xlabel("Complete Visit Index After Perturbation", fontsize=12)
    ax.set_ylabel("Mean Absolute Dwell Deviation", fontsize=12)
    title = _figure_title("Visit-Index Recovery", benchmark_label)
    context = _recovery_title_context(maze_name, observable, condition_label)
    ax.set_title(f"{title}\n{context}", fontsize=13)

    filename = f"visit_index_recovery_comparison_{maze_name}_{_obs_tag(observable)}_{perturbation_label}"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filepath = FIGURES_DIR / f"{filename}.png"
    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig
def plot_recovery_heatmap(
    auc_matrix: dict[PolicyInput, dict[str, float]],
    perturbation_labels: dict[str, str],
    agent_order: list[PolicyInput],
    *,
    observable: bool = True,
    agent_labels: dict[PolicyInput, str] | None = None,
    column_group_boundaries: Sequence[int] = (),
    save: bool = False,
    show: bool = True,
    filepath: Path | None = None,
) -> plt.Figure:
    """Heatmap: perturbations (rows) × agents (columns).

    Cells show mean recovery AUC; high AUC (slow recovery) = red, low = green.
    NaN cells are shown in grey with 'N/A'.
    """
    col_keys = list(perturbation_labels.keys())
    n_rows = len(col_keys)
    n_cols = len(agent_order)

    data = np.full((n_rows, n_cols), np.nan)
    for r, maze in enumerate(col_keys):
        for c, agent in enumerate(agent_order):
            val = auc_matrix.get(agent, {}).get(maze, np.nan)
            data[r, c] = val

    fig, ax = plt.subplots(figsize=(max(9, n_cols * 1.3), max(4, n_rows * 1.2 + 1.5)))

    # Build masked array so NaN cells can be coloured separately
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#cccccc")

    vmin = np.nanmin(data) if not np.all(np.isnan(data)) else 0
    vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 1
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Cell annotations
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            if np.isnan(val):
                ax.text(c, r, "N/A", ha="center", va="center", fontsize=9, color="#666666")
            else:
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="black", fontweight="bold")

    ax.set_xticks(range(n_cols))
    agent_labels = agent_labels or {}
    ax.set_xticklabels(
        [agent_labels.get(a, _policy_label(a)) for a in agent_order],
        fontsize=11,
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([perturbation_labels[k] for k in col_keys], fontsize=11)
    for boundary in column_group_boundaries:
        if 0 < boundary < n_cols:
            ax.axvline(boundary - 0.5, color="white", linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Recovery AUC (lower = faster)", fontsize=10)

    obs_tag = "FO" if observable else "PO"
    ax.set_title(f"Recovery AUC Heatmap ({obs_tag})", fontsize=13, fontweight="bold")

    fig.tight_layout()

    if filepath is None:
        ensure_output_directories()
        filepath = FIGURES_DIR / f"recovery_heatmap_{obs_tag.lower()}.png"

    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


def plot_recovery_heatmap_delta(
    auc_fo: dict[PolicyInput, dict[str, float]],
    auc_po: dict[PolicyInput, dict[str, float]],
    perturbation_labels: dict[str, str],
    agent_order: list[PolicyInput],
    *,
    agent_labels: dict[PolicyInput, str] | None = None,
    column_group_boundaries: Sequence[int] = (),
    save: bool = False,
    show: bool = True,
    filepath: Path | None = None,
) -> plt.Figure:
    """Heatmap of AUC(PO) − AUC(FO): positive = PO hurts recovery more.

    Uses a coolwarm colormap centred at 0.
    """
    col_keys = list(perturbation_labels.keys())
    n_rows = len(col_keys)
    n_cols = len(agent_order)

    delta = np.full((n_rows, n_cols), np.nan)
    for r, maze in enumerate(col_keys):
        for c, agent in enumerate(agent_order):
            fo_val = auc_fo.get(agent, {}).get(maze, np.nan)
            po_val = auc_po.get(agent, {}).get(maze, np.nan)
            if not (np.isnan(fo_val) or np.isnan(po_val)):
                delta[r, c] = po_val - fo_val

    fig, ax = plt.subplots(figsize=(max(9, n_cols * 1.3), max(4, n_rows * 1.2 + 1.5)))

    masked = np.ma.masked_invalid(delta)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#cccccc")

    abs_max = np.nanmax(np.abs(delta)) if not np.all(np.isnan(delta)) else 1.0
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-abs_max, vmax=abs_max)

    for r in range(n_rows):
        for c in range(n_cols):
            val = delta[r, c]
            if np.isnan(val):
                ax.text(c, r, "N/A", ha="center", va="center", fontsize=9, color="#666666")
            else:
                sign = "+" if val >= 0 else ""
                ax.text(c, r, f"{sign}{val:.2f}", ha="center", va="center", fontsize=9,
                        color="black", fontweight="bold")

    ax.set_xticks(range(n_cols))
    agent_labels = agent_labels or {}
    ax.set_xticklabels(
        [agent_labels.get(a, _policy_label(a)) for a in agent_order],
        fontsize=11,
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([perturbation_labels[k] for k in col_keys], fontsize=11)
    for boundary in column_group_boundaries:
        if 0 < boundary < n_cols:
            ax.axvline(boundary - 0.5, color="white", linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("AUC(PO) − AUC(FO)  (positive = PO hurts more)", fontsize=10)

    ax.set_title("Observability Impact: AUC(PO) − AUC(FO)", fontsize=13, fontweight="bold")

    fig.tight_layout()

    if filepath is None:
        ensure_output_directories()
        filepath = FIGURES_DIR / "recovery_heatmap_fo_po_delta.png"

    _finalize_figure(fig, save=save, show=show, filepath=filepath)
    return fig


if __name__ == '__main__':
    plot_aggregate_comparison(
        Agent.QLearning,
        [Agent.MBRL, Agent.SRDyna],
        maze_name="full_one_way_perturbed_detour",
        num_datasets=100,
        observable=True,
        save=True,
        show=True
    )

    plot_aggregate_comparison(
        Agent.MBRL,
        [Agent.QLearning, Agent.SRDyna],
        maze_name="full_one_way_perturbed_detour",
        num_datasets=100,
        observable=True,
        save=True,
        show=True
    )

    plot_aggregate_comparison(
        Agent.SRDyna,
        [Agent.MBRL, Agent.QLearning],
        maze_name="full_one_way_perturbed_detour",
        num_datasets=100,
        observable=True,
        save=True,
        show=True
    )
