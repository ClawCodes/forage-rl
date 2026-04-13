"""Visualization functions for model comparison and Q-value analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl import RunDataset, Trajectory
from forage_rl.analysis.patch_timing import (
    CurveSummary,
    aggregate_curves,
    extract_decision_rows,
    infer_hidden_states_for_trajectory,
    leave_probability_curve,
    mvt_optimal_dwell_by_state,
    mvt_residency_deviation_by_patch,
    normalized_curve_auc,
    observation_group_patch_labels,
)
from forage_rl.agents import QLearning
from forage_rl.agents.base import BaseAgent
from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    PolicySpec,
    is_neural_agent,
    registered_agents,
)
from forage_rl.config import FIGURES_DIR, ensure_directories
from forage_rl.environments import load_builtin_maze_spec, resolve_effective_horizon
from forage_rl.environments.maze import Maze, MazeMDP, MazePOMDP
from forage_rl.environments.maze import maze_from_builtin_maze_spec
from forage_rl.utils import (
    list_run_dataset_run_ids,
    load_logprobs,
    load_run_dataset,
    load_run_dataset_metadata,
)
from forage_rl.visualization.aggregate_plots import (
    plot_aggregate_comparison_impl,
    plot_aggregate_trajectory_stats_impl,
    plot_episode_return_comparison_impl,
    plot_mean_cumulative_reward_impl,
    plot_mean_trajectory_stats_impl,
    plot_modal_residency_impl,
)
from forage_rl.visualization.aggregate_metadata import (
    SOURCE_LEAD_NOTE as _SOURCE_LEAD_NOTE,
    add_figure_notes as _add_figure_notes,
    aggregate_comparison_filename,
    aggregate_trajectory_axis_label as _aggregate_trajectory_axis_label,
    bar_style as _bar_style,
    base_agent_display_label as _base_agent_display_label,
    benchmark_title as _benchmark_title,
    compact_source_policy_line as _compact_source_policy_line,
    comparison_specs as _comparison_specs,
    context_display_label as _context_display_label,
    count_label as _count_label,
    cumulative_residency_coarse_sample_note as _cumulative_residency_coarse_sample_note,
    cumulative_residency_interpretation_note as _cumulative_residency_interpretation_note,
    display_label as _display_label,
    figure_horizon_suffix as _figure_horizon_suffix,
    filename_label as _filename_label,
    include_context_labels as _include_context_labels,
    line_style as _line_style,
    low_sample_note as _low_sample_note,
    normalize_evaluator as _normalize_evaluator,
    normalize_policy as _normalize_policy,
    policy_artifact_label as _policy_artifact_label,
    policy_display_label as _policy_display_label,
    policy_line_style as _policy_line_style,
    policy_uses_explicit_context as _policy_uses_explicit_context,
    running_win_rate as _running_win_rate,
    self_evaluator as _self_evaluator,
    setting_note as _setting_note,
    source_policy_description as _source_policy_description,
    trajectory_sample_metadata_note as _trajectory_sample_metadata_note,
    trajectory_source_policy_description as _trajectory_source_policy_description,
)
from forage_rl.visualization.simple_plots import (
    draw_cumulative_reward_impl,
    draw_residency_location_impl,
    plot_cumulative_reward_impl,
    plot_model_accuracies_from_trajectory_type_impl,
    plot_q_history_impl,
    plot_q_values_impl,
    plot_q_values_with_time_impl,
    plot_residency_location_impl,
    plot_returns_impl,
    plot_trajectory_stats_impl,
)

PolicyInput = Agent | PolicySpec
EvaluatorInput = Agent | EvaluatorSpec


@dataclass(frozen=True)
class _AggregateTrajectoryCohort:
    run_ids: tuple[int, ...]
    horizon: int
    episodes_per_run: int
    transitions_per_run: int
    excluded_cohorts: tuple[tuple[int, int, int, int], ...] = ()
    auto_selected: bool = False

    @property
    def plotted_episodes(self) -> int:
        return len(self.run_ids) * self.episodes_per_run


@dataclass(frozen=True)
class _PatchTimingSummary:
    upper_leave_prob_by_time: CurveSummary
    lower_leave_prob_by_time: CurveSummary
    upper_mvt_deviation_mean: float
    upper_mvt_deviation_std: float
    lower_mvt_deviation_mean: float
    lower_mvt_deviation_std: float
    upper_leave_prob_auc_mean: float
    upper_leave_prob_auc_std: float
    lower_leave_prob_auc_mean: float
    lower_leave_prob_auc_std: float
    sample_count: int
    uses_hidden_state_inference: bool


def _finite_mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _build_patch_timing_summary(
    trajectories: list[Trajectory] | None,
    *,
    maze: MazeMDP,
) -> _PatchTimingSummary | None:
    if not trajectories:
        return None
    if any(
        not all(hasattr(transition, "time_spent") for transition in trajectory.transitions)
        for trajectory in trajectories
    ):
        return None

    maze_name = maze.maze_spec.maze.name
    try:
        patch_labels = observation_group_patch_labels(maze_name)
        optimal_dwell_by_state = mvt_optimal_dwell_by_state(
            maze_name=maze_name,
            horizon=maze.horizon,
        )
    except ValueError:
        return None

    try:
        leave_action = maze.action_labels.index("leave")
    except ValueError:
        return None

    inference_maze = Maze(load_builtin_maze_spec(maze_name), seed=0, horizon=maze.horizon)
    upper_deviations: list[float] = []
    lower_deviations: list[float] = []
    upper_aucs: list[float] = []
    lower_aucs: list[float] = []
    upper_curves: list[np.ndarray] = []
    lower_curves: list[np.ndarray] = []

    for trajectory in trajectories:
        rows = extract_decision_rows(
            trajectory,
            patch_labels=patch_labels,
            resolved_states=infer_hidden_states_for_trajectory(
                trajectory,
                maze=inference_maze,
            ),
        )
        if not rows:
            continue

        deviations = mvt_residency_deviation_by_patch(
            rows,
            leave_action=leave_action,
            optimal_dwell_by_state=optimal_dwell_by_state,
        )
        upper_curve = leave_probability_curve(
            rows,
            value_getter=lambda row: row.time_spent,
            leave_action=leave_action,
            max_value=maze.horizon,
            patch_label="Upper Patch",
        )
        lower_curve = leave_probability_curve(
            rows,
            value_getter=lambda row: row.time_spent,
            leave_action=leave_action,
            max_value=maze.horizon,
            patch_label="Lower Patch",
        )

        upper_deviations.append(
            float(np.mean(deviations["Upper Patch"]))
            if deviations["Upper Patch"]
            else float("nan")
        )
        lower_deviations.append(
            float(np.mean(deviations["Lower Patch"]))
            if deviations["Lower Patch"]
            else float("nan")
        )
        upper_aucs.append(normalized_curve_auc(upper_curve))
        lower_aucs.append(normalized_curve_auc(lower_curve))
        upper_curves.append(upper_curve)
        lower_curves.append(lower_curve)

    if not upper_curves or not lower_curves:
        return None

    upper_deviation_mean, upper_deviation_std = _finite_mean_std(upper_deviations)
    lower_deviation_mean, lower_deviation_std = _finite_mean_std(lower_deviations)
    upper_auc_mean, upper_auc_std = _finite_mean_std(upper_aucs)
    lower_auc_mean, lower_auc_std = _finite_mean_std(lower_aucs)

    return _PatchTimingSummary(
        upper_leave_prob_by_time=aggregate_curves(upper_curves),
        lower_leave_prob_by_time=aggregate_curves(lower_curves),
        upper_mvt_deviation_mean=upper_deviation_mean,
        upper_mvt_deviation_std=upper_deviation_std,
        lower_mvt_deviation_mean=lower_deviation_mean,
        lower_mvt_deviation_std=lower_deviation_std,
        upper_leave_prob_auc_mean=upper_auc_mean,
        upper_leave_prob_auc_std=upper_auc_std,
        lower_leave_prob_auc_mean=lower_auc_mean,
        lower_leave_prob_auc_std=lower_auc_std,
        sample_count=len(upper_curves),
        uses_hidden_state_inference=isinstance(maze, MazePOMDP),
    )


def _load_logprobs_for_policy(
    source: PolicyInput,
    evaluator: EvaluatorInput,
    run_id: int,
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> np.ndarray:
    source_spec = _normalize_policy(source)
    evaluator_spec = _normalize_evaluator(evaluator)
    kwargs: dict[str, object] = {}
    if is_neural_agent(source_spec.agent) and source_spec.context_mode != "legacy_context":
        kwargs["source_context_mode"] = source_spec.context_mode
    if is_neural_agent(evaluator_spec.agent) and evaluator_spec.context_mode != "legacy_context":
        kwargs["evaluator_context_mode"] = evaluator_spec.context_mode
    if horizon is not None:
        kwargs["horizon"] = horizon
    if kwargs:
        return load_logprobs(
            source_spec.agent,
            evaluator_spec,
            run_id,
            maze_name,
            observable,
            **kwargs,
        )
    return load_logprobs(
        source_spec.agent,
        evaluator_spec,
        run_id,
        maze_name,
        observable,
    )


def _list_run_ids_for_policy(
    source: PolicyInput,
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> list[int]:
    source_spec = _normalize_policy(source)
    if is_neural_agent(source_spec.agent) and source_spec.context_mode != "legacy_context":
        return list_run_dataset_run_ids(
            source_spec.agent,
            maze_name,
            observable,
            context_mode=source_spec.context_mode,
            horizon=horizon,
        )
    return list_run_dataset_run_ids(
        source_spec.agent,
        maze_name,
        observable,
        horizon=horizon,
    )


def _load_run_dataset_for_policy(
    source: PolicyInput,
    run_id: int,
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> RunDataset:
    source_spec = _normalize_policy(source)
    if is_neural_agent(source_spec.agent) and source_spec.context_mode != "legacy_context":
        return load_run_dataset(
            source_spec.agent,
            run_id,
            maze_name,
            observable,
            context_mode=source_spec.context_mode,
            horizon=horizon,
        )
    return load_run_dataset(
        source_spec.agent,
        run_id,
        maze_name,
        observable,
        horizon=horizon,
    )


def _selected_run_ids(
    source: PolicyInput,
    maze_name: str,
    observable: bool,
    num_datasets: Optional[int],
    horizon: int | None = None,
) -> list[int]:
    run_ids = _list_run_ids_for_policy(source, maze_name, observable, horizon=horizon)
    if num_datasets is None:
        return run_ids
    return run_ids[:num_datasets]


def _draw_running_win_rate(
    ax: plt.Axes,
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    horizon: int | None = None,
) -> int:
    """Draw source running lead-rate curves against each evaluator onto ax."""
    comparisons = _comparison_specs(source, compare_to)
    if not comparisons:
        ax.text(
            0.5,
            0.5,
            "No comparisons",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return 0

    run_ids = _selected_run_ids(
        source,
        maze_name,
        observable,
        num_datasets,
        horizon=horizon,
    )
    if not run_ids:
        source_label = _policy_artifact_label(source)
        ax.text(
            0.5,
            0.5,
            f"No data for '{source_label}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return 0

    source_eval = _self_evaluator(source)
    include_context = _include_context_labels(
        policies=[source],
        evaluators=compare_to,
    )
    for evaluator in comparisons:
        curves = []
        for run_id in run_ids:
            source_cumsum = _load_logprobs_for_policy(
                source,
                source_eval,
                run_id,
                maze_name,
                observable,
                horizon=horizon,
            )
            eval_cumsum = _load_logprobs_for_policy(
                source,
                evaluator,
                run_id,
                maze_name,
                observable,
                horizon=horizon,
            )
            curves.append(_running_win_rate(source_cumsum, eval_cumsum))

        min_len = min(len(curve) for curve in curves)
        avg_curve = np.mean([curve[:min_len] for curve in curves], axis=0)
        ax.plot(
            avg_curve,
            label=_display_label(evaluator, include_context=include_context),
            **_line_style(evaluator),
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Parity")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Observed Transitions", fontsize=14)
    ax.set_ylabel("Source Lead Rate", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(
        f"Source Running Lead Rate vs Evaluator (runs={len(run_ids)})",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    return len(run_ids)


def plot_cumulative_sum_accuracy(
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    save: bool = False,
    show: bool = True,
    horizon: int | None = None,
):
    """Plot source running lead rate over observed transitions against evaluators."""
    comparisons = _comparison_specs(source, compare_to)
    if not comparisons:
        print("No comparisons to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    run_count = _draw_running_win_rate(
        ax,
        source,
        compare_to,
        maze_name,
        num_datasets,
        observable,
        horizon=horizon,
    )
    ax.set_title(
        "Source Running Lead Rate over Observed Transitions "
        f"({_source_policy_description(source)}, runs={run_count})",
        fontsize=16,
    )
    notes: list[str] = [f"horizon={resolve_effective_horizon(maze_name, horizon)}"]
    note = _setting_note(maze_name, observable)
    if note is not None:
        notes.append(note)
    _add_figure_notes(fig, notes)

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / f"cumulative_accuracy_{_policy_artifact_label(source)}.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def _draw_model_accuracies(
    ax: plt.Axes,
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    horizon: int | None = None,
) -> int:
    """Draw paired final lead-rate bars for source vs each evaluator onto ax."""
    comparisons = _comparison_specs(source, compare_to)
    if not comparisons:
        ax.text(
            0.5,
            0.5,
            "No comparisons",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return 0

    run_ids = _selected_run_ids(
        source,
        maze_name,
        observable,
        num_datasets,
        horizon=horizon,
    )
    if not run_ids:
        source_label = _policy_artifact_label(source)
        ax.text(
            0.5,
            0.5,
            f"No data for '{source_label}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return 0

    source_eval = _self_evaluator(source)
    include_context = _include_context_labels(
        policies=[source],
        evaluators=compare_to,
    )
    source_scores = {spec: 0.0 for spec in comparisons}
    for run_id in run_ids:
        source_final = _load_logprobs_for_policy(
            source,
            source_eval,
            run_id,
            maze_name,
            observable,
            horizon=horizon,
        )[-1]
        for evaluator in comparisons:
            eval_final = _load_logprobs_for_policy(
                source,
                evaluator,
                run_id,
                maze_name,
                observable,
                horizon=horizon,
            )[-1]
            if np.isclose(source_final, eval_final):
                source_scores[evaluator] += 0.5
            elif source_final > eval_final:
                source_scores[evaluator] += 1.0

    x = np.arange(len(comparisons))
    width = 0.36
    source_rates = [source_scores[evaluator] / len(run_ids) for evaluator in comparisons]
    eval_rates = [1.0 - rate for rate in source_rates]

    ax.bar(
        x - width / 2,
        source_rates,
        width,
        label=f"{_policy_display_label(source, include_context=include_context)} (source policy)",
        color="#3498db",
        alpha=0.9,
    )
    for idx, (evaluator, rate) in enumerate(zip(comparisons, eval_rates)):
        ax.bar(
            x[idx] + width / 2,
            rate,
            width,
            label=_display_label(evaluator, include_context=include_context)
            if idx == 0 or evaluator != comparisons[idx - 1]
            else "_nolegend_",
            **_bar_style(evaluator),
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Final Source Lead Rate", fontsize=12)
    ax.set_xlabel("Comparison", fontsize=12)
    ax.set_title(
        f"Final Source Lead Rate on "
        f"'{_policy_display_label(source, include_context=include_context)}' "
        f"Trajectories (runs={len(run_ids)})",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{_policy_display_label(source, include_context=include_context)} vs "
            f"{_display_label(evaluator, include_context=include_context)}"
            for evaluator in comparisons
        ],
        rotation=20,
        ha="right",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Parity (0.50)")
    ax.legend(fontsize=10)

    for idx, (source_rate, evaluator_rate) in enumerate(zip(source_rates, eval_rates)):
        ax.text(
            x[idx] - width / 2,
            min(source_rate + 0.02, 0.97),
            f"{source_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            x[idx] + width / 2,
            min(evaluator_rate + 0.02, 0.97),
            f"{evaluator_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    return len(run_ids)
_aggregate_comparison_filename = aggregate_comparison_filename


def _draw_cumulative_reward(ax: plt.Axes, trajectory: Trajectory) -> None:
    """Draw cumulative reward over transitions onto ax."""
    return draw_cumulative_reward_impl(ax, trajectory)


def _draw_residency_location(
    ax: plt.Axes, trajectory: Trajectory, maze: Maze
) -> None:
    """Draw state residency scatter plot onto ax."""
    return draw_residency_location_impl(ax, trajectory, maze)


def plot_model_accuracies_from_trajectory_type(
    source: PolicyInput,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    save: bool = False,
    show: bool = True,
):
    """Plot paired bar charts comparing source final lead rate against evaluators."""
    return plot_model_accuracies_from_trajectory_type_impl(
        source,
        compare_to,
        maze_name=maze_name,
        num_datasets=num_datasets,
        observable=observable,
        save=save,
        show=show,
        deps={
            "plt": plt,
            "comparison_specs": _comparison_specs,
            "draw_model_accuracies": _draw_model_accuracies,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
            "policy_artifact_label": _policy_artifact_label,
        },
    )


def plot_q_values_with_time(
    q_agent: BaseAgent,
    max_time_to_display: int = 6,
    save: bool = False,
    show: bool = True,
):
    """Plot Q-values as heatmaps for each action."""
    return plot_q_values_with_time_impl(
        q_agent,
        max_time_to_display=max_time_to_display,
        save=save,
        show=show,
        deps={
            "np": np,
            "plt": plt,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
        },
    )


def plot_q_values(q_agent: BaseAgent, show: bool = True):
    """Plot Q-values as a heatmap."""
    return plot_q_values_impl(q_agent, show=show, deps={"plt": plt})


def plot_q_history(q_agent: QLearning, show: bool = True):
    """Plot Q-value history over training episodes."""
    return plot_q_history_impl(q_agent, show=show, deps={"plt": plt})


def plot_returns(q_agent: QLearning, show: bool = True):
    """Plot total reward over training episodes."""
    return plot_returns_impl(q_agent, show=show, deps={"plt": plt})


def plot_cumulative_reward(
    trajectory: Trajectory,
    save: bool = False,
    show: bool = True,
):
    """Plot cumulative reward over transitions for a single trajectory."""
    return plot_cumulative_reward_impl(
        trajectory,
        save=save,
        show=show,
        deps={
            "plt": plt,
            "draw_cumulative_reward": _draw_cumulative_reward,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
        },
    )


def plot_residency_location(
    trajectory: Trajectory,
    maze: Maze,
    save: bool = False,
    show: bool = True,
):
    """Plot state residency as a scatter plot over transitions."""
    return plot_residency_location_impl(
        trajectory,
        maze,
        save=save,
        show=show,
        deps={
            "plt": plt,
            "draw_residency_location": _draw_residency_location,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
        },
    )


def plot_trajectory_stats(
    trajectory: Trajectory,
    maze: Maze,
    source: Agent,
    save: bool = False,
    show: bool = True,
):
    """High-level overview of a single trajectory: cumulative reward and residency."""
    return plot_trajectory_stats_impl(
        trajectory,
        maze,
        source,
        save=save,
        show=show,
        timing_summary=_build_patch_timing_summary([trajectory], maze=maze),
        deps={
            "plt": plt,
            "draw_cumulative_reward": _draw_cumulative_reward,
            "draw_residency_location": _draw_residency_location,
            "draw_patch_leave_probability_by_time": _draw_patch_leave_probability_by_time,
            "draw_patch_timing_summary": _draw_patch_timing_summary,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
        },
    )


def _draw_mean_cumulative_reward(
    ax: plt.Axes,
    reward_sequences: list[np.ndarray],
    *,
    axis_label: str = "Transition Across Run",
    sample_label: str = "runs",
    single_label: str = "Run",
) -> None:
    """Draw cumulative reward summary across aligned reward sequences."""
    cumsums = [np.cumsum(rewards) for rewards in reward_sequences]
    min_len = min(len(cumsum) for cumsum in cumsums)
    arr = np.array([cumsum[:min_len] for cumsum in cumsums])
    mean = arr.mean(axis=0)
    x = np.arange(min_len)

    if len(reward_sequences) == 1:
        ax.plot(x, mean, linewidth=2, color="#2ecc71", label=single_label)
        ax.set_title(f"Cumulative Reward (single {single_label.lower()})", fontsize=14)
    else:
        std = arr.std(axis=0)
        ax.plot(x, mean, linewidth=2, color="#2ecc71", label="Mean")
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.3,
            color="#2ecc71",
            label="±1 SD",
        )
        ax.set_title(
            f"Mean Cumulative Reward ({sample_label}={len(reward_sequences)})",
            fontsize=14,
        )

    ax.set_xlabel(axis_label, fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.legend(fontsize=10)
def _normalize_cohort_policies(
    source: PolicyInput,
    cohort_policies: list[PolicyInput] | None,
) -> list[PolicySpec]:
    """Return a de-duplicated policy list that always includes the source."""
    normalized: list[PolicySpec] = []
    seen: set[PolicySpec] = set()
    for policy in [source, *(cohort_policies or [])]:
        spec = _normalize_policy(policy)
        if spec in seen:
            continue
        normalized.append(spec)
        seen.add(spec)
    return normalized


def _load_run_dataset_metadata_for_policy(
    source: PolicyInput,
    run_id: int,
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> dict[str, object]:
    """Load saved run-dataset metadata for one source policy."""
    source_spec = _normalize_policy(source)
    return load_run_dataset_metadata(
        source_spec.agent,
        run_id,
        maze_name,
        observable,
        context_mode=source_spec.context_mode,
        horizon=horizon,
    )


def _metadata_counts(metadata: dict[str, object]) -> tuple[int, int, int]:
    """Return canonical episode, transition, and horizon counts from metadata."""
    if "horizon" not in metadata:
        raise ValueError("Run dataset metadata is missing required horizon information.")
    return (
        int(metadata["num_episodes"]),
        int(metadata["num_transitions"]),
        int(metadata["horizon"]),
    )


def _matched_run_counts_for_policies(
    policies: list[PolicySpec],
    run_id: int,
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> tuple[int, int, int]:
    """Return one homogeneous count pair for a matched run id across policies."""
    counts_by_policy = {
        policy.artifact_label: _metadata_counts(
            (
                _load_run_dataset_metadata_for_policy(
                    policy,
                    run_id,
                    maze_name,
                    observable,
                )
                if horizon is None
                else _load_run_dataset_metadata_for_policy(
                    policy,
                    run_id,
                    maze_name,
                    observable,
                    horizon=horizon,
                )
            )
        )
        for policy in policies
    }
    unique_counts = set(counts_by_policy.values())
    if len(unique_counts) != 1:
        details = ", ".join(
            f"{label}={episodes}/{transitions}/h{horizon}"
            for label, (episodes, transitions, horizon) in sorted(counts_by_policy.items())
        )
        raise ValueError(
            "Matched run datasets must share one (num_episodes, num_transitions, horizon) "
            f"pair across policies for run_id={run_id}; got {details}."
        )
    return unique_counts.pop()


def _format_run_id_label(run_ids: list[int] | tuple[int, ...]) -> str:
    """Return a compact run-id range label."""
    if not run_ids:
        return "none"

    ordered = sorted(run_ids)
    ranges: list[str] = []
    start = ordered[0]
    end = ordered[0]
    for run_id in ordered[1:]:
        if run_id == end + 1:
            end = run_id
            continue
        ranges.append(f"{start}" if start == end else f"{start}-{end}")
        start = end = run_id
    ranges.append(f"{start}" if start == end else f"{start}-{end}")
    return ",".join(ranges)


def _resolve_aggregate_trajectory_cohort(
    source: PolicyInput,
    maze_name: str,
    observable: bool,
    *,
    cohort_policies: list[PolicyInput] | None = None,
    run_ids: list[int] | None = None,
    horizon: int | None = None,
) -> _AggregateTrajectoryCohort:
    """Return the homogeneous matched cohort used for aggregate trajectory plots."""
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    policies = _normalize_cohort_policies(source, cohort_policies)
    available_by_policy = {
        policy: set(
            _list_run_ids_for_policy(
                policy,
                maze_name,
                observable,
                horizon=horizon,
            )
        )
        for policy in policies
    }

    if run_ids is not None:
        selected_run_ids = sorted(set(run_ids))
        if not selected_run_ids:
            raise ValueError("Explicit run_ids for aggregate trajectory plotting cannot be empty.")
        missing = {
            policy.artifact_label: sorted(set(selected_run_ids) - available)
            for policy, available in available_by_policy.items()
            if not set(selected_run_ids).issubset(available)
        }
        if missing:
            details = ", ".join(
                f"{label} missing {values}" for label, values in sorted(missing.items())
            )
            raise ValueError(
                "Explicit run_ids for aggregate trajectory plotting must exist for every "
                f"matched policy; {details}."
            )

        cohort_keys = {
            _matched_run_counts_for_policies(
                policies,
                run_id,
                maze_name,
                observable,
                horizon=horizon,
            )
            for run_id in selected_run_ids
        }
        if len(cohort_keys) != 1:
            raise ValueError(
                "Explicit run_ids for aggregate trajectory plotting must belong to one "
                f"homogeneous matched cohort; got run_ids={_format_run_id_label(selected_run_ids)}."
            )
        episodes_per_run, transitions_per_run, cohort_horizon = cohort_keys.pop()
        return _AggregateTrajectoryCohort(
            run_ids=tuple(selected_run_ids),
            horizon=cohort_horizon,
            episodes_per_run=episodes_per_run,
            transitions_per_run=transitions_per_run,
        )

    common_run_ids = sorted(set.intersection(*(available for available in available_by_policy.values())))
    if not common_run_ids:
        labels = ", ".join(policy.artifact_label for policy in policies)
        raise ValueError(
            "No matched run_ids are available for aggregate trajectory plotting across "
            f"policies: {labels}."
        )

    grouped_run_ids: dict[tuple[int, int, int], list[int]] = {}
    for run_id in common_run_ids:
        counts = _matched_run_counts_for_policies(
            policies,
            run_id,
            maze_name,
            observable,
            horizon=horizon,
        )
        grouped_run_ids.setdefault(counts, []).append(run_id)

    ranked_cohorts = sorted(
        grouped_run_ids.items(),
        key=lambda item: (-item[0][0], -item[0][1], -len(item[1]), item[1][0]),
    )
    (episodes_per_run, transitions_per_run, cohort_horizon), selected_run_ids = ranked_cohorts[0]
    excluded_cohorts = tuple(
        (len(run_id_group), counts[0], counts[1], counts[2])
        for counts, run_id_group in ranked_cohorts[1:]
    )
    return _AggregateTrajectoryCohort(
        run_ids=tuple(selected_run_ids),
        horizon=cohort_horizon,
        episodes_per_run=episodes_per_run,
        transitions_per_run=transitions_per_run,
        excluded_cohorts=excluded_cohorts,
        auto_selected=len(ranked_cohorts) > 1,
    )


def _trajectory_axis_label(maze: MazeMDP) -> str:
    """Return the x-axis label for run-level trajectory summaries."""
    return "Transition Across Run"


def _episode_boundary_positions(sequence_length: int, horizon: int) -> np.ndarray:
    """Return run-level x positions where episode boundaries occur."""
    if horizon <= 0:
        return np.array([], dtype=float)
    return np.arange(horizon, sequence_length, horizon, dtype=float)


def _draw_episode_boundary_markers(
    ax: plt.Axes,
    sequence_length: int,
    horizon: int,
    y_min: float,
    y_max: float,
) -> None:
    """Draw vertical markers at episode boundaries without adding data-series lines."""
    boundaries = _episode_boundary_positions(sequence_length, horizon)
    if len(boundaries) == 0:
        return
    ax.vlines(
        boundaries,
        y_min,
        y_max,
        colors="#95a5a6",
        linestyles="--",
        linewidth=1.0,
        alpha=0.55,
        zorder=0,
    )


def _run_level_sequences(
    run_datasets: list[RunDataset],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """Flatten each run dataset into one reward/state sequence in episode order."""
    reward_sequences: list[np.ndarray] = []
    state_sequences: list[np.ndarray] = []
    episode_counts: list[int] = []
    transition_counts: list[int] = []

    for run_dataset in run_datasets:
        transitions = list(run_dataset.iter_transitions())
        reward_sequences.append(
            np.array([transition.reward for transition in transitions], dtype=float)
        )
        state_sequences.append(
            np.array([transition.state for transition in transitions], dtype=int)
        )
        episode_counts.append(run_dataset.num_episodes())
        transition_counts.append(run_dataset.num_transitions())

    return reward_sequences, state_sequences, episode_counts, transition_counts


def _episode_level_sequences(
    run_datasets: list[RunDataset],
    run_ids: list[int] | tuple[int, ...],
    horizon: int,
    *,
    expected_episodes_per_run: int,
    expected_transitions_per_run: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pool one aligned reward/state sequence per episode across selected runs."""
    reward_sequences: list[np.ndarray] = []
    state_sequences: list[np.ndarray] = []

    for run_id, run_dataset in zip(run_ids, run_datasets):
        if run_dataset.num_episodes() != expected_episodes_per_run:
            raise ValueError(
                "Aggregate trajectory plotting requires homogeneous run datasets; "
                f"run_id={run_id} has {run_dataset.num_episodes()} episodes, expected "
                f"{expected_episodes_per_run}."
            )
        if run_dataset.num_transitions() != expected_transitions_per_run:
            raise ValueError(
                "Aggregate trajectory plotting requires homogeneous run datasets; "
                f"run_id={run_id} has {run_dataset.num_transitions()} transitions, "
                f"expected {expected_transitions_per_run}."
            )

        for episode_index, trajectory in enumerate(run_dataset):
            if len(trajectory) != horizon:
                raise ValueError(
                    "Aggregate trajectory plotting requires every selected episode "
                    f"to match maze.horizon={horizon}; run_id={run_id}, "
                    f"episode_index={episode_index} has length {len(trajectory)}."
                )
            reward_sequences.append(
                np.array([transition.reward for transition in trajectory], dtype=float)
            )
            state_sequences.append(
                np.array([transition.state for transition in trajectory], dtype=int)
            )

    return reward_sequences, state_sequences


def _patch_state_indices(maze: Maze) -> tuple[np.ndarray, np.ndarray]:
    """Return state indices grouped by upper and lower patch labels."""
    state_labels = maze.state_labels or [
        f"State {state}" for state in range(maze.num_states)
    ]
    upper = np.array(
        [state for state, label in enumerate(state_labels) if label == "Upper Patch"],
        dtype=int,
    )
    lower = np.array(
        [state for state, label in enumerate(state_labels) if label == "Lower Patch"],
        dtype=int,
    )

    if len(upper) == 0 or len(lower) == 0:
        raise ValueError(
            "Patch-fraction residency plot requires both 'Upper Patch' and "
            "'Lower Patch' labels in maze.state_labels."
        )

    recognized = {"Upper Patch", "Lower Patch"}
    unknown = sorted(set(state_labels) - recognized)
    if unknown:
        raise ValueError(
            "Patch-fraction residency plot only supports mazes labeled with "
            f"'Upper Patch'/'Lower Patch', got extra labels: {unknown}"
        )

    return upper, lower


def _observed_patch_group_indices(maze: MazePOMDP) -> tuple[np.ndarray, np.ndarray]:
    """Return observation-group ids grouped by observed upper/lower patch labels."""
    group_to_labels: dict[int, set[str]] = {}
    for state in maze.maze_spec.states:
        group_to_labels.setdefault(state.observation_group, set()).add(state.label)

    upper = np.array(
        [
            group_id
            for group_id, labels in sorted(group_to_labels.items())
            if labels == {"Upper Patch"}
        ],
        dtype=int,
    )
    lower = np.array(
        [
            group_id
            for group_id, labels in sorted(group_to_labels.items())
            if labels == {"Lower Patch"}
        ],
        dtype=int,
    )

    if len(upper) == 0 or len(lower) == 0:
        raise ValueError(
            "Observed patch-occupancy plot requires both 'Upper Patch' and "
            "'Lower Patch' observation groups."
        )

    recognized = {frozenset({"Upper Patch"}), frozenset({"Lower Patch"})}
    unknown = sorted(
        group_id
        for group_id, labels in group_to_labels.items()
        if frozenset(labels) not in recognized
    )
    if unknown:
        raise ValueError(
            "Observed patch-occupancy plot only supports observation groups "
            "mapping cleanly to 'Upper Patch'/'Lower Patch', got group ids: "
            f"{unknown}"
        )

    return upper, lower


def _patch_group_plot_metadata(
    maze: MazeMDP,
) -> tuple[list[np.ndarray], list[str], list[str], str]:
    """Return grouped upper/lower patch metadata for trajectory summary plots."""
    upper_color = "#1f77b4"
    lower_color = "#d95f02"
    if isinstance(maze, MazePOMDP):
        observed_upper, observed_lower = _observed_patch_group_indices(maze)
        return (
            [observed_upper, observed_lower],
            ["Observed Upper Patch", "Observed Lower Patch"],
            [upper_color, lower_color],
            "Observed Patch Occupancy Percentage",
        )

    upper_states, lower_states = _patch_state_indices(maze)
    return (
        [upper_states, lower_states],
        ["Upper Patch", "Lower Patch"],
        [upper_color, lower_color],
        "Patch Occupancy Percentage",
    )


def _occupancy_fractions(
    state_sequences: list[np.ndarray],
    domain_ids: list[int] | np.ndarray,
) -> np.ndarray:
    """Return occupancy fractions over an explicit id domain at each timestep."""
    min_len = min(len(states) for states in state_sequences)
    arr = np.array([states[:min_len] for states in state_sequences])
    return np.array([(arr == state_id).mean(axis=0) for state_id in domain_ids])


def _group_occupancy_fractions(
    state_sequences: list[np.ndarray],
    grouped_domain_ids: list[np.ndarray],
) -> np.ndarray:
    """Return grouped occupancy fractions over sets of ids at each timestep."""
    min_len = min(len(states) for states in state_sequences)
    arr = np.array([states[:min_len] for states in state_sequences])
    return np.array(
        [
            np.isin(arr, np.asarray(group_ids, dtype=int)).mean(axis=0)
            for group_ids in grouped_domain_ids
        ]
    )


def _cumulative_occupancy_shares(
    state_sequences: list[np.ndarray],
    domain_ids: list[int] | np.ndarray,
) -> np.ndarray:
    """Return cumulative occupancy share over an explicit id domain."""
    min_len = min(len(states) for states in state_sequences)
    arr = np.array([states[:min_len] for states in state_sequences])
    denominators = np.arange(1, min_len + 1, dtype=float)

    cumulative_shares: list[np.ndarray] = []
    for state_id in domain_ids:
        membership = (arr == state_id).astype(float)
        cumulative = np.cumsum(membership, axis=1) / denominators[None, :]
        cumulative_shares.append(cumulative.mean(axis=0))

    return np.array(cumulative_shares)


def _cumulative_group_occupancy_shares(
    state_sequences: list[np.ndarray],
    grouped_domain_ids: list[np.ndarray],
) -> np.ndarray:
    """Return cumulative occupancy share over grouped ids."""
    min_len = min(len(states) for states in state_sequences)
    arr = np.array([states[:min_len] for states in state_sequences])
    denominators = np.arange(1, min_len + 1, dtype=float)

    cumulative_shares: list[np.ndarray] = []
    for group_ids in grouped_domain_ids:
        membership = np.isin(arr, np.asarray(group_ids, dtype=int)).astype(float)
        cumulative = np.cumsum(membership, axis=1) / denominators[None, :]
        cumulative_shares.append(cumulative.mean(axis=0))

    return np.array(cumulative_shares)


def _state_occupancy_fractions(
    state_sequences: list[np.ndarray],
    maze: Maze,
) -> np.ndarray:
    """Return per-state occupancy fractions at each timestep."""
    return _occupancy_fractions(state_sequences, np.arange(maze.num_states))


def _residency_line_labels(maze: Maze) -> list[str]:
    """Return human-readable labels for residency lines."""
    state_labels = maze.state_labels or [
        f"State {state}" for state in range(maze.num_states)
    ]
    counts: dict[str, int] = {}
    for label in state_labels:
        counts[label] = counts.get(label, 0) + 1

    labels: list[str] = []
    for state, label in enumerate(state_labels):
        if counts[label] == 1:
            labels.append(label)
        else:
            labels.append(f"{label} (S{state})")
    return labels


def _residency_colors(maze: Maze) -> list[str]:
    """Return line colors grouped by patch where possible."""
    state_labels = maze.state_labels or [
        f"State {state}" for state in range(maze.num_states)
    ]
    upper_palette = ["#1f77b4", "#4fa3d9", "#8ecae6"]
    lower_palette = ["#d95f02", "#f4a261", "#f6bd60"]
    fallback_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    upper_index = 0
    lower_index = 0
    colors: list[str] = []
    for label in state_labels:
        if label == "Upper Patch":
            colors.append(upper_palette[upper_index % len(upper_palette)])
            upper_index += 1
        elif label == "Lower Patch":
            colors.append(lower_palette[lower_index % len(lower_palette)])
            lower_index += 1
        else:
            colors.append(fallback_palette[len(colors) % len(fallback_palette)])
    return colors


def _observed_group_plot_metadata(
    maze: MazePOMDP,
) -> tuple[list[int], list[str], list[str], str]:
    """Return domain ids, labels, colors, and title base for PO occupancy plots."""
    upper_palette = ["#1f77b4", "#4fa3d9", "#8ecae6"]
    lower_palette = ["#d95f02", "#f4a261", "#f6bd60"]
    fallback_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    group_to_labels: dict[int, set[str]] = {}
    for state in maze.maze_spec.states:
        group_to_labels.setdefault(state.observation_group, set()).add(state.label)

    group_ids = sorted(group_to_labels)
    labels: list[str] = []
    colors: list[str] = []
    upper_index = 0
    lower_index = 0
    recognized_groups = True
    for group_id in group_ids:
        group_labels = group_to_labels[group_id]
        if len(group_labels) == 1:
            label = next(iter(group_labels))
            if label == "Upper Patch":
                labels.append("Observed Upper Patch")
                colors.append(upper_palette[upper_index % len(upper_palette)])
                upper_index += 1
                continue
            if label == "Lower Patch":
                labels.append("Observed Lower Patch")
                colors.append(lower_palette[lower_index % len(lower_palette)])
                lower_index += 1
                continue
        recognized_groups = False
        labels.append(f"Observation Group {group_id}")
        colors.append(fallback_palette[len(colors) % len(fallback_palette)])

    title_base = (
        "Observed Patch Occupancy"
        if recognized_groups
        else "Observation Group Occupancy"
    )
    return group_ids, labels, colors, title_base


def _occupancy_plot_metadata(
    maze: MazeMDP,
) -> tuple[list[int], list[str], list[str], str]:
    """Return occupancy domain metadata for FO and PO trajectory summaries."""
    if isinstance(maze, MazePOMDP):
        return _observed_group_plot_metadata(maze)
    title_base = "Patch Occupancy" if maze.num_states == 2 else "State Occupancy"
    return (
        list(range(maze.num_states)),
        _residency_line_labels(maze),
        _residency_colors(maze),
        title_base,
    )


def _cumulative_residency_title_base(title_base: str) -> str:
    """Convert occupancy-style title bases into cumulative-residency titles."""
    if title_base.endswith("Occupancy Percentage"):
        return (
            f"{title_base[: -len('Occupancy Percentage')]}"
            "Cumulative Residency Share"
        )
    if title_base.endswith("Occupancy"):
        return f"{title_base[: -len('Occupancy')]}Cumulative Residency Share"
    return f"{title_base} Cumulative Residency Share"


def _modal_residency_axis_metadata(
    maze: MazeMDP,
) -> tuple[list[int], list[str], str]:
    """Return y-axis ids, labels, and title base for run-level modal residency."""
    domain_ids, labels, _, title_base = _occupancy_plot_metadata(maze)
    if isinstance(maze, MazePOMDP):
        title = (
            "Observed Modal Residency Location"
            if title_base == "Observed Patch Occupancy"
            else "Observation-Group Modal Residency Location"
        )
        return domain_ids, labels, title
    return domain_ids, labels, "Modal Residency Location"


def _draw_residency_fractions(
    ax: plt.Axes,
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
) -> None:
    """Draw occupancy fraction at each timestep across the plotted domain."""
    domain_ids, labels, colors, title_base = _occupancy_plot_metadata(maze)
    occupancy = _occupancy_fractions(state_sequences, domain_ids)
    x = np.arange(occupancy.shape[1])
    for state, series in enumerate(occupancy):
        ax.plot(
            x,
            series,
            linewidth=2.0,
            color=colors[state],
            label=labels[state],
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Transition Within Episode", fontsize=12)
    ax.set_ylabel("Fraction of Episodes", fontsize=12)
    if len(state_sequences) == 1:
        title = f"{title_base} (single episode)"
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"{title_base} Fraction (episodes={len(state_sequences)})", fontsize=14)
    ax.legend(fontsize=9, ncol=2 if len(domain_ids) > 3 else 1)


def _draw_modal_residency(
    ax: plt.Axes,
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
) -> None:
    """Draw the modal location at each run-level transition across runs."""
    domain_ids, labels, title_base = _modal_residency_axis_metadata(maze)
    min_len = min(len(states) for states in state_sequences)
    arr = np.array([states[:min_len] for states in state_sequences])

    modal_states: list[int] = []
    frequencies: list[float] = []
    for step in range(min_len):
        counts = np.array(
            [(arr[:, step] == state_id).sum() for state_id in domain_ids],
            dtype=float,
        )
        modal_index = int(np.argmax(counts))
        modal_states.append(domain_ids[modal_index])
        frequencies.append(float(counts[modal_index] / len(state_sequences)))

    sizes = np.maximum(np.array(frequencies) * 60.0, 8.0)
    ax.scatter(
        np.arange(min_len),
        modal_states,
        s=sizes,
        alpha=0.7,
        color="#3498db",
    )
    ax.set_yticks(domain_ids)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Transition Across Run", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title(
        f"{title_base} ({'single run' if len(state_sequences) == 1 else f'runs={len(state_sequences)}'})",
        fontsize=14,
    )


def _draw_cumulative_residency_shares(
    ax: plt.Axes,
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
    *,
    axis_label: str,
    sample_label: str,
    single_label: str,
    draw_episode_boundaries: bool = False,
) -> None:
    """Draw cumulative residency-share lines across aligned episode sequences."""
    try:
        groups, labels, colors, title_base = _patch_group_plot_metadata(maze)
        occupancy = 100.0 * _cumulative_group_occupancy_shares(
            state_sequences,
            groups,
        )
    except ValueError:
        domain_ids, labels, colors, title_base = _occupancy_plot_metadata(maze)
        occupancy = 100.0 * _cumulative_occupancy_shares(state_sequences, domain_ids)
    title_base = _cumulative_residency_title_base(title_base)

    x = np.arange(occupancy.shape[1])
    for index, series in enumerate(occupancy):
        ax.plot(
            x,
            series,
            linewidth=2.5,
            color=colors[index],
            label=labels[index],
        )
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 25.0))
    ax.set_xlabel(axis_label, fontsize=12)
    ax.set_ylabel("Cumulative Share of Episode (%)", fontsize=12)
    ax.set_title(
        f"{title_base} ({f'single {single_label}' if len(state_sequences) == 1 else f'{sample_label.lower()}={len(state_sequences)}'})",
        fontsize=14,
    )
    ax.legend(fontsize=9)


def _set_curve_axis_limits(ax: plt.Axes, curve_summaries: list[CurveSummary]) -> None:
    max_index = 0
    for curve in curve_summaries:
        finite = np.flatnonzero(np.isfinite(curve.mean))
        if finite.size > 0:
            max_index = max(max_index, int(finite[-1]))
    ax.set_xlim(0, max_index if max_index > 0 else 1)


def _draw_patch_leave_probability_by_time(
    ax: plt.Axes,
    timing_summary: _PatchTimingSummary,
) -> None:
    curves = (
        (
            "Upper Patch",
            timing_summary.upper_leave_prob_by_time,
            "#1f77b4",
        ),
        (
            "Lower Patch",
            timing_summary.lower_leave_prob_by_time,
            "#ff7f0e",
        ),
    )
    for label, curve, color in curves:
        ax.plot(curve.x, curve.mean, label=label, color=color, linewidth=2.0)
        if timing_summary.sample_count > 1:
            ax.fill_between(
                curve.x,
                np.clip(curve.mean - curve.std, 0.0, 1.0),
                np.clip(curve.mean + curve.std, 0.0, 1.0),
                color=color,
                alpha=0.12,
            )
    ax.set_title("Leave Probability by Time Spent in Patch", fontsize=14)
    ax.set_xlabel("Time Spent In Patch", fontsize=12)
    ax.set_ylabel("Leave Probability", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    _set_curve_axis_limits(
        ax,
        [
            timing_summary.upper_leave_prob_by_time,
            timing_summary.lower_leave_prob_by_time,
        ],
    )
    ax.legend(fontsize=9)


def _draw_patch_timing_summary(
    ax: plt.Axes,
    timing_summary: _PatchTimingSummary,
) -> None:
    x = np.arange(2, dtype=float)
    patch_labels = ["Upper Patch", "Lower Patch"]
    deviation_means = [
        timing_summary.upper_mvt_deviation_mean,
        timing_summary.lower_mvt_deviation_mean,
    ]
    deviation_stds = [
        timing_summary.upper_mvt_deviation_std,
        timing_summary.lower_mvt_deviation_std,
    ]
    auc_means = [
        timing_summary.upper_leave_prob_auc_mean,
        timing_summary.lower_leave_prob_auc_mean,
    ]
    auc_stds = [
        timing_summary.upper_leave_prob_auc_std,
        timing_summary.lower_leave_prob_auc_std,
    ]

    colors = ["#1f77b4", "#ff7f0e"]
    ax.bar(
        x,
        deviation_means,
        yerr=deviation_stds,
        color=colors,
        alpha=0.82,
        width=0.56,
        label="Mean MVT Deviation",
    )
    ax.axhline(0.0, color="#2f2f2f", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Signed Dwell Deviation", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(patch_labels)
    ax.set_title("Patch Residency vs MVT (AUC Overlay)", fontsize=14)

    ax_auc = ax.twinx()
    ax_auc.errorbar(
        x,
        auc_means,
        yerr=auc_stds,
        color="#2f2f2f",
        marker="o",
        linestyle="--",
        linewidth=1.8,
        capsize=4,
        label="Leave-Time AUC",
    )
    ax_auc.set_ylabel("Normalized AUC", fontsize=12)
    ax_auc.set_ylim(0.0, 1.0)

    handles, labels = ax.get_legend_handles_labels()
    auc_handles, auc_labels = ax_auc.get_legend_handles_labels()
    ax.legend(handles + auc_handles, labels + auc_labels, fontsize=9, loc="upper left")


def plot_mean_cumulative_reward(
    reward_sequences: list[np.ndarray],
    save: bool = False,
    show: bool = True,
):
    """Plot cumulative reward summary across aligned reward sequences."""
    return plot_mean_cumulative_reward_impl(
        reward_sequences,
        save=save,
        show=show,
        deps={
            "plt": plt,
            "draw_mean_cumulative_reward": _draw_mean_cumulative_reward,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
        },
    )


def plot_modal_residency(
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
    save: bool = False,
    show: bool = True,
):
    """Plot per-state occupancy fraction across aligned sequences."""
    return plot_modal_residency_impl(
        state_sequences,
        maze,
        save=save,
        show=show,
        deps={
            "plt": plt,
            "draw_residency_fractions": _draw_residency_fractions,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
        },
    )


def plot_mean_trajectory_stats(
    reward_sequences: list[np.ndarray],
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
    source: PolicyInput,
    save: bool = False,
    show: bool = True,
    run_count: Optional[int] = None,
    episodes_per_run: str | int | None = None,
    transitions_per_run: str | int | None = None,
    plotted_episodes: Optional[int] = None,
    matched_run_ids: list[int] | tuple[int, ...] | None = None,
    excluded_cohorts: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...] | None = None,
    setting_note: Optional[str] = None,
    filename_suffix: str | None = None,
    benchmark_label: str | None = None,
    benchmark_note: str | None = None,
    timing_trajectories: list[Trajectory] | None = None,
):
    """High-level source-policy overview aggregated across matched episode rollouts."""
    return plot_mean_trajectory_stats_impl(
        reward_sequences,
        state_sequences,
        maze,
        source,
        save=save,
        show=show,
        run_count=run_count,
        episodes_per_run=episodes_per_run,
        transitions_per_run=transitions_per_run,
        plotted_episodes=plotted_episodes,
        matched_run_ids=matched_run_ids,
        excluded_cohorts=excluded_cohorts,
        setting_note=setting_note,
        filename_suffix=filename_suffix,
        benchmark_label=benchmark_label,
        benchmark_note=benchmark_note,
        timing_summary=_build_patch_timing_summary(timing_trajectories, maze=maze),
        deps={
            "plt": plt,
            "aggregate_trajectory_axis_label": _aggregate_trajectory_axis_label,
            "benchmark_title": _benchmark_title,
            "compact_source_policy_line": _compact_source_policy_line,
            "trajectory_source_policy_description": _trajectory_source_policy_description,
            "draw_mean_cumulative_reward": _draw_mean_cumulative_reward,
            "draw_cumulative_residency_shares": _draw_cumulative_residency_shares,
            "draw_patch_leave_probability_by_time": _draw_patch_leave_probability_by_time,
            "draw_patch_timing_summary": _draw_patch_timing_summary,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
            "figure_horizon_suffix": _figure_horizon_suffix,
            "policy_artifact_label": _policy_artifact_label,
            "is_pomdp": lambda candidate_maze: isinstance(candidate_maze, MazePOMDP),
            "setting_note": _setting_note,
            "trajectory_sample_metadata_note": _trajectory_sample_metadata_note,
            "low_sample_note": _low_sample_note,
            "cumulative_residency_coarse_sample_note": _cumulative_residency_coarse_sample_note,
            "cumulative_residency_interpretation_note": _cumulative_residency_interpretation_note,
            "format_run_id_label": _format_run_id_label,
            "add_figure_notes": _add_figure_notes,
        },
    )


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
    """Load a matched cohort of saved runs and plot episode-aligned trajectory stats."""
    return plot_aggregate_trajectory_stats_impl(
        source,
        maze_name=maze_name,
        observable=observable,
        cohort_policies=cohort_policies,
        run_ids=run_ids,
        save=save,
        show=show,
        filename_suffix=filename_suffix,
        benchmark_label=benchmark_label,
        benchmark_note=benchmark_note,
        horizon=horizon,
        deps={
            "resolve_effective_horizon": resolve_effective_horizon,
            "resolve_aggregate_trajectory_cohort": _resolve_aggregate_trajectory_cohort,
            "load_run_dataset_for_policy": _load_run_dataset_for_policy,
            "maze_from_builtin_maze_spec": maze_from_builtin_maze_spec,
            "episode_level_sequences": _episode_level_sequences,
            "plot_mean_trajectory_stats": plot_mean_trajectory_stats,
            "setting_note": _setting_note,
        },
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
    """Plot episode-return learning curves for one setting across agents."""
    return plot_episode_return_comparison_impl(
        maze_name=maze_name,
        observable=observable,
        agents=agents,
        save=save,
        show=show,
        filename_suffix=filename_suffix,
        benchmark_label=benchmark_label,
        benchmark_note=benchmark_note,
        horizon=horizon,
        deps={
            "plt": plt,
            "resolve_effective_horizon": resolve_effective_horizon,
            "registered_agents": registered_agents,
            "include_context_labels": _include_context_labels,
            "list_run_ids_for_policy": _list_run_ids_for_policy,
            "load_run_dataset_metadata_for_policy": _load_run_dataset_metadata_for_policy,
            "load_run_dataset_for_policy": _load_run_dataset_for_policy,
            "policy_display_label": _policy_display_label,
            "policy_line_style": _policy_line_style,
            "count_label": _count_label,
            "benchmark_title": _benchmark_title,
            "setting_note": _setting_note,
            "add_figure_notes": _add_figure_notes,
            "figure_horizon_suffix": _figure_horizon_suffix,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
        },
    )


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
    """Plot a 2-panel source-centric comparison figure for source vs evaluators."""
    return plot_aggregate_comparison_impl(
        source,
        compare_to,
        maze_name=maze_name,
        num_datasets=num_datasets,
        observable=observable,
        save=save,
        show=show,
        filename_suffix=filename_suffix,
        benchmark_label=benchmark_label,
        benchmark_note=benchmark_note,
        horizon=horizon,
        deps={
            "plt": plt,
            "resolve_effective_horizon": resolve_effective_horizon,
            "draw_model_accuracies": _draw_model_accuracies,
            "draw_running_win_rate": _draw_running_win_rate,
            "benchmark_title": _benchmark_title,
            "source_policy_description": _source_policy_description,
            "setting_note": _setting_note,
            "source_lead_note": _SOURCE_LEAD_NOTE,
            "add_figure_notes": _add_figure_notes,
            "ensure_directories": ensure_directories,
            "figures_dir": FIGURES_DIR,
            "aggregate_comparison_filename": _aggregate_comparison_filename,
        },
    )


if __name__ == "__main__":
    mode_aware_evaluators: list[EvaluatorInput] = [
        Agent.MBRL,
        Agent.QLearning,
        EvaluatorSpec(agent=Agent.DQN, mode="fresh"),
        EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        EvaluatorSpec(agent=Agent.ELMAN, mode="fresh"),
        EvaluatorSpec(agent=Agent.ELMAN, mode="pretrained"),
        EvaluatorSpec(agent=Agent.GRU, mode="fresh"),
        EvaluatorSpec(agent=Agent.GRU, mode="pretrained"),
        EvaluatorSpec(agent=Agent.LSTM, mode="fresh"),
        EvaluatorSpec(agent=Agent.LSTM, mode="pretrained"),
    ]
    supported_settings = [("simple", True), ("full", True), ("full", False)]

    for maze_name, observable in supported_settings:
        plot_episode_return_comparison(
            maze_name=maze_name,
            observable=observable,
            save=True,
            show=False,
        )
        for source_agent in registered_agents():
            plot_aggregate_trajectory_stats(
                source_agent,
                maze_name=maze_name,
                observable=observable,
                save=True,
            )
            plot_aggregate_comparison(
                source_agent,
                mode_aware_evaluators,
                maze_name=maze_name,
                observable=observable,
                save=True,
                show=False,
            )
