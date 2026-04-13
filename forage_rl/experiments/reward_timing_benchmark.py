"""Behavior probes and reports for the full/PO reward-timing benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from forage_rl.analysis.patch_timing import (
    CurveSummary,
    DecisionRow,
    aggregate_curves as _aggregate_curve,
    dwell_lengths_by_patch as _dwell_lengths_by_patch,
    extract_decision_rows,
    infer_hidden_states_for_trajectory as _infer_hidden_states_for_trajectory,
    leave_probability_curve as _leave_probability_curve,
    mvt_optimal_dwell_by_state as _mvt_optimal_dwell_by_state,
    mvt_residency_deviation_by_patch as _mvt_residency_deviation_by_patch,
    normalized_curve_auc as _normalized_curve_auc,
    observation_group_patch_labels as _observation_group_patch_labels,
)
import forage_rl.config as config_module
from forage_rl.agents.registry import Agent, NeuralContextMode, PolicySpec
from forage_rl.environments import Maze, load_builtin_maze_spec, resolve_effective_horizon
from forage_rl.visualization.plots import _policy_display_label, _policy_line_style
from forage_rl.utils import list_run_dataset_run_ids, load_run_dataset


BENCHMARK_CONTEXT_MODES: tuple[NeuralContextMode, ...] = (
    "prev_reward",
    "prev_reward_time",
)
BENCHMARK_NOTE = (
    "Suite role: full/PO clean comparison of obs+prev_reward vs obs+prev_reward+time."
)


@dataclass(frozen=True)
class PolicyRunSummary:
    """Per-run reward-timing summary used for aggregation and reporting."""

    mean_return: float
    tail_return: float
    tail_window_episodes: int
    upper_dwell: float
    lower_dwell: float
    best_upper_threshold: int
    best_lower_threshold: int
    patch_threshold_balanced_accuracy: float
    best_zero_streak_threshold: int
    zero_streak_balanced_accuracy: float
    upper_mvt_deviation: float
    lower_mvt_deviation: float
    upper_leave_prob_auc: float
    lower_leave_prob_auc: float
    upper_leave_prob_by_time: np.ndarray
    lower_leave_prob_by_time: np.ndarray
    leave_prob_by_zero_streak: np.ndarray


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Filesystem outputs emitted by the benchmark report pipeline."""

    report_path: Path
    figure_paths: tuple[Path, ...]
    matched_run_ids: tuple[int, ...]


def reward_timing_policies() -> list[PolicySpec]:
    """Return the canonical policy cohort for the reward-timing benchmark."""
    policies: list[PolicySpec] = []
    for context_mode in BENCHMARK_CONTEXT_MODES:
        for agent in (Agent.DQN, Agent.ELMAN, Agent.GRU, Agent.LSTM):
            policies.append(PolicySpec(agent=agent, context_mode=context_mode))
    return policies


def _obs_tag(observable: bool) -> str:
    return "FO" if observable else "PO"


def _report_dir() -> Path:
    reports_dir = config_module.FIGURES_DIR.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _figure_path(
    stem: str,
    *,
    maze_name: str,
    observable: bool,
    filename_suffix: str,
) -> Path:
    return (
        config_module.FIGURES_DIR
        / f"{stem}_{maze_name}_{_obs_tag(observable)}_{filename_suffix}.png"
    )


def _report_path(
    *,
    maze_name: str,
    observable: bool,
    filename_suffix: str,
) -> Path:
    return _report_dir() / (
        f"reward_timing_{maze_name}_{_obs_tag(observable)}_{filename_suffix}.md"
    )


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _format_mean_std(values: Iterable[float], *, digits: int = 2) -> str:
    mean, std = _mean_std(values)
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return class-balanced accuracy, averaging over present classes only."""
    actual_bool = actual.astype(bool)
    predicted_bool = predicted.astype(bool)
    positive_mask = actual_bool
    negative_mask = ~actual_bool
    recalls: list[float] = []
    if positive_mask.any():
        recalls.append(
            float(np.mean(predicted_bool[positive_mask] == actual_bool[positive_mask]))
        )
    if negative_mask.any():
        recalls.append(
            float(np.mean(predicted_bool[negative_mask] == actual_bool[negative_mask]))
        )
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def _tail_window_episodes(num_episodes: int) -> int:
    """Return the dynamic tail window used for late-stage return summaries."""
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be > 0, got {num_episodes}")
    return max(1, int(np.ceil(0.2 * num_episodes)))


def fit_patch_threshold_rule(
    rows: list[DecisionRow],
    *,
    horizon: int,
    leave_action: int,
) -> tuple[tuple[int, int], float]:
    """Fit leave-if-time-spent-thresholds by patch using balanced accuracy."""
    actual_leave = np.array([row.action == leave_action for row in rows], dtype=bool)
    upper_times = np.array([row.time_spent for row in rows], dtype=int)
    is_upper_patch = np.array([row.patch_label == "Upper Patch" for row in rows], dtype=bool)
    best_thresholds = (0, 0)
    best_score = -np.inf

    for upper_threshold in range(horizon):
        upper_leave = is_upper_patch & (upper_times >= upper_threshold)
        for lower_threshold in range(horizon):
            predicted_leave = upper_leave | (
                (~is_upper_patch) & (upper_times >= lower_threshold)
            )
            score = _balanced_accuracy(actual_leave, predicted_leave)
            if score > best_score:
                best_score = score
                best_thresholds = (upper_threshold, lower_threshold)

    return best_thresholds, float(best_score)


def fit_zero_streak_rule(
    rows: list[DecisionRow],
    *,
    horizon: int,
    leave_action: int,
) -> tuple[int, float]:
    """Fit leave-if-zero-streak-threshold rule using balanced accuracy."""
    actual_leave = np.array([row.action == leave_action for row in rows], dtype=bool)
    zero_streaks = np.array([row.zero_streak for row in rows], dtype=int)
    best_threshold = 0
    best_score = -np.inf

    for threshold in range(horizon):
        predicted_leave = zero_streaks >= threshold
        score = _balanced_accuracy(actual_leave, predicted_leave)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, float(best_score)


def summarize_policy_run(
    run_dataset,
    *,
    maze_name: str,
    patch_labels: dict[int, str],
    leave_action: int,
    horizon: int,
    optimal_dwell_by_state: dict[int, int],
) -> PolicyRunSummary:
    """Summarize one saved run dataset for reward-timing analysis."""
    episode_returns = np.array(
        [
            sum(float(transition.reward) for transition in trajectory.transitions)
            for trajectory in run_dataset.trajectories
        ],
        dtype=float,
    )
    tail_window = _tail_window_episodes(len(episode_returns))
    inference_maze = Maze(load_builtin_maze_spec(maze_name), seed=0, horizon=horizon)
    rows = [
        row
        for trajectory in run_dataset.trajectories
        for row in extract_decision_rows(
            trajectory,
            patch_labels=patch_labels,
            resolved_states=_infer_hidden_states_for_trajectory(
                trajectory,
                maze=inference_maze,
            ),
        )
    ]
    dwell_lengths = _dwell_lengths_by_patch(rows, leave_action=leave_action)
    mvt_deviation_by_patch = _mvt_residency_deviation_by_patch(
        rows,
        leave_action=leave_action,
        optimal_dwell_by_state=optimal_dwell_by_state,
    )
    (upper_threshold, lower_threshold), patch_score = fit_patch_threshold_rule(
        rows,
        horizon=horizon,
        leave_action=leave_action,
    )
    zero_threshold, zero_score = fit_zero_streak_rule(
        rows,
        horizon=horizon,
        leave_action=leave_action,
    )
    upper_leave_prob_by_time = _leave_probability_curve(
        rows,
        value_getter=lambda row: row.time_spent,
        leave_action=leave_action,
        max_value=horizon,
        patch_label="Upper Patch",
    )
    lower_leave_prob_by_time = _leave_probability_curve(
        rows,
        value_getter=lambda row: row.time_spent,
        leave_action=leave_action,
        max_value=horizon,
        patch_label="Lower Patch",
    )

    return PolicyRunSummary(
        mean_return=float(np.mean(episode_returns)),
        tail_return=float(np.mean(episode_returns[-tail_window:])),
        tail_window_episodes=tail_window,
        upper_dwell=float(np.mean(dwell_lengths["Upper Patch"]))
        if dwell_lengths["Upper Patch"]
        else float("nan"),
        lower_dwell=float(np.mean(dwell_lengths["Lower Patch"]))
        if dwell_lengths["Lower Patch"]
        else float("nan"),
        best_upper_threshold=upper_threshold,
        best_lower_threshold=lower_threshold,
        patch_threshold_balanced_accuracy=patch_score,
        best_zero_streak_threshold=zero_threshold,
        zero_streak_balanced_accuracy=zero_score,
        upper_mvt_deviation=float(np.mean(mvt_deviation_by_patch["Upper Patch"]))
        if mvt_deviation_by_patch["Upper Patch"]
        else float("nan"),
        lower_mvt_deviation=float(np.mean(mvt_deviation_by_patch["Lower Patch"]))
        if mvt_deviation_by_patch["Lower Patch"]
        else float("nan"),
        upper_leave_prob_auc=_normalized_curve_auc(upper_leave_prob_by_time),
        lower_leave_prob_auc=_normalized_curve_auc(lower_leave_prob_by_time),
        upper_leave_prob_by_time=upper_leave_prob_by_time,
        lower_leave_prob_by_time=lower_leave_prob_by_time,
        leave_prob_by_zero_streak=_leave_probability_curve(
            rows,
            value_getter=lambda row: row.zero_streak,
            leave_action=leave_action,
            max_value=horizon,
        ),
    )
def _matched_run_ids(
    policies: list[PolicySpec],
    *,
    maze_name: str,
    observable: bool,
    num_datasets: int | None,
    horizon: int | None,
) -> list[int]:
    run_id_sets = [
        set(
            list_run_dataset_run_ids(
                policy.agent,
                maze_name,
                observable,
                context_mode=policy.context_mode,
                horizon=horizon,
            )
        )
        for policy in policies
    ]
    if not run_id_sets:
        return []
    common_run_ids = sorted(set.intersection(*run_id_sets))
    if num_datasets is None:
        return common_run_ids
    return common_run_ids[:num_datasets]


def _aggregate_policy_summaries(
    summaries_by_policy: dict[PolicySpec, list[PolicyRunSummary]],
) -> dict[PolicySpec, dict[str, object]]:
    aggregated: dict[PolicySpec, dict[str, object]] = {}
    for policy, summaries in summaries_by_policy.items():
        aggregated[policy] = {
            "mean_return": [summary.mean_return for summary in summaries],
            "tail_return": [summary.tail_return for summary in summaries],
            "tail_window_episodes": [
                summary.tail_window_episodes for summary in summaries
            ],
            "upper_dwell": [summary.upper_dwell for summary in summaries],
            "lower_dwell": [summary.lower_dwell for summary in summaries],
            "best_upper_threshold": [
                summary.best_upper_threshold for summary in summaries
            ],
            "best_lower_threshold": [
                summary.best_lower_threshold for summary in summaries
            ],
            "patch_threshold_balanced_accuracy": [
                summary.patch_threshold_balanced_accuracy for summary in summaries
            ],
            "best_zero_streak_threshold": [
                summary.best_zero_streak_threshold for summary in summaries
            ],
            "zero_streak_balanced_accuracy": [
                summary.zero_streak_balanced_accuracy for summary in summaries
            ],
            "upper_mvt_deviation": [
                summary.upper_mvt_deviation for summary in summaries
            ],
            "lower_mvt_deviation": [
                summary.lower_mvt_deviation for summary in summaries
            ],
            "upper_leave_prob_auc": [
                summary.upper_leave_prob_auc for summary in summaries
            ],
            "lower_leave_prob_auc": [
                summary.lower_leave_prob_auc for summary in summaries
            ],
            "upper_leave_prob_by_time": _aggregate_curve(
                [summary.upper_leave_prob_by_time for summary in summaries]
            ),
            "lower_leave_prob_by_time": _aggregate_curve(
                [summary.lower_leave_prob_by_time for summary in summaries]
            ),
            "leave_prob_by_zero_streak": _aggregate_curve(
                [summary.leave_prob_by_zero_streak for summary in summaries]
            ),
        }
    return aggregated


def _set_curve_axis_limits(ax: plt.Axes, curve_summaries: Iterable[CurveSummary]) -> None:
    max_index = 0
    for curve in curve_summaries:
        finite = np.flatnonzero(np.isfinite(curve.mean))
        if finite.size > 0:
            max_index = max(max_index, int(finite[-1]))
    ax.set_xlim(0, max_index if max_index > 0 else 1)


def _plot_leave_probability_by_time(
    aggregated: dict[PolicySpec, dict[str, object]],
    *,
    maze_name: str,
    observable: bool,
    filename_suffix: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    patches = ("Upper Patch", "Lower Patch")
    curve_keys = ("upper_leave_prob_by_time", "lower_leave_prob_by_time")

    for ax, patch_label, curve_key in zip(axes, patches, curve_keys, strict=True):
        for policy, summary in aggregated.items():
            curve = summary[curve_key]
            assert isinstance(curve, CurveSummary)
            style = _policy_line_style(policy)
            label = _policy_display_label(policy, include_context=True)
            ax.plot(curve.x, curve.mean, label=label, **style)
            ax.fill_between(
                curve.x,
                np.clip(curve.mean - curve.std, 0.0, 1.0),
                np.clip(curve.mean + curve.std, 0.0, 1.0),
                color=style["color"],
                alpha=0.08,
            )
        ax.set_title(patch_label, fontsize=13)
        ax.set_xlabel("Time Spent In Patch", fontsize=11)
        ax.set_ylabel("Leave Probability", fontsize=11)
        ax.set_ylim(0.0, 1.0)
        _set_curve_axis_limits(
            ax,
            [summary[curve_key] for summary in aggregated.values()],
        )
    axes[1].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(
        "Reward Timing Benchmark\nLeave Probability by Time Spent in Patch",
        fontsize=15,
        fontweight="bold",
    )

    path = _figure_path(
        "leave_prob_by_time",
        maze_name=maze_name,
        observable=observable,
        filename_suffix=filename_suffix,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_leave_probability_by_zero_streak(
    aggregated: dict[PolicySpec, dict[str, object]],
    *,
    maze_name: str,
    observable: bool,
    filename_suffix: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for policy, summary in aggregated.items():
        curve = summary["leave_prob_by_zero_streak"]
        assert isinstance(curve, CurveSummary)
        style = _policy_line_style(policy)
        label = _policy_display_label(policy, include_context=True)
        ax.plot(curve.x, curve.mean, label=label, **style)
        ax.fill_between(
            curve.x,
            np.clip(curve.mean - curve.std, 0.0, 1.0),
            np.clip(curve.mean + curve.std, 0.0, 1.0),
            color=style["color"],
            alpha=0.08,
        )
    ax.set_title("Reward Timing Benchmark\nLeave Probability by Zero-Reward Streak", fontsize=15)
    ax.set_xlabel("Consecutive Zero-Reward Streak", fontsize=11)
    ax.set_ylabel("Leave Probability", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    _set_curve_axis_limits(
        ax,
        [summary["leave_prob_by_zero_streak"] for summary in aggregated.values()],
    )
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    path = _figure_path(
        "leave_prob_by_zero_streak",
        maze_name=maze_name,
        observable=observable,
        filename_suffix=filename_suffix,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_dwell_length_by_patch(
    aggregated: dict[PolicySpec, dict[str, object]],
    *,
    maze_name: str,
    observable: bool,
    filename_suffix: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    labels = [
        _policy_display_label(policy, include_context=True) for policy in aggregated
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.35
    upper_means = [
        _mean_std(summary["upper_dwell"])[0] for summary in aggregated.values()
    ]
    upper_stds = [_mean_std(summary["upper_dwell"])[1] for summary in aggregated.values()]
    lower_means = [
        _mean_std(summary["lower_dwell"])[0] for summary in aggregated.values()
    ]
    lower_stds = [_mean_std(summary["lower_dwell"])[1] for summary in aggregated.values()]

    ax.bar(
        x - width / 2.0,
        upper_means,
        width,
        yerr=upper_stds,
        label="Upper Patch",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2.0,
        lower_means,
        width,
        yerr=lower_stds,
        label="Lower Patch",
        color="#ff7f0e",
        alpha=0.85,
    )
    ax.set_ylabel("Mean Dwell Length Before Leave", fontsize=11)
    ax.set_title("Reward Timing Benchmark\nMean Dwell Length by Patch", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    path = _figure_path(
        "dwell_length_by_patch",
        maze_name=maze_name,
        observable=observable,
        filename_suffix=filename_suffix,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_mvt_deviation_and_auc(
    aggregated: dict[PolicySpec, dict[str, object]],
    *,
    maze_name: str,
    observable: bool,
    filename_suffix: str,
) -> Path:
    fig, (ax_deviation, ax_auc) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        constrained_layout=True,
    )
    labels = [
        _policy_display_label(policy, include_context=True) for policy in aggregated
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.35

    upper_deviation_means = [
        _mean_std(summary["upper_mvt_deviation"])[0] for summary in aggregated.values()
    ]
    upper_deviation_stds = [
        _mean_std(summary["upper_mvt_deviation"])[1] for summary in aggregated.values()
    ]
    lower_deviation_means = [
        _mean_std(summary["lower_mvt_deviation"])[0] for summary in aggregated.values()
    ]
    lower_deviation_stds = [
        _mean_std(summary["lower_mvt_deviation"])[1] for summary in aggregated.values()
    ]
    upper_auc_means = [
        _mean_std(summary["upper_leave_prob_auc"])[0] for summary in aggregated.values()
    ]
    upper_auc_stds = [
        _mean_std(summary["upper_leave_prob_auc"])[1] for summary in aggregated.values()
    ]
    lower_auc_means = [
        _mean_std(summary["lower_leave_prob_auc"])[0] for summary in aggregated.values()
    ]
    lower_auc_stds = [
        _mean_std(summary["lower_leave_prob_auc"])[1] for summary in aggregated.values()
    ]

    ax_deviation.bar(
        x - width / 2.0,
        upper_deviation_means,
        width,
        yerr=upper_deviation_stds,
        label="Upper Patch",
        color="#1f77b4",
        alpha=0.85,
    )
    ax_deviation.bar(
        x + width / 2.0,
        lower_deviation_means,
        width,
        yerr=lower_deviation_stds,
        label="Lower Patch",
        color="#ff7f0e",
        alpha=0.85,
    )
    ax_deviation.axhline(0.0, color="#2f2f2f", linewidth=1.0, alpha=0.6)
    ax_deviation.set_ylabel("Mean Signed Deviation From MVT Dwell", fontsize=11)
    ax_deviation.set_title("Patch Residency Deviation", fontsize=13)
    ax_deviation.set_xticks(x)
    ax_deviation.set_xticklabels(labels, rotation=30, ha="right")
    ax_deviation.legend()

    ax_auc.bar(
        x - width / 2.0,
        upper_auc_means,
        width,
        yerr=upper_auc_stds,
        label="Upper Patch",
        color="#1f77b4",
        alpha=0.85,
    )
    ax_auc.bar(
        x + width / 2.0,
        lower_auc_means,
        width,
        yerr=lower_auc_stds,
        label="Lower Patch",
        color="#ff7f0e",
        alpha=0.85,
    )
    ax_auc.set_ylabel("Normalized Leave-Curve AUC", fontsize=11)
    ax_auc.set_title("Leave Probability by Time AUC", fontsize=13)
    ax_auc.set_ylim(0.0, 1.0)
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(labels, rotation=30, ha="right")
    ax_auc.legend()

    fig.suptitle(
        "Reward Timing Benchmark\nMVT Residency Deviation and Leave-Curve AUC",
        fontsize=15,
        fontweight="bold",
    )

    path = _figure_path(
        "mvt_residency_deviation_auc",
        maze_name=maze_name,
        observable=observable,
        filename_suffix=filename_suffix,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _build_report(
    aggregated: dict[PolicySpec, dict[str, object]],
    *,
    maze_name: str,
    observable: bool,
    matched_run_ids: list[int],
    figure_paths: tuple[Path, ...],
) -> str:
    tail_window_values = {
        int(window)
        for summary in aggregated.values()
        for window in summary["tail_window_episodes"]
    }
    tail_window_label = (
        str(next(iter(tail_window_values)))
        if len(tail_window_values) == 1
        else "varies"
    )
    table_lines = [
        "| policy | tail return | upper dwell | lower dwell | best (k_upper, k_lower) | patch-threshold balanced accuracy | best n | zero-streak balanced accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    mvt_table_lines = [
        "| policy | upper MVT deviation | lower MVT deviation | upper leave-time AUC | lower leave-time AUC |",
        "| --- | --- | --- | --- | --- |",
    ]

    for policy, summary in aggregated.items():
        upper_threshold = _format_mean_std(summary["best_upper_threshold"], digits=1)
        lower_threshold = _format_mean_std(summary["best_lower_threshold"], digits=1)
        best_thresholds = f"({upper_threshold}, {lower_threshold})"
        table_lines.append(
            "| "
            + " | ".join(
                [
                    _policy_display_label(policy, include_context=True),
                    _format_mean_std(summary["tail_return"]),
                    _format_mean_std(summary["upper_dwell"]),
                    _format_mean_std(summary["lower_dwell"]),
                    best_thresholds,
                    _format_mean_std(
                        summary["patch_threshold_balanced_accuracy"],
                        digits=3,
                    ),
                    _format_mean_std(summary["best_zero_streak_threshold"], digits=1),
                    _format_mean_std(
                        summary["zero_streak_balanced_accuracy"],
                        digits=3,
                    ),
                ]
            )
            + " |"
        )
        mvt_table_lines.append(
            "| "
            + " | ".join(
                [
                    _policy_display_label(policy, include_context=True),
                    _format_mean_std(summary["upper_mvt_deviation"]),
                    _format_mean_std(summary["lower_mvt_deviation"]),
                    _format_mean_std(summary["upper_leave_prob_auc"], digits=3),
                    _format_mean_std(summary["lower_leave_prob_auc"], digits=3),
                ]
            )
            + " |"
        )

    figure_lines = [
        f"- `{path.name}`" for path in figure_paths
    ]
    return "\n".join(
        [
            "# Reward Timing Benchmark",
            "",
            f"- Setting: `{maze_name}/{_obs_tag(observable)}`",
            f"- Matched runs: `{len(matched_run_ids)}` (`{matched_run_ids[0]}`-`{matched_run_ids[-1]}`)"
            if matched_run_ids
            else "- Matched runs: `0`",
            f"- Note: {BENCHMARK_NOTE}",
            "- Dwell length uses `time_spent + 1` at the leave decision.",
            (
                "- Tail return uses the last 20% of episodes per run "
                f"(window=`{tail_window_label}` here)."
            ),
            "- Thresholds and balanced accuracies are fit per run, then summarized as mean ± std.",
            "- MVT deviation uses the true-state full-MDP value-iteration policy as the dwell reference and is reported in dwell-length units.",
            "- Leave-time AUC is normalized over the finite observed support of each run's leave-probability-by-time curve.",
            "",
            "## Summary Table",
            "",
            *table_lines,
            "",
            "## MVT Timing Summary",
            "",
            *mvt_table_lines,
            "",
            "## Figures",
            "",
            *figure_lines,
        ]
    )


def analyze_reward_timing_benchmark(
    *,
    maze_name: str = "full",
    observable: bool = False,
    policies: list[PolicySpec] | None = None,
    num_datasets: int | None = None,
    filename_suffix: str = "reward_timing_benchmark",
    horizon: int | None = None,
    verbose: bool = True,
) -> BenchmarkArtifacts:
    """Generate reward-timing benchmark figures and Markdown report."""
    if observable:
        raise ValueError("Reward-timing benchmark is defined only for partially observable runs.")

    config_module.ensure_directories()
    policies = reward_timing_policies() if policies is None else policies
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    run_ids = _matched_run_ids(
        policies,
        maze_name=maze_name,
        observable=observable,
        num_datasets=num_datasets,
        horizon=horizon,
    )
    if not run_ids:
        raise ValueError(
            "No matched run datasets are available for the reward-timing benchmark. "
            "Generate reward-timing trajectories first."
        )

    maze_spec = load_builtin_maze_spec(maze_name)
    leave_action = maze_spec.maze.action_labels.index("leave")
    patch_labels = _observation_group_patch_labels(maze_name)
    optimal_dwell_by_state = _mvt_optimal_dwell_by_state(
        maze_name=maze_name,
        horizon=resolved_horizon,
    )
    summaries_by_policy: dict[PolicySpec, list[PolicyRunSummary]] = {
        policy: [] for policy in policies
    }

    for policy in policies:
        for run_id in run_ids:
            run_dataset = load_run_dataset(
                policy.agent,
                run_id,
                maze_name,
                observable,
                context_mode=policy.context_mode,
                horizon=horizon,
            )
            summaries_by_policy[policy].append(
                summarize_policy_run(
                    run_dataset,
                    maze_name=maze_name,
                    patch_labels=patch_labels,
                    leave_action=leave_action,
                    horizon=resolved_horizon,
                    optimal_dwell_by_state=optimal_dwell_by_state,
                )
            )

    aggregated = _aggregate_policy_summaries(summaries_by_policy)
    figure_paths = (
        _plot_leave_probability_by_time(
            aggregated,
            maze_name=maze_name,
            observable=observable,
            filename_suffix=filename_suffix,
        ),
        _plot_leave_probability_by_zero_streak(
            aggregated,
            maze_name=maze_name,
            observable=observable,
            filename_suffix=filename_suffix,
        ),
        _plot_dwell_length_by_patch(
            aggregated,
            maze_name=maze_name,
            observable=observable,
            filename_suffix=filename_suffix,
        ),
        _plot_mvt_deviation_and_auc(
            aggregated,
            maze_name=maze_name,
            observable=observable,
            filename_suffix=filename_suffix,
        ),
    )
    report_path = _report_path(
        maze_name=maze_name,
        observable=observable,
        filename_suffix=filename_suffix,
    )
    report_path.write_text(
        _build_report(
            aggregated,
            maze_name=maze_name,
            observable=observable,
            matched_run_ids=run_ids,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )

    if verbose:
        print(f"Saved reward-timing report to {report_path}")
        for figure_path in figure_paths:
            print(f"Saved reward-timing figure to {figure_path}")

    return BenchmarkArtifacts(
        report_path=report_path,
        figure_paths=figure_paths,
        matched_run_ids=tuple(run_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the full/PO reward-timing benchmark report and figures.",
    )
    parser.add_argument(
        "--maze",
        default="full",
        help="Built-in maze spec name; reward-timing benchmark defaults to full.",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=None,
        help="Optional maximum number of matched run datasets to analyze.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional episode-length override used to locate saved artifacts.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output.")

    args = parser.parse_args()
    analyze_reward_timing_benchmark(
        maze_name=args.maze,
        observable=False,
        num_datasets=args.num_datasets,
        horizon=args.horizon,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
