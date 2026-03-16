"""Visualization functions for model comparison and Q-value analysis."""

from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.config import FIGURES_DIR, ensure_directories
from forage_rl.environments.maze import Maze
from forage_rl.types import Trajectory
from .metrics import (
    MISSING_LOGPROBS_MESSAGE as _MISSING_LOGPROBS_MESSAGE,
    compute_mean_cumulative_accuracy,
    compute_model_comparison,
    compute_pairwise_accuracy_matrix,
    compute_pairwise_cumulative_accuracy,
    compute_pairwise_logprob_gap_matrix,
    compute_source_win_rates,
)


def _draw_empty(ax: plt.Axes, message: str) -> None:
    """Render a centered empty-state message."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _env_filename(stem: str, env_key: str | None) -> str:
    return f"{stem}__{env_key}.png" if env_key is not None else f"{stem}.png"


def _env_title(title: str, env_label: str | None) -> str:
    return f"{title} [{env_label}]" if env_label is not None else title


def _maybe_print(verbose: bool, message: str) -> None:
    """Emit a plotting status message when verbosity is enabled."""
    if verbose:
        print(message)


def plot_model_comparison(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot average self-vs-other win rate for each selected source agent."""
    comparison = compute_model_comparison(
        agents=agents,
        num_datasets=num_datasets,
        env_key=env_key,
    )
    if comparison is None:
        _maybe_print(verbose, _MISSING_LOGPROBS_MESSAGE)
        return None
    selected_agents, agent_scores = comparison

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(selected_agents)), 6))
    bars = ax.bar(selected_agents, agent_scores, color="#3498db")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average Win Rate", fontsize=14)
    ax.set_title(
        _env_title("Average Self-Eval Win Rate By Source Agent", env_label),
        fontsize=16,
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    for bar, score in zip(bars, agent_scores):
        if np.isnan(score):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(score + 0.02, 0.97),
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / _env_filename("model_comparison", env_key)
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_cumulative_sum_accuracy(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot average transition-wise classification accuracy across all pairs."""
    avg_accuracy = compute_mean_cumulative_accuracy(
        agents=agents, num_datasets=num_datasets, env_key=env_key
    )
    if avg_accuracy is None:
        _maybe_print(verbose, _MISSING_LOGPROBS_MESSAGE)
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_accuracy, linewidth=3, color="#2ecc71")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Number of Observed Transitions", fontsize=16)
    ax.set_ylabel("Prediction Accuracy", fontsize=16)
    ax.set_title(
        _env_title("All-Agent Classification Accuracy vs. Sample Size", env_label),
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / _env_filename("cumulative_accuracy", env_key)
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def _draw_model_accuracies(
    ax: plt.Axes,
    source: str,
    compare_to: list[str],
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
) -> None:
    """Draw paired win-rate bars for source vs each agent in compare_to onto ax."""
    if not [agent for agent in compare_to if agent != source]:
        _draw_empty(ax, "No comparisons")
        return

    rates = compute_source_win_rates(
        source=source,
        compare_to=compare_to,
        num_datasets=num_datasets,
        env_key=env_key,
    )
    if rates is None:
        _draw_empty(ax, "No comparisons")
        return
    comparisons, source_rates, eval_rates = rates
    if all(np.isnan(rate) for rate in source_rates):
        _draw_empty(ax, _MISSING_LOGPROBS_MESSAGE)
        return

    x = np.arange(len(comparisons))
    width = 0.35
    eval_colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#34495e"]

    ax.bar(x - width / 2, source_rates, width, label=source, color="#3498db")
    for idx, (evaluator, rate) in enumerate(zip(comparisons, eval_rates)):
        ax.bar(
            x[idx] + width / 2,
            rate,
            width,
            label=evaluator
            if idx == 0 or evaluator not in comparisons[:idx]
            else "_nolegend_",
            color=eval_colors[idx % len(eval_colors)],
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_xlabel("Comparison", fontsize=12)
    ax.set_title(
        _env_title(
            f"Model Accuracy on '{source}'-Generated Trajectories",
            env_label,
        ),
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{source} vs {evaluator}" for evaluator in comparisons],
        rotation=30,
        ha="right",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (0.50)")
    ax.legend()

    for idx, (source_rate, evaluator_rate) in enumerate(zip(source_rates, eval_rates)):
        if np.isnan(source_rate) or np.isnan(evaluator_rate):
            continue
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


def plot_model_accuracies_from_trajectory_type(
    source: str,
    compare_to: list[str],
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot paired bar charts comparing source self-eval win rate against each evaluator."""
    n = len([agent for agent in compare_to if agent != source])
    fig, ax = plt.subplots(
        figsize=(max(6, 2.5 * max(n, 1)), 5), constrained_layout=True
    )
    _draw_model_accuracies(ax, source, compare_to, num_datasets, env_key, env_label)

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / _env_filename(f"model_accuracies_{source}", env_key)
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_pairwise_accuracy_matrix(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot a heatmap of self-vs-other final accuracy rates."""
    matrix_data = compute_pairwise_accuracy_matrix(
        agents=agents, num_datasets=num_datasets, env_key=env_key
    )
    if matrix_data is None:
        _maybe_print(verbose, _MISSING_LOGPROBS_MESSAGE)
        return None
    selected_agents, matrix = matrix_data

    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(
        figsize=(max(7, 1.3 * len(selected_agents)), max(6, 1.1 * len(selected_agents)))
    )
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(selected_agents)))
    ax.set_xticklabels(selected_agents, rotation=30, ha="right")
    ax.set_yticks(range(len(selected_agents)))
    ax.set_yticklabels(selected_agents)
    ax.set_xlabel("Evaluator Agent", fontsize=12)
    ax.set_ylabel("Source Agent", fontsize=12)
    ax.set_title(_env_title("Pairwise Final Accuracy Matrix", env_label), fontsize=16)
    fig.colorbar(im, ax=ax, label="Win Rate")

    for row in range(len(selected_agents)):
        for col in range(len(selected_agents)):
            if np.isnan(matrix[row, col]):
                continue
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / _env_filename(
            "pairwise_final_accuracy_matrix", env_key
        )
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_pairwise_logprob_gap_matrix(
    agents: Optional[list[str]] = None,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot a heatmap of mean final cumulative log-prob gaps."""
    matrix_data = compute_pairwise_logprob_gap_matrix(
        agents=agents, num_datasets=num_datasets, env_key=env_key
    )
    if matrix_data is None:
        _maybe_print(verbose, _MISSING_LOGPROBS_MESSAGE)
        return None
    selected_agents, matrix = matrix_data

    masked = np.ma.masked_invalid(matrix)
    vmax = float(np.nanmax(np.abs(matrix))) if not np.isnan(matrix).all() else 1.0
    vmax = max(vmax, 1e-9)
    fig, ax = plt.subplots(
        figsize=(max(7, 1.3 * len(selected_agents)), max(6, 1.1 * len(selected_agents)))
    )
    im = ax.imshow(masked, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_xticks(range(len(selected_agents)))
    ax.set_xticklabels(selected_agents, rotation=30, ha="right")
    ax.set_yticks(range(len(selected_agents)))
    ax.set_yticklabels(selected_agents)
    ax.set_xlabel("Evaluator Agent", fontsize=12)
    ax.set_ylabel("Source Agent", fontsize=12)
    ax.set_title(
        _env_title("Pairwise Final Log-Prob Gap Matrix", env_label),
        fontsize=16,
    )
    fig.colorbar(im, ax=ax, label="Mean Final Gap")

    for row in range(len(selected_agents)):
        for col in range(len(selected_agents)):
            if np.isnan(matrix[row, col]):
                continue
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / _env_filename(
            "pairwise_final_logprob_gap_matrix", env_key
        )
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_pairwise_cumulative_accuracy(
    source: str,
    evaluator: str,
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot mean transition-wise accuracy for a specific source/evaluator pair."""
    avg_accuracy = compute_pairwise_cumulative_accuracy(
        source=source,
        evaluator=evaluator,
        num_datasets=num_datasets,
        env_key=env_key,
    )
    if avg_accuracy is None:
        _maybe_print(verbose, _MISSING_LOGPROBS_MESSAGE)
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_accuracy, linewidth=3, color="#2ecc71")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Number of Observed Transitions", fontsize=16)
    ax.set_ylabel("Prediction Accuracy", fontsize=16)
    ax.set_title(
        _env_title(f"Cumulative Accuracy: {source} vs {evaluator}", env_label),
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / _env_filename(
            f"cumulative_accuracy__source_{source}__vs_{evaluator}",
            env_key,
        )
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_q_values_with_time(
    q_agent: BaseAgent,
    max_time_to_display: int = 6,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot Q-values as heatmaps for each action.

    Raises:
        ValueError: If ``max_time_to_display`` is not positive or ``q_table``
            has no time-axis entries.
    """
    if max_time_to_display <= 0:
        raise ValueError("max_time_to_display must be > 0")

    q_table = q_agent.q_table
    if q_table.shape[1] <= 0:
        raise ValueError("q_table must include at least one time step")

    maze = q_agent.maze
    num_states = maze.agent_num_states

    max_time = min(max_time_to_display, q_table.shape[1])
    fig, axes = plt.subplots(
        1,
        maze.num_actions,
        figsize=(max(6, 6 * maze.num_actions), 5),
    )
    if maze.num_actions == 1:
        axes = [axes]
    else:
        axes = list(np.atleast_1d(axes))

    vmin = np.min(q_table[:, :max_time, :])
    vmax = np.max(q_table[:, :max_time, :])

    for action_idx, action in enumerate(range(maze.num_actions)):
        im = axes[action_idx].imshow(
            q_table[:, :max_time, action], vmin=vmin, vmax=vmax
        )
        axes[action_idx].set_title(
            f"Q-values for Action {action} ({maze.get_action_label(action_idx)})"
        )
        axes[action_idx].set_xlabel("Time Spent")
        axes[action_idx].set_ylabel("States")
        axes[action_idx].set_xticks(range(max_time))
        axes[action_idx].set_xticklabels([f"t={step}" for step in range(max_time)])
        axes[action_idx].set_yticks(range(num_states))

        state_labels = maze.agent_state_labels or [
            f"State {state}" for state in range(num_states)
        ]
        axes[action_idx].set_yticklabels(state_labels)

        for y_idx in range(num_states):
            for x_idx in range(max_time):
                value = q_table[y_idx, x_idx, action]
                color = (
                    "w"
                    if (vmax - vmin) > 0 and (value - vmin) / (vmax - vmin) > 0.5
                    else "black"
                )
                axes[action_idx].text(
                    x_idx,
                    y_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        fig.colorbar(im, ax=axes[action_idx])

    plt.tight_layout()

    if save:
        ensure_directories()
        dtm = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filepath = FIGURES_DIR / f"{dtm}_q_values_with_time.png"
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def _draw_mean_cumulative_reward(ax: plt.Axes, trajectories: list[Trajectory]) -> None:
    """Draw mean cumulative reward with ±1 SD shading across trajectories."""
    if not trajectories:
        _draw_empty(ax, "No trajectories")
        return

    cumsums = [
        np.cumsum([transition.reward for transition in trajectory.transitions])
        for trajectory in trajectories
    ]
    min_len = min(len(cumsum) for cumsum in cumsums)
    arr = np.array([cumsum[:min_len] for cumsum in cumsums])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x_axis = np.arange(min_len)
    ax.plot(x_axis, mean, linewidth=2, color="#2ecc71", label="Mean")
    ax.fill_between(
        x_axis, mean - std, mean + std, alpha=0.3, color="#2ecc71", label="±1 SD"
    )
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title(f"Mean Cumulative Reward (n={len(trajectories)})", fontsize=14)
    ax.legend(fontsize=10)


def _draw_modal_residency(
    ax: plt.Axes,
    trajectories: list[Trajectory],
    maze: Maze,
) -> None:
    """Draw the modal state at each time step across trajectories."""
    if not trajectories:
        _draw_empty(ax, "No trajectories")
        return

    state_sequences = [
        [
            getattr(transition, "true_state", transition.state)
            for transition in trajectory.transitions
        ]
        for trajectory in trajectories
    ]
    min_len = min(len(sequence) for sequence in state_sequences)
    arr = np.array([sequence[:min_len] for sequence in state_sequences])

    modal_states = []
    frequencies = []
    for step in range(min_len):
        counts = np.bincount(arr[:, step], minlength=maze.true_num_states)
        modal = int(np.argmax(counts))
        modal_states.append(modal)
        frequencies.append(counts[modal] / len(trajectories))

    sizes = np.array(frequencies) * 40
    ax.scatter(range(min_len), modal_states, s=sizes, alpha=0.7, color="#3498db")
    state_labels = maze.true_state_labels or [
        f"State {state}" for state in range(maze.true_num_states)
    ]
    ax.set_yticks(range(maze.true_num_states))
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title(f"Modal Residency Location (n={len(trajectories)})", fontsize=14)


def plot_mean_trajectory_stats(
    trajectories: list[Trajectory],
    maze: Maze,
    source: str,
    compare_to: list[str],
    num_datasets: Optional[int] = None,
    env_key: str | None = None,
    env_label: str | None = None,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
):
    """Plot aggregate reward, residency, and model-accuracy summaries."""
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax_reward = fig.add_subplot(gs[0, 0])
    ax_residency = fig.add_subplot(gs[0, 1])
    ax_accuracy = fig.add_subplot(gs[1, :])

    _draw_mean_cumulative_reward(ax_reward, trajectories)
    _draw_modal_residency(ax_residency, trajectories, maze)
    _draw_model_accuracies(
        ax_accuracy, source, compare_to, num_datasets, env_key, env_label
    )

    fig.suptitle(
        _env_title(
            f"Average Trajectory Overview: '{source}' (n={len(trajectories)})",
            env_label,
        ),
        fontsize=16,
        fontweight="bold",
    )

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / _env_filename(
            f"mean_trajectory_stats_{source}", env_key
        )
        plt.savefig(filepath, dpi=150)
        _maybe_print(verbose, f"Saved to {filepath}")
    if show:
        plt.show()
    return fig
