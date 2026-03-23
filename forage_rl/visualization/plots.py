"""Visualization functions for model comparison and Q-value analysis."""

from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.registry import Agent
from forage_rl.config import FIGURES_DIR, ensure_directories
from forage_rl.environments.maze import Maze, MazeMDP, MazePOMDP
from forage_rl.environments.maze import maze_from_builtin_maze_spec
from forage_rl.types import Trajectory
from forage_rl.utils import get_run_count, load_logprobs, load_run_dataset


def _draw_cumulative_accuracy(
    ax: plt.Axes,
    source: Agent,
    compare_to: list[Agent],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
) -> None:
    """Draw per-evaluator win-rate-over-time lines onto ax."""
    comparisons = [a for a in compare_to if a != source]
    if not comparisons:
        ax.text(
            0.5, 0.5, "No comparisons", ha="center", va="center", transform=ax.transAxes
        )
        return

    num_datasets = num_datasets or get_run_count(source, maze_name, observable)
    if num_datasets == 0:
        ax.text(
            0.5,
            0.5,
            f"No data for '{source}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    for evaluator in comparisons:
        accuracies = []
        for j in range(num_datasets):
            source_cumsum = load_logprobs(source, source, j, maze_name, observable)
            eval_cumsum = load_logprobs(source, evaluator, j, maze_name, observable)
            accuracy = np.where(
                np.isclose(source_cumsum, eval_cumsum),
                0.5,
                (source_cumsum > eval_cumsum).astype(float),
            )
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies, axis=0)
        ax.plot(avg_accuracy, linewidth=3, label=evaluator.value)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Number of Observed Transitions", fontsize=16)
    ax.set_ylabel("Prediction Accuracy", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)


def plot_cumulative_sum_accuracy(
    source: Agent,
    compare_to: list[Agent],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    save: bool = False,
    show: bool = True,
):
    """Plot source-model win rate over observed transitions against each evaluator.

    For each evaluator in compare_to (excluding source), plots the fraction of
    timesteps where the source model's cumulative log-likelihood exceeds the
    evaluator's, averaged across runs.

    Args:
        source: Agent whose trajectories are being evaluated
        compare_to: Agents to compare against
        maze_name: Built-in maze spec name
        num_datasets: Number of datasets to analyze; defaults to all available
        observable: True for fully observable (FO), False for partially observable (PO)
        save: Whether to save the figure
        show: Whether to display the figure
    """
    comparisons = [a for a in compare_to if a != source]
    if not comparisons:
        print("No comparisons to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_cumulative_accuracy(
        ax, source, compare_to, maze_name, num_datasets, observable
    )
    ax.set_title(
        f"Classification Accuracy over Sample Size (Trajectory source: {source.value})",
        fontsize=16,
    )

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / f"cumulative_accuracy_{source.value}.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def _draw_model_accuracies(
    ax: plt.Axes,
    source: Agent,
    compare_to: list[Agent],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
) -> None:
    """Draw paired win-rate bars for source vs each agent in compare_to onto ax."""
    comparisons = [a for a in compare_to if a != source]
    if not comparisons:
        ax.text(
            0.5, 0.5, "No comparisons", ha="center", va="center", transform=ax.transAxes
        )
        return

    num_datasets = num_datasets or get_run_count(source, maze_name, observable)
    if num_datasets == 0:
        ax.text(
            0.5,
            0.5,
            f"No data for '{source}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    source_wins = {ev: 0 for ev in comparisons}
    for i in range(num_datasets):
        source_final = load_logprobs(source, source, i, maze_name, observable)[-1]
        for ev in comparisons:
            eval_final = load_logprobs(source, ev, i, maze_name, observable)[-1]
            if source_final > eval_final:
                source_wins[ev] += 1

    n = len(comparisons)
    x = np.arange(n)
    width = 0.35
    eval_colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    source_rates = [source_wins[ev] / num_datasets for ev in comparisons]
    eval_rates = [1 - r for r in source_rates]

    ax.bar(x - width / 2, source_rates, width, label=source, color="#3498db")
    for i, (ev, rate) in enumerate(zip(comparisons, eval_rates)):
        ax.bar(
            x[i] + width / 2,
            rate,
            width,
            label=ev if i == 0 or ev not in comparisons[:i] else "_nolegend_",
            color=eval_colors[i % len(eval_colors)],
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_xlabel("Comparison", fontsize=12)
    ax.set_title(f"Model Accuracy on '{source}'-Generated Trajectories", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{source} vs {ev}" for ev in comparisons], ha="right")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance (0.50)")
    ax.legend()

    for i, (s_rate, e_rate) in enumerate(zip(source_rates, eval_rates)):
        ax.text(
            x[i] - width / 2,
            min(s_rate + 0.02, 0.97),
            f"{s_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            x[i] + width / 2,
            min(e_rate + 0.02, 0.97),
            f"{e_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _draw_cumulative_reward(ax: plt.Axes, trajectory: "Trajectory") -> None:
    """Draw cumulative reward over transitions onto ax."""
    rewards = [t.reward for t in trajectory.transitions]
    ax.plot(np.cumsum(rewards), linewidth=2, color="#2ecc71")
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title("Cumulative Reward Over Time", fontsize=14)


def _draw_residency_location(
    ax: plt.Axes, trajectory: "Trajectory", maze: "Maze"
) -> None:
    """Draw state residency scatter plot onto ax."""
    states = [t.state for t in trajectory.transitions]
    state_labels = maze.state_labels or [f"State {s}" for s in range(maze.num_states)]
    ax.scatter(range(len(states)), states, s=8, alpha=0.6, color="#3498db")
    ax.set_yticks(range(maze.num_states))
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title("Residency Location Over Time", fontsize=14)


def plot_model_accuracies_from_trajectory_type(
    source: Agent,
    compare_to: list[Agent],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    save: bool = False,
    show: bool = True,
):
    """Plot paired bar charts comparing source self-eval win rate against each evaluator.

    For each agent in compare_to, shows a head-to-head comparison: the fraction of
    datasets where the source agent assigns higher final log-likelihood than the evaluator
    (and vice versa) on source-generated trajectories.

    Args:
        source: Agent type whose saved trajectories to load (e.g. "mbrl")
        compare_to: Evaluator agent names to compare against the source (source excluded)
        num_datasets: Number of trajectory files to analyze; defaults to all available
        observable: True for fully observable (FO), False for partially observable (PO)
        save: Whether to save the figure
        show: Whether to display the figure
    """
    n = len([a for a in compare_to if a != source])
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n), 5), constrained_layout=True)
    _draw_model_accuracies(ax, source, compare_to, maze_name, num_datasets, observable)

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / f"model_accuracies_{source}.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_q_values_with_time(
    q_agent: BaseAgent,
    max_time_to_display: int = 6,
    save: bool = False,
    show: bool = True,
):
    """Plot Q-values as heatmaps for each action.

    Args:
        q_agent: 3D (with time) Q-Agent which has been trained
        max_time_to_display: Maximum time steps to show
        save: Whether to save the figure
        show: Whether to display the figure
    """
    q_table = q_agent.q_table
    maze = q_agent.maze
    num_states = maze.num_states

    max_time = min(max_time_to_display, q_table.horizon)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    q_array = q_table.to_array()
    horizon_vals = q_array[:, :max_time, :]
    vmin = np.min(horizon_vals)
    vmax = np.max(horizon_vals)

    for i, action in enumerate(range(maze.num_actions)):
        im = axes[i].imshow(q_array[:, :max_time, action], vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Q-values for Action {action} ({maze.get_action_label(i)})")
        axes[i].set_xlabel("Time Spent")
        axes[i].set_ylabel("States")
        axes[i].set_xticks(range(max_time))
        axes[i].set_xticklabels([f"t={j}" for j in range(max_time)])
        axes[i].set_yticks(range(num_states))

        state_labels = maze.state_labels or [f"State {s}" for s in range(num_states)]
        axes[i].set_yticklabels(state_labels)

        # Annotate cells
        for y in range(num_states):
            for x in range(max_time):
                value = q_array[y, x, action]
                color = (
                    "w"
                    if (vmax - vmin) > 0 and (value - vmin) / (vmax - vmin) > 0.5
                    else "black"
                )
                axes[i].text(
                    x,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        fig.colorbar(im, ax=axes[i])

    plt.tight_layout()

    if save:
        ensure_directories()
        dtm = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filepath = FIGURES_DIR / f"{dtm}_q_values_with_time.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_q_values(q_agent: BaseAgent, show: bool = True):
    """Plot Q-values as a heatmap.

    Args:
        q_agent: 2D Q-Agent which has been trained
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(q_agent.q_table.to_array())
    maze = q_agent.maze
    ax.set_title("Q-values")
    ax.set_xlabel("Actions")
    ax.set_xticks(list(range(maze.num_actions)))
    ax.set_xticklabels(maze.action_labels)
    ax.set_ylabel("States")
    ax.set_yticks(range(maze.num_states))
    ax.set_yticklabels(
        maze.state_labels or [f"State {s}" for s in range(maze.num_states)]
    )
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_cumulative_reward(
    trajectory: "Trajectory",
    save: bool = False,
    show: bool = True,
):
    """Plot cumulative reward over transitions for a single trajectory.

    Args:
        trajectory: Trajectory whose rewards to plot
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    _draw_cumulative_reward(ax, trajectory)
    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "cumulative_reward.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def plot_residency_location(
    trajectory: "Trajectory",
    maze: "Maze",
    save: bool = False,
    show: bool = True,
):
    """Plot state residency as a scatter plot over transitions.

    Args:
        trajectory: Trajectory whose states to plot
        maze: Maze providing state labels
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    _draw_residency_location(ax, trajectory, maze)
    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "residency_location.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def plot_trajectory_stats(
    trajectory: "Trajectory",
    maze: "Maze",
    source: Agent,
    save: bool = False,
    show: bool = True,
):
    """High-level overview of a single trajectory: cumulative reward and residency.

    Left panel: cumulative reward over transitions.
    Right panel: state residency scatter plot.

    Args:
        trajectory: Trajectory to visualise
        maze: Maze providing state labels
        source: Agent type that generated the trajectory
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, (ax_reward, ax_residency) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    _draw_cumulative_reward(ax_reward, trajectory)
    _draw_residency_location(ax_residency, trajectory, maze)

    fig.suptitle(f"Trajectory Overview: '{source}'", fontsize=16, fontweight="bold")

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / f"trajectory_stats_{source}.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def _draw_mean_cumulative_reward(
    ax: plt.Axes,
    reward_sequences: list[np.ndarray],
) -> None:
    """Draw cumulative reward summary across episode-level reward sequences."""
    cumsums = [np.cumsum(rewards) for rewards in reward_sequences]
<<<<<<< HEAD
    min_len = min(len(c) for c in cumsums)
    arr = np.array([c[:min_len] for c in cumsums])  # (n_runs, min_len)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x = np.arange(min_len)
    ax.plot(x, mean, linewidth=2, color="#2ecc71", label="Mean")
    ax.fill_between(
        x, mean - std, mean + std, alpha=0.3, color="#2ecc71", label="±1 SD"
    )
    ax.set_xlabel("Transition", fontsize=12)
=======
    min_len = min(len(cumsum) for cumsum in cumsums)
    arr = np.array([cumsum[:min_len] for cumsum in cumsums])
    mean = arr.mean(axis=0)
    x = np.arange(min_len)

    if len(reward_sequences) == 1:
        ax.plot(x, mean, linewidth=2, color="#2ecc71", label="Episode")
        ax.set_title("Cumulative Reward (single episode)", fontsize=14)
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
            f"Mean Cumulative Reward (episodes={len(reward_sequences)})",
            fontsize=14,
        )

    ax.set_xlabel("Transition Within Episode", fontsize=12)
>>>>>>> b83a7d1 (update)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.legend(fontsize=10)


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


def _occupancy_fractions(
    state_sequences: list[np.ndarray],
    domain_ids: list[int] | np.ndarray,
) -> np.ndarray:
    """Return occupancy fractions over an explicit id domain at each timestep."""
    min_len = min(len(states) for states in state_sequences)
    arr = np.array([states[:min_len] for states in state_sequences])
    return np.array([(arr == state_id).mean(axis=0) for state_id in domain_ids])


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


def _draw_residency_fractions(
    ax: plt.Axes,
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
) -> None:
<<<<<<< HEAD
    """Draw the modal (most frequent) location at each time step across runs.

    Marker size is proportional to the fraction of runs in the modal bin.
    For partially observable mazes, accepts either raw states or observation-group
    sequences and labels the y-axis with observation group names.
    """
    min_len = min(len(states) for states in state_sequences)
    arr = np.asarray([states[:min_len] for states in state_sequences], dtype=int)

    if isinstance(maze, MazePOMDP):
        if np.all((arr >= 0) & (arr < maze.num_observations)):
            binned_arr = arr
        else:
            obs_map = maze._state_to_observation_group
            binned_arr = np.vectorize(obs_map.__getitem__)(arr)
        n_bins = maze.num_observations
        y_labels = maze.maze_spec.observation_labels
    else:
        binned_arr = arr
        n_bins = maze.num_states
        y_labels = maze.state_labels or [f"State {s}" for s in range(n_bins)]

    modal_vals = []
    frequencies = []
    for step in range(min_len):
        counts = np.bincount(binned_arr[:, step], minlength=n_bins)
        modal = int(np.argmax(counts))
        modal_vals.append(modal)
        frequencies.append(counts[modal] / len(state_sequences))

    sizes = np.array(frequencies) * 40
    ax.scatter(range(min_len), modal_vals, s=sizes, alpha=0.7, color="#3498db")
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title(f"Modal Residency Location (n={len(state_sequences)})", fontsize=14)


=======
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


>>>>>>> b83a7d1 (update)
def plot_mean_cumulative_reward(
    reward_sequences: list[np.ndarray],
    save: bool = False,
    show: bool = True,
):
    """Plot cumulative reward summary across episode-level sequences."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    _draw_mean_cumulative_reward(ax, reward_sequences)
    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "mean_cumulative_reward.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def plot_modal_residency(
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
    save: bool = False,
    show: bool = True,
):
    """Plot per-state occupancy fraction across episode-level sequences."""
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    _draw_residency_fractions(ax, state_sequences, maze)
    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "modal_residency.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def plot_mean_trajectory_stats(
    reward_sequences: list[np.ndarray],
    state_sequences: list[np.ndarray],
    maze: MazeMDP,
    source: Agent,
    save: bool = False,
    show: bool = True,
    run_count: Optional[int] = None,
):
    """High-level overview aggregated across episode trajectories."""
    fig, (ax_reward, ax_residency) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    if not reward_sequences or not state_sequences:
        ax_reward.text(0.5, 0.5, "No episode data", ha="center", va="center")
        ax_residency.text(0.5, 0.5, "No episode data", ha="center", va="center")
        fig.suptitle(
            f"Aggregate Trajectory Overview: '{source}' (episodes=0, runs={run_count or 0})",
            fontsize=16,
            fontweight="bold",
        )
        if show:
            plt.show()
        return fig

    _draw_mean_cumulative_reward(ax_reward, reward_sequences)
    _draw_residency_fractions(ax_residency, state_sequences, maze)

    episode_count = len(reward_sequences)
    resolved_run_count = run_count if run_count is not None else episode_count
    if episode_count == 1:
        title = (
            f"Trajectory Overview: '{source}' (episodes={episode_count}, runs={resolved_run_count})"
        )
    else:
        title = (
            f"Aggregate Trajectory Overview: '{source}' "
            f"(episodes={episode_count}, runs={resolved_run_count})"
        )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    fig.text(
        0.5,
        0.01,
        "Episode-aligned summary: episodes are aligned by within-episode "
        "timestep; x-axis spans the episode horizon, not total transitions "
        "across all episodes.",
        ha="center",
        fontsize=10,
        color="#7f8c8d",
    )
    note = _low_sample_note(resolved_run_count, episode_count)
    if note is not None:
        fig.text(0.5, 0.04, note, ha="center", fontsize=10, color="#7f8c8d")

    if save:
        ensure_directories()
        obs_tag = "PO" if isinstance(maze, MazePOMDP) else "FO"
        filepath = (
            FIGURES_DIR
            / f"mean_trajectory_stats_{source}_{maze.maze_spec.maze.name}_{obs_tag}.png"
        )
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    elif save:
        plt.close(fig)
    return fig


def plot_aggregate_trajectory_stats(
    source: Agent,
    maze_name: str = "simple",
    observable: bool = True,
    save: bool = True,
    show: bool = True,
) -> None:
    """Load saved run datasets for source and plot episode-level trajectory stats."""
    run_ids = list_run_dataset_run_ids(source, maze_name, observable)
    run_datasets = [
        load_run_dataset(source, run_id, maze_name, observable) for run_id in run_ids
    ]
    reward_sequences = [
        np.array([transition.reward for transition in trajectory.transitions])
        for run_dataset in run_datasets
        for trajectory in run_dataset.trajectories
    ]
    state_sequences = [
        np.array([transition.state for transition in trajectory.transitions])
        for run_dataset in run_datasets
        for trajectory in run_dataset.trajectories
    ]
    maze = maze_from_builtin_maze_spec(maze_name, observable)
    plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        source,
        save=save,
        show=show,
        run_count=len(run_ids),
    )
<<<<<<< HEAD


def plot_aggregate_comparison(
    source: Agent,
    compare_to: list[Agent],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    save: bool = False,
    show: bool = True,
):
    """Plot a 2-panel model comparison figure for source vs evaluators.

    Left panel: overall win-rate bar chart.
    Right panel: win rate over observed transitions (cumulative accuracy).

    Args:
        source: Agent whose trajectories are being evaluated
        compare_to: Agents to compare against
        maze_name: Built-in maze spec name
        num_datasets: Number of datasets to analyze; defaults to all available
        observable: True for fully observable (FO), False for partially observable (PO)
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, (ax_bars, ax_lines) = plt.subplots(
        1, 2, figsize=(16, 6), constrained_layout=True
    )

    _draw_model_accuracies(
        ax_bars, source, compare_to, maze_name, num_datasets, observable
    )
    _draw_cumulative_accuracy(
        ax_lines, source, compare_to, maze_name, num_datasets, observable
    )

    fig.suptitle(
        f"Model Comparison: '{source.value}' vs evaluators",
        fontsize=16,
        fontweight="bold",
    )

    if save:
        ensure_directories()
        comparisons = "_".join([a for a in compare_to])
        obs_tag = "FO" if observable else "PO"
        filepath = (
            FIGURES_DIR
            / f"agg_compare_{source.value}_to_{comparisons}_{maze_name}_{obs_tag}.png"
        )
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    plot_aggregate_trajectory_stats(
        Agent.QLearning, maze_name="full_one_way", observable=True, save=True
    )
    plot_aggregate_trajectory_stats(
        Agent.QLearning, maze_name="full_one_way", observable=False, save=True
    )
    plot_aggregate_trajectory_stats(
        Agent.MBRL, maze_name="full_one_way", observable=True, save=True
    )
    plot_aggregate_trajectory_stats(
        Agent.MBRL, maze_name="full_one_way", observable=False, save=True
    )

    plot_aggregate_trajectory_stats(
        Agent.QLearning, maze_name="simple_one_way", observable=True, save=True
    )
    plot_aggregate_trajectory_stats(
        Agent.QLearning, maze_name="simple_one_way", observable=False, save=True
    )
    plot_aggregate_trajectory_stats(
        Agent.MBRL, maze_name="simple_one_way", observable=True, save=True
    )
    plot_aggregate_trajectory_stats(
        Agent.MBRL, maze_name="simple_one_way", observable=False, save=True
    )

    # plot_aggregate_trajectory_stats(Agent.MBRL, maze_name="simple", save=True)
    # plot_aggregate_comparison(
    #     Agent.MBRL, [Agent.QLearning], maze_name="simple", save=True
    # )
    # plot_aggregate_comparison(
    #     Agent.QLearning, [Agent.MBRL], maze_name="simple", save=True
    # )
    #
    # plot_aggregate_trajectory_stats(Agent.MBRL, maze_name="full", save=True)
    # plot_aggregate_comparison(
    #     Agent.MBRL, [Agent.QLearning], maze_name="full", save=True
    # )
    # plot_aggregate_comparison(
    #     Agent.QLearning, [Agent.MBRL], maze_name="full", save=True
    # )
    #
    # plot_aggregate_trajectory_stats(
    #     Agent.MBRL, maze_name="full", observable=False, save=True
    # )
    # plot_aggregate_comparison(
    #     Agent.MBRL, [Agent.QLearning], maze_name="full", observable=False, save=True
    # )
    # plot_aggregate_comparison(
    #     Agent.QLearning, [Agent.MBRL], maze_name="full", observable=False, save=True
    # )
=======


def plot_aggregate_comparison(
    source: Agent,
    compare_to: list[EvaluatorInput],
    maze_name: str = "simple",
    num_datasets: Optional[int] = None,
    observable: bool = True,
    save: bool = False,
    show: bool = True,
):
    """Plot a 2-panel source-centric comparison figure for source vs evaluators."""
    fig, (ax_bars, ax_lines) = plt.subplots(
        1, 2, figsize=(18, 6), constrained_layout=True
    )

    run_count = _draw_model_accuracies(
        ax_bars, source, compare_to, maze_name, num_datasets, observable
    )
    _draw_running_win_rate(
        ax_lines, source, compare_to, maze_name, num_datasets, observable
    )

    title = (
        "Source-Centric Model Comparison on Saved Source Trajectories\n"
        f"{_source_policy_description(source)}"
    )
    note = f"Low sample size (runs={run_count})" if run_count < 2 else None
    if note is not None:
        title = f"{title}\n{note}"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    if save:
        ensure_directories()
        comparisons = "_".join(_filename_label(evaluator) for evaluator in compare_to)
        obs_tag = "FO" if observable else "PO"
        filepath = (
            FIGURES_DIR
            / f"agg_compare_{source.value}_to_{comparisons}_{maze_name}_{obs_tag}.png"
        )
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()
    elif save:
        plt.close(fig)

    return fig


if __name__ == "__main__":
    mode_aware_evaluators: list[EvaluatorInput] = [
        Agent.MBRL,
        Agent.QLearning,
        EvaluatorSpec(agent=Agent.DQN, mode="fresh"),
        EvaluatorSpec(agent=Agent.DQN, mode="pretrained"),
        EvaluatorSpec(agent=Agent.DRQN, mode="fresh"),
        EvaluatorSpec(agent=Agent.DRQN, mode="pretrained"),
    ]

    for source_agent in [Agent.MBRL, Agent.QLearning, Agent.DQN, Agent.DRQN]:
        for maze_name in ["simple", "full"]:
            for observable in [True, False]:
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
>>>>>>> b83a7d1 (update)
