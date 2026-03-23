"""Visualization functions for model comparison and Q-value analysis."""

from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl import RunDataset, Trajectory
from forage_rl.agents import QLearning
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
    note = _setting_note(maze_name, observable)
    if note is not None:
        _add_figure_notes(fig, [note])

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
    *,
    axis_label: str = "Transition Across Run",
    sample_label: str = "runs",
    single_label: str = "Run",
) -> None:
    """Draw cumulative reward summary across aligned reward sequences."""
    cumsums = [np.cumsum(rewards) for rewards in reward_sequences]
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
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.legend(fontsize=10)


def _count_label(values: list[int]) -> str:
    """Format an integer count or count range for figure titles."""
    if not values:
        return "0"

    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return str(unique_values[0])
    return f"{unique_values[0]}-{unique_values[-1]}"


def _setting_note(maze_name: str, observable: bool) -> Optional[str]:
    """Return a note for settings whose observability variants are redundant."""
    if maze_name == "simple" and not observable:
        return (
            "Note: simple/PO is equivalent to simple/FO for this spec because each "
            "observation group maps to one true state."
        )
    return None


def _add_figure_notes(fig: plt.Figure, notes: list[str]) -> None:
    """Render footnotes at the bottom of a figure."""
    for index, note in enumerate(note for note in notes if note):
        fig.text(
            0.5,
            0.01 + (0.03 * index),
            note,
            ha="center",
            fontsize=10,
            color="#7f8c8d",
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


def plot_mean_cumulative_reward(
    reward_sequences: list[np.ndarray],
    save: bool = False,
    show: bool = True,
):
    """Plot cumulative reward summary across aligned reward sequences."""
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
    """Plot per-state occupancy fraction across aligned sequences."""
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
    episodes_per_run: str | int | None = None,
    transitions_per_run: str | int | None = None,
    setting_note: Optional[str] = None,
):
    """High-level overview aggregated across run-level trajectories."""
    fig, (ax_reward, ax_residency) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )
    resolved_run_count = run_count if run_count is not None else len(reward_sequences)
    resolved_episode_label = (
        str(episodes_per_run) if episodes_per_run is not None else "1"
    )
    resolved_transition_label = (
        str(transitions_per_run)
        if transitions_per_run is not None
        else _count_label([len(sequence) for sequence in reward_sequences])
    )
    resolved_setting_note = setting_note or _setting_note(
        maze.maze_spec.maze.name,
        not isinstance(maze, MazePOMDP),
    )

    if not reward_sequences or not state_sequences:
        ax_reward.text(0.5, 0.5, "No episode data", ha="center", va="center")
        ax_residency.text(0.5, 0.5, "No episode data", ha="center", va="center")
        fig.suptitle(
            "Aggregate Trajectory Overview: "
            f"'{source}' (runs={resolved_run_count}, episodes_per_run=0, transitions_per_run=0)",
            fontsize=16,
            fontweight="bold",
        )
        notes: list[str] = []
        if resolved_setting_note is not None:
            notes.append(resolved_setting_note)
        _add_figure_notes(fig, notes)
        if show:
            plt.show()
        elif save:
            plt.close(fig)
        return fig

    _draw_mean_cumulative_reward(
        ax_reward,
        reward_sequences,
        axis_label="Transition Across Run",
        sample_label="runs",
        single_label="Run",
    )
    _draw_modal_residency(ax_residency, state_sequences, maze)

    title_prefix = (
        "Trajectory Overview" if resolved_run_count == 1 else "Aggregate Trajectory Overview"
    )
    title = (
        f"{title_prefix}: '{source}' "
        f"(runs={resolved_run_count}, episodes_per_run={resolved_episode_label}, "
        f"transitions_per_run={resolved_transition_label})"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    notes: list[str] = []
    if resolved_setting_note is not None:
        notes.append(resolved_setting_note)
    if resolved_run_count < 2:
        notes.append(f"Low sample size (runs={resolved_run_count})")
    _add_figure_notes(fig, notes)

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
    """Load saved run datasets for source and plot run-level trajectory stats."""
    run_ids = list_run_dataset_run_ids(source, maze_name, observable)
    run_datasets = [
        load_run_dataset(source, run_id, maze_name, observable) for run_id in run_ids
    ]
    (
        reward_sequences,
        state_sequences,
        episode_counts,
        transition_counts,
    ) = _run_level_sequences(run_datasets)
    maze = maze_from_builtin_maze_spec(maze_name, observable)
    plot_mean_trajectory_stats(
        reward_sequences,
        state_sequences,
        maze,
        source,
        save=save,
        show=show,
        run_count=len(run_ids),
        episodes_per_run=_count_label(episode_counts),
        transitions_per_run=_count_label(transition_counts),
        setting_note=_setting_note(maze_name, observable),
    )


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
