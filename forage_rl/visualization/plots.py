"""Visualization functions for model comparison and Q-value analysis."""

from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.registry import Agent
from forage_rl.config import FIGURES_DIR, ensure_directories
from forage_rl.environments.maze import Maze
from forage_rl.environments.maze import maze_from_builtin_maze_spec
from forage_rl.types import Trajectory
from forage_rl.utils import get_run_count, load_logprobs, load_trajectories


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
    trajectories: "list[Trajectory]",
) -> None:
    """Draw mean cumulative reward with ±1 SD shading across trajectories onto ax."""
    cumsums = [np.cumsum([t.reward for t in traj.transitions]) for traj in trajectories]
    min_len = min(len(c) for c in cumsums)
    arr = np.array([c[:min_len] for c in cumsums])  # (n_trajs, min_len)
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
    ax: plt.Axes,
    trajectories: "list[Trajectory]",
    maze: "Maze",
) -> None:
    """Draw the modal (most frequent) state at each time step across trajectories.

    Marker size is proportional to the fraction of trajectories in the modal state.
    For partially observable mazes, bins trajectories by observation group rather
    than raw state, and labels the y-axis with observation group names.
    """
    state_seqs = [[t.state for t in traj.transitions] for traj in trajectories]
    min_len = min(len(s) for s in state_seqs)
    arr = np.array([s[:min_len] for s in state_seqs])  # (n_trajs, min_len)

    if maze.observable:
        obs_map = maze._state_to_observation_group
        obs_arr = np.vectorize(obs_map.__getitem__)(arr)
        n_bins = maze.num_observations
        modal_vals = []
        frequencies = []
        for step in range(min_len):
            counts = np.bincount(obs_arr[:, step], minlength=n_bins)
            modal = int(np.argmax(counts))
            modal_vals.append(modal)
            frequencies.append(counts[modal] / len(trajectories))
        y_labels = maze.current_maze_spec.observation_labels
    else:
        n_bins = maze.num_states
        modal_vals = []
        frequencies = []
        for step in range(min_len):
            counts = np.bincount(arr[:, step], minlength=n_bins)
            modal = int(np.argmax(counts))
            modal_vals.append(modal)
            frequencies.append(counts[modal] / len(trajectories))
        y_labels = maze.state_labels or [f"State {s}" for s in range(n_bins)]

    sizes = np.array(frequencies) * 40
    ax.scatter(range(min_len), modal_vals, s=sizes, alpha=0.7, color="#3498db")
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title(f"Modal Residency Location (n={len(trajectories)})", fontsize=14)


def plot_mean_cumulative_reward(
    trajectories: "list[Trajectory]",
    save: bool = False,
    show: bool = True,
):
    """Plot mean cumulative reward with ±1 SD shading across a list of trajectories.

    Args:
        trajectories: Trajectories to aggregate
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    _draw_mean_cumulative_reward(ax, trajectories)
    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "mean_cumulative_reward.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def plot_modal_residency(
    trajectories: "list[Trajectory]",
    maze: Maze,
    save: bool = False,
    show: bool = True,
):
    """Plot the modal state at each time step across a list of trajectories.

    Args:
        trajectories: Trajectories to aggregate
        maze: Maze providing state labels
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    _draw_modal_residency(ax, trajectories, maze)
    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "modal_residency.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def plot_mean_trajectory_stats(
    trajectories: list[Trajectory],
    maze: Maze,
    source: Agent,
    save: bool = False,
    show: bool = True,
):
    """High-level overview aggregated across multiple trajectories.

    Left panel: mean cumulative reward with SD shading.
    Right panel: modal residency scatter plot.

    Args:
        trajectories: Trajectories to aggregate
        maze: Maze providing state labels
        source: Agent type that generated the trajectories
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, (ax_reward, ax_residency) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    _draw_mean_cumulative_reward(ax_reward, trajectories)
    _draw_modal_residency(ax_residency, trajectories, maze)

    fig.suptitle(
        f"Average Trajectory Overview: '{source}' (n={len(trajectories)})",
        fontsize=16,
        fontweight="bold",
    )

    if save:
        ensure_directories()
        obs_tag = "PO" if maze.observable else "FO"
        filepath = (
            FIGURES_DIR
            / f"mean_trajectory_stats_{source}_{maze.current_maze_spec.maze.name}_{obs_tag}.png"
        )
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    return fig


def plot_aggregate_trajectory_stats(
    source: Agent,
    maze_name: str = "simple",
    observable: bool = True,
    save: bool = True,
) -> None:
    """Load all saved trajectories for source and plot aggregate trajectory stats.

    Produces a 2-panel figure: mean cumulative reward (left) and modal residency (right).

    Args:
        source: Agent whose saved trajectories to load
        maze_name: Built-in maze spec name
        observable: True for fully observable (FO), False for partially observable (PO)
        save: Whether to save the figure
    """
    trajectories = [
        load_trajectories(source, i, maze_name, observable)
        for i in range(get_run_count(source, maze_name, observable))
    ]
    maze = maze_from_builtin_maze_spec(maze_name, observable)
    plot_mean_trajectory_stats(trajectories, maze, source, save=save)


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
