"""Single-trajectory and Q-table plotting implementations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


def draw_cumulative_reward_impl(ax, trajectory) -> None:
    """Draw cumulative reward over transitions onto ax."""
    rewards = [transition.reward for transition in trajectory.transitions]
    ax.plot(np.cumsum(rewards), linewidth=2, color="#2ecc71")
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title("Cumulative Reward Over Time", fontsize=14)


def draw_residency_location_impl(ax, trajectory, maze) -> None:
    """Draw state residency scatter plot onto ax."""
    states = [transition.state for transition in trajectory.transitions]
    state_labels = maze.state_labels or [
        f"State {state}" for state in range(maze.num_states)
    ]
    ax.scatter(range(len(states)), states, s=8, alpha=0.6, color="#3498db")
    ax.set_yticks(range(maze.num_states))
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)
    ax.set_title("Residency Location Over Time", fontsize=14)


def plot_model_accuracies_from_trajectory_type_impl(
    source,
    compare_to,
    *,
    maze_name: str,
    num_datasets: int | None,
    observable: bool,
    save: bool,
    show: bool,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    n = len(deps["comparison_specs"](source, compare_to))
    fig, ax = plt.subplots(
        figsize=(max(6, 2.5 * max(n, 1)), 5),
        constrained_layout=True,
    )
    deps["draw_model_accuracies"](
        ax,
        source,
        compare_to,
        maze_name,
        num_datasets,
        observable,
    )

    if save:
        deps["ensure_directories"]()
        filepath = deps["figures_dir"] / (
            f"model_accuracies_{deps['policy_artifact_label'](source)}.png"
        )
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_q_values_with_time_impl(
    q_agent,
    *,
    max_time_to_display: int,
    save: bool,
    show: bool,
    deps: dict[str, Any],
):
    np = deps["np"]
    plt = deps["plt"]
    q_table = q_agent.q_table
    maze = q_agent.maze
    num_states = maze.num_states

    max_time = min(max_time_to_display, q_table.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmin = np.min(q_table[:, :max_time, :])
    vmax = np.max(q_table[:, :max_time, :])

    for index, action in enumerate(range(maze.num_actions)):
        im = axes[index].imshow(q_table[:, :max_time, action], vmin=vmin, vmax=vmax)
        axes[index].set_title(
            f"Q-values for Action {action} ({maze.get_action_label(index)})"
        )
        axes[index].set_xlabel("Time Spent")
        axes[index].set_ylabel("States")
        axes[index].set_xticks(range(max_time))
        axes[index].set_xticklabels([f"t={step}" for step in range(max_time)])
        axes[index].set_yticks(range(num_states))

        state_labels = maze.state_labels or [
            f"State {state}" for state in range(num_states)
        ]
        axes[index].set_yticklabels(state_labels)

        for y_pos in range(num_states):
            for x_pos in range(max_time):
                value = q_table[y_pos, x_pos, action]
                color = (
                    "w"
                    if (vmax - vmin) > 0 and (value - vmin) / (vmax - vmin) > 0.5
                    else "black"
                )
                axes[index].text(
                    x_pos,
                    y_pos,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        fig.colorbar(im, ax=axes[index])

    plt.tight_layout()

    if save:
        deps["ensure_directories"]()
        dtm = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filepath = deps["figures_dir"] / f"{dtm}_q_values_with_time.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_q_values_impl(q_agent, *, show: bool, deps: dict[str, Any]):
    plt = deps["plt"]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(q_agent.q_table)
    maze = q_agent.maze
    ax.set_title("Q-values")
    ax.set_xlabel("Actions")
    ax.set_xticks(list(range(maze.num_actions)))
    ax.set_xticklabels(maze.action_labels)
    ax.set_ylabel("States")
    ax.set_yticks(range(maze.num_states))
    ax.set_yticklabels(
        maze.state_labels or [f"State {state}" for state in range(maze.num_states)]
    )
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_q_history_impl(q_agent, *, show: bool, deps: dict[str, Any]):
    plt = deps["plt"]
    fig, ax = plt.subplots(figsize=(10, 6))
    maze = q_agent.maze
    labels = [
        f"S{state} {'Stay' if action == 0 else 'Leave'}"
        for state in range(maze.num_states)
        for action in range(maze.num_actions)
    ]

    for index, history in enumerate(q_agent.q_history):
        if history:
            ax.plot(history, label=labels[index])

    ax.set_title("Q-values over time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-value")
    ax.legend()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_returns_impl(q_agent, *, show: bool, deps: dict[str, Any]):
    plt = deps["plt"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(q_agent.returns)
    ax.set_title("Returns over episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cumulative_reward_impl(
    trajectory,
    *,
    save: bool,
    show: bool,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    deps["draw_cumulative_reward"](ax, trajectory)
    if save:
        deps["ensure_directories"]()
        filepath = deps["figures_dir"] / "cumulative_reward.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_residency_location_impl(
    trajectory,
    maze,
    *,
    save: bool,
    show: bool,
    deps: dict[str, Any],
):
    plt = deps["plt"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    deps["draw_residency_location"](ax, trajectory, maze)
    if save:
        deps["ensure_directories"]()
        filepath = deps["figures_dir"] / "residency_location.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_trajectory_stats_impl(
    trajectory,
    maze,
    source,
    *,
    save: bool,
    show: bool,
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

    deps["draw_cumulative_reward"](ax_reward, trajectory)
    deps["draw_residency_location"](ax_residency, trajectory, maze)
    if timing_summary is not None:
        deps["draw_patch_leave_probability_by_time"](ax_leave, timing_summary)
        deps["draw_patch_timing_summary"](ax_timing, timing_summary)

    fig.suptitle(f"Trajectory Overview: '{source}'", fontsize=16, fontweight="bold")

    if save:
        deps["ensure_directories"]()
        filepath = deps["figures_dir"] / f"trajectory_stats_{source}.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
