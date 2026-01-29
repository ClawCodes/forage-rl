"""Visualization functions for model comparison and Q-value analysis."""

from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.config import FIGURES_DIR, ensure_directories
from forage_rl.utils import get_run_count, load_logprobs


def plot_model_comparison(
    num_datasets: Optional[int] = None, save: bool = False, show: bool = True
):
    """Plot bar chart comparing model classification accuracy.

    Args:
        num_datasets: Number of datasets to analyze
        save: Whether to save the figure
        show: Whether to display the figure
    """
    num_datasets = num_datasets or min(
        get_run_count("mbrl"),
        get_run_count("q_learning"),
    )

    if num_datasets == 0:
        print("No log probability files found. Run model_inference.py first.")
        return

    mb_accuracies = []
    ql_accuracies = []

    for i in range(num_datasets):
        # MBRL-generated data: MBRL should have higher likelihood
        mb_logprobs = load_logprobs("mbrl_true", i)
        ql_logprobs = load_logprobs("ql_false", i)

        if mb_logprobs[-1] > ql_logprobs[-1]:
            mb_accuracies.append(1)
        else:
            mb_accuracies.append(0)

        # Q-learning-generated data: Q-learning should have higher likelihood
        mb_logprobs_ql = load_logprobs("mbrl_false", i)
        ql_logprobs_ql = load_logprobs("ql_true", i)

        if ql_logprobs_ql[-1] > mb_logprobs_ql[-1]:
            ql_accuracies.append(1)
        else:
            ql_accuracies.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    accuracies = [np.mean(mb_accuracies), np.mean(ql_accuracies)]
    bars = ax.bar(
        ["Model-Based RL", "Q-Learning"], accuracies, color=["#3498db", "#e74c3c"]
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Classification Accuracy", fontsize=14)
    ax.set_title("Model Comparison: Classification Accuracy", fontsize=16)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Add chance line
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
    ax.legend()

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "model_comparison.png"
        plt.savefig(filepath, dpi=150)
        print(f"Saved to {filepath}")

    if show:
        plt.show()

    return fig


def plot_cumulative_sum_accuracy(
    num_datasets: Optional[int] = None, save: bool = False, show: bool = True
):
    """Plot accuracy as a function of observed transitions.

    Shows how classification accuracy improves as more transitions are observed.

    Args:
        num_datasets: Number of datasets to analyze
        save: Whether to save the figure
        show: Whether to display the figure
    """
    num_datasets = num_datasets or min(
        get_run_count("mbrl"),
        get_run_count("q_learning"),
    )

    if num_datasets == 0:
        print("No log probability files found. Run model_inference.py first.")
        return

    accuracies = []

    for j in range(num_datasets):
        # MBRL-generated data
        mb_cumsum = load_logprobs("mbrl_true", j)
        ql_cumsum = load_logprobs("ql_false", j)

        accuracy_mb = np.zeros(len(mb_cumsum))
        for i in range(len(mb_cumsum)):
            if np.isclose(mb_cumsum[i], ql_cumsum[i]):
                accuracy_mb[i] = 0.5  # Tie
            elif mb_cumsum[i] > ql_cumsum[i]:
                accuracy_mb[i] = 1
            else:
                accuracy_mb[i] = 0
        accuracies.append(accuracy_mb)

        # Q-learning-generated data
        mb_cumsum_ql = load_logprobs("mbrl_false", j)
        ql_cumsum_ql = load_logprobs("ql_true", j)

        accuracy_ql = np.zeros(len(mb_cumsum_ql))
        for i in range(len(mb_cumsum_ql)):
            if np.isclose(mb_cumsum_ql[i], ql_cumsum_ql[i]):
                accuracy_ql[i] = 0.5  # Tie
            elif ql_cumsum_ql[i] > mb_cumsum_ql[i]:
                accuracy_ql[i] = 1
            else:
                accuracy_ql[i] = 0
        accuracies.append(accuracy_ql)

    # Compute average accuracy
    avg_accuracy = np.mean(accuracies, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(avg_accuracy, linewidth=3, color="#2ecc71")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")

    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Number of Observed Transitions", fontsize=16)
    ax.set_ylabel("Prediction Accuracy", fontsize=16)
    ax.set_title("Classification Accuracy vs. Sample Size", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()

    if save:
        ensure_directories()
        filepath = FIGURES_DIR / "cumulative_accuracy.png"
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

    max_time = min(max_time_to_display, q_table.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmin = np.min(q_table[:, :max_time, :])
    vmax = np.max(q_table[:, :max_time, :])

    for i, action in enumerate(range(maze.num_actions)):
        im = axes[i].imshow(q_table[:, :max_time, action], vmin=vmin, vmax=vmax)
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
                value = q_table[y, x, action]
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
    im = ax.imshow(q_agent.q_table)
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


if __name__ == "__main__":
    print("Plotting model comparison results...")
    plot_model_comparison(save=True)
    plot_cumulative_sum_accuracy(save=True)
