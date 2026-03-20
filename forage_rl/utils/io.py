"""File I/O utilities for saving and loading experiment data."""

from pathlib import Path
from typing import Optional

import numpy as np

from forage_rl import Transition, TimedTransition, Trajectory
from forage_rl.agents.registry import Agent
from forage_rl.config import LOGPROBS_DIR, TRAJECTORIES_DIR, ensure_directories


def _obs_tag(observable: bool) -> str:
    return "FO" if observable else "PO"


def save_trajectories(
    trajectory: Trajectory,
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
) -> Path:
    """Save trajectory data to organized directory.

    Args:
        trajectory: Instance of Trajectory class.
        agent: Agent that produced the trajectory
        run_id: Run identifier
        maze_name: Maze name (e.g., 'simple', 'full')
        observable: True for fully observable (FO), False for partially observable (PO)

    Returns:
        Path to saved file
    """
    ensure_directories()
    filename = (
        f"{maze_name}_{_obs_tag(observable)}_{agent.value}_trajectories_{run_id}.npy"
    )
    filepath = TRAJECTORIES_DIR / filename
    np.save(filepath, trajectory.to_numpy())
    return filepath


def load_trajectories(
    agent: Agent, run_id: int, maze_name: str, observable: bool = True
) -> Trajectory:
    """Load trajectory data from organized directory.

    Automatically detects the transition type based on array shape:
    - 4 columns → Transition
    - 5 columns → TimedTransition

    Args:
        agent: Agent name that produced the trajectory (e.g., 'mbrl', 'q_learning')
        run_id: Run identifier
        maze_name: Maze name (e.g., 'simple', 'full')
        observable: True for fully observable (FO), False for partially observable (PO)

    Returns:
        Trajectory containing the loaded transitions
    """
    filename = (
        f"{maze_name}_{_obs_tag(observable)}_{agent.value}_trajectories_{run_id}.npy"
    )
    filepath = TRAJECTORIES_DIR / filename
    arr = np.load(filepath, allow_pickle=True)

    # Auto-detect transition type from column count
    num_cols = arr.shape[1] if arr.ndim > 1 else len(arr[0])
    if num_cols == len(Transition.model_fields):
        transition_cls = Transition
    elif num_cols == len(TimedTransition.model_fields):
        transition_cls = TimedTransition
    else:
        raise ValueError(f"Unknown transition format with {num_cols} columns")

    return Trajectory.from_numpy(arr, transition_cls)


def save_logprobs(
    data: np.ndarray,
    source: Agent,
    evaluator: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
) -> Path:
    """Save log probability data to organized directory.

    Args:
        data: Array of cumulative log-likelihoods
        source: Agent that generated the trajectory
        evaluator: Agent used to evaluate the trajectory
        run_id: Run identifier
        maze_name: Maze name (e.g., 'simple', 'full')
        observable: True for fully observable (FO), False for partially observable (PO)

    Returns:
        Path to saved file
    """
    ensure_directories()
    label = f"source_{source.value}_eval_{evaluator.value}"
    filename = (
        f"{maze_name}_{_obs_tag(observable)}_{label}_log_likelihoods_{run_id}.npy"
    )
    filepath = LOGPROBS_DIR / filename
    np.save(filepath, data)
    return filepath


def load_logprobs(
    source: Agent,
    evaluator: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
) -> np.ndarray:
    """Load log probability data from organized directory.

    Args:
        source: Agent that generated the trajectory
        evaluator: Agent used to evaluate the trajectory
        run_id: Run identifier
        maze_name: Maze name (e.g., 'simple', 'full')
        observable: True for fully observable (FO), False for partially observable (PO)

    Returns:
        Array of cumulative log-likelihoods
    """
    label = f"source_{source.value}_eval_{evaluator.value}"
    filename = (
        f"{maze_name}_{_obs_tag(observable)}_{label}_log_likelihoods_{run_id}.npy"
    )
    filepath = LOGPROBS_DIR / filename
    return np.load(filepath, allow_pickle=True)


def list_trajectory_files(
    agent: Optional[Agent] = None,
    maze_name: Optional[str] = None,
    observable: Optional[bool] = None,
) -> list:
    """List all trajectory files in the data directory.

    Args:
        agent: Filter by agent (optional)
        maze_name: Filter by maze name (optional)
        observable: Filter by observability — True for FO, False for PO, None for both

    Returns:
        List of file paths
    """
    if not TRAJECTORIES_DIR.exists():
        return []

    maze_part = maze_name or "*"
    obs_part = _obs_tag(observable) if observable is not None else "*"
    agent_part = agent.value if agent is not None else "*"
    pattern = f"{maze_part}_{obs_part}_{agent_part}_trajectories_*.npy"
    return sorted(TRAJECTORIES_DIR.glob(pattern))


def list_logprob_files(
    label: Optional[str] = None,
    maze_name: Optional[str] = None,
    observable: Optional[bool] = None,
) -> list:
    """List all log probability files in the data directory.

    Args:
        label: Filter by label (optional)
        maze_name: Filter by maze name (optional)
        observable: Filter by observability — True for FO, False for PO, None for both

    Returns:
        List of file paths
    """
    if not LOGPROBS_DIR.exists():
        return []

    maze_part = maze_name or "*"
    obs_part = _obs_tag(observable) if observable is not None else "*"
    label_part = label or "*"
    pattern = f"{maze_part}_{obs_part}_{label_part}_log_likelihoods_*.npy"
    return sorted(LOGPROBS_DIR.glob(pattern))


def get_run_count(agent: Agent, maze_name: str, observable: bool = True) -> int:
    """Get the number of trajectory files for an agent and maze.

    Args:
        agent: Agent whose trajectory files to count
        maze_name: Maze name
        observable: True for fully observable (FO), False for partially observable (PO)

    Returns:
        Number of trajectory files
    """
    return len(list_trajectory_files(agent, maze_name, observable))
