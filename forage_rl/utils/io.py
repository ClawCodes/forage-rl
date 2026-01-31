"""File I/O utilities for saving and loading experiment data."""

from pathlib import Path
from typing import Optional

import numpy as np

from forage_rl import Transition, TimedTransition, Trajectory
from forage_rl.config import LOGPROBS_DIR, TRAJECTORIES_DIR, ensure_directories


def save_trajectories(trajectory: Trajectory, algo_name: str, run_id: int) -> Path:
    """Save trajectory data to organized directory.

    Args:
        trajectory: Instance of Trajectory class.
        algo_name: Algorithm name (e.g., 'mbrl', 'q_learning')
        run_id: Run identifier

    Returns:
        Path to saved file
    """
    ensure_directories()
    filename = f"{algo_name}_trajectories_{run_id}.npy"
    filepath = TRAJECTORIES_DIR / filename
    np.save(filepath, trajectory.to_numpy())
    return filepath


def load_trajectories(algo_name: str, run_id: int) -> Trajectory:
    """Load trajectory data from organized directory.

    Automatically detects the transition type based on array shape:
    - 4 columns → Transition
    - 5 columns → TimedTransition

    Args:
        algo_name: Algorithm name (e.g., 'mbrl', 'q_learning')
        run_id: Run identifier

    Returns:
        Trajectory containing the loaded transitions
    """
    filename = f"{algo_name}_trajectories_{run_id}.npy"
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


def save_logprobs(data: np.ndarray, label: str, run_id: int) -> Path:
    """Save log probability data to organized directory.

    Args:
        data: Array of cumulative log-likelihoods
        label: Data label (e.g., 'mbrl_true', 'ql_false')
        run_id: Run identifier

    Returns:
        Path to saved file
    """
    ensure_directories()
    filename = f"{label}_log_likelihoods_{run_id}.npy"
    filepath = LOGPROBS_DIR / filename
    np.save(filepath, data)
    return filepath


def load_logprobs(label: str, run_id: int) -> np.ndarray:
    """Load log probability data from organized directory.

    Args:
        label: Data label (e.g., 'mbrl_true', 'ql_false')
        run_id: Run identifier

    Returns:
        Array of cumulative log-likelihoods
    """
    filename = f"{label}_log_likelihoods_{run_id}.npy"
    filepath = LOGPROBS_DIR / filename
    return np.load(filepath, allow_pickle=True)


def list_trajectory_files(algo_name: Optional[str] = None) -> list:
    """List all trajectory files in the data directory.

    Args:
        algo_name: Filter by algorithm name (optional)

    Returns:
        List of file paths
    """
    if not TRAJECTORIES_DIR.exists():
        return []

    pattern = f"{algo_name}_trajectories_*.npy" if algo_name else "*_trajectories_*.npy"
    return sorted(TRAJECTORIES_DIR.glob(pattern))


def list_logprob_files(label: Optional[str] = None) -> list:
    """List all log probability files in the data directory.

    Args:
        label: Filter by label (optional)

    Returns:
        List of file paths
    """
    if not LOGPROBS_DIR.exists():
        return []

    pattern = f"{label}_log_likelihoods_*.npy" if label else "*_log_likelihoods_*.npy"
    return sorted(LOGPROBS_DIR.glob(pattern))


def get_run_count(algo_name: str) -> int:
    """Get the number of trajectory files for an algorithm.

    Args:
        algo_name: Algorithm name

    Returns:
        Number of trajectory files
    """
    return len(list_trajectory_files(algo_name))
