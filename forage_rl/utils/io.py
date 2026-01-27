"""File I/O utilities for saving and loading experiment data."""

import numpy as np
from pathlib import Path

from forage_rl.config import TRAJECTORIES_DIR, LOGPROBS_DIR, ensure_directories


def save_trajectories(data: list, algo_name: str, run_id: int) -> Path:
    """Save trajectory data to organized directory.

    Args:
        data: List of transition tuples
        algo_name: Algorithm name (e.g., 'mbrl', 'q_learning')
        run_id: Run identifier

    Returns:
        Path to saved file
    """
    ensure_directories()
    filename = f"{algo_name}_trajectories_{run_id}.npy"
    filepath = TRAJECTORIES_DIR / filename
    np.save(filepath, data)
    return filepath


def load_trajectories(algo_name: str, run_id: int) -> np.ndarray:
    """Load trajectory data from organized directory.

    Args:
        algo_name: Algorithm name (e.g., 'mbrl', 'q_learning')
        run_id: Run identifier

    Returns:
        Array of transition tuples
    """
    filename = f"{algo_name}_trajectories_{run_id}.npy"
    filepath = TRAJECTORIES_DIR / filename
    return np.load(filepath, allow_pickle=True)


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


def list_trajectory_files(algo_name: str = None) -> list:
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


def list_logprob_files(label: str = None) -> list:
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
