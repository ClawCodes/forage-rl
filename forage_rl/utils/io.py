"""File I/O utilities for saving and loading experiment data."""

from pathlib import Path
from typing import Optional

import numpy as np

from forage_rl import ObservedTimedTransition, TimedTransition, Trajectory, Transition
from forage_rl.config import LOGPROBS_DIR, TRAJECTORIES_DIR, ensure_directories


def _env_prefix(env_key: str | None) -> str:
    return f"env_{env_key}__" if env_key is not None else ""


def save_trajectories(
    trajectory: Trajectory,
    algo_name: str,
    run_id: int,
    env_key: str | None = None,
) -> Path:
    """Save trajectory data to organized directory.

    Args:
        trajectory: Instance of Trajectory class.
        algo_name: Algorithm name (e.g., 'mbrl', 'q_learning')
        run_id: Run identifier
        env_key: Optional environment key prefix used to namespace saved artifacts.

    Returns:
        Path to saved file

    Raises:
        ValueError: If the trajectory is empty.
    """
    if len(trajectory) == 0:
        raise ValueError(
            "Empty trajectories are not supported. "
            "Generate at least one transition before saving."
        )

    ensure_directories()
    filename = f"{_env_prefix(env_key)}{algo_name}_trajectories_{run_id}.npy"
    filepath = TRAJECTORIES_DIR / filename
    np.save(filepath, trajectory.to_numpy())
    return filepath


def load_trajectories(
    algo_name: str,
    run_id: int,
    env_key: str | None = None,
) -> Trajectory:
    """Load trajectory data from organized directory.

    Supported transition formats:
    - 6 columns -> TimedTransition with terminal markers
    - 8 columns -> ObservedTimedTransition with hidden-state metadata

    Legacy 4-column and 5-column trajectory files are rejected because replay
    now requires terminal-aware timed transitions.

    Args:
        algo_name: Algorithm name (e.g., 'mbrl', 'q_learning')
        run_id: Run identifier
        env_key: Optional environment key prefix used to namespace saved artifacts.

    Returns:
        Trajectory containing the loaded transitions

    Raises:
        ValueError: If the saved payload is empty or not a 2D transition array.
    """
    filename = f"{_env_prefix(env_key)}{algo_name}_trajectories_{run_id}.npy"
    filepath = TRAJECTORIES_DIR / filename
    arr = np.load(filepath, allow_pickle=True)
    if arr.size == 0:
        raise ValueError(
            "Empty trajectory files are not supported. "
            "Regenerate trajectories with num_episodes > 0."
        )
    if arr.ndim != 2:
        raise ValueError(
            "Trajectory files must contain a 2D transition array. "
            f"Got array with shape {arr.shape}."
        )

    # Auto-detect transition type from column count
    num_cols = arr.shape[1]
    timed_cols = len(TimedTransition.model_fields)
    observed_timed_cols = len(ObservedTimedTransition.model_fields)
    if num_cols == len(Transition.model_fields):
        raise ValueError(
            "Legacy 4-column trajectory files are no longer supported. "
            "Regenerate trajectories with terminal-aware timed transitions."
        )
    elif num_cols == timed_cols:
        transition_cls = TimedTransition
        terminal_flags_present = True
    elif num_cols == observed_timed_cols:
        transition_cls = ObservedTimedTransition
        terminal_flags_present = True
    elif num_cols == timed_cols - 1:
        raise ValueError(
            "Legacy 5-column timed trajectory files are no longer supported. "
            "Regenerate trajectories with terminal-aware timed transitions."
        )
    elif num_cols == timed_cols + 1:
        raise ValueError(
            "Observed-state trajectory files are no longer supported. "
            "Regenerate deep-agent artifacts with the simplified DQN/RDQN pipeline."
        )
    else:
        raise ValueError(f"Unknown transition format with {num_cols} columns")

    trajectory = Trajectory.from_numpy(arr, transition_cls)
    trajectory.terminal_flags_present = terminal_flags_present
    return trajectory


def save_logprobs(
    data: np.ndarray,
    label: str,
    run_id: int,
    env_key: str | None = None,
) -> Path:
    """Save log probability data to organized directory.

    Args:
        data: Array of cumulative log-likelihoods
        label: Data label (e.g., 'mbrl_true', 'ql_false')
        run_id: Run identifier
        env_key: Optional environment key prefix used to namespace saved artifacts.

    Returns:
        Path to saved file
    """
    ensure_directories()
    filename = f"{_env_prefix(env_key)}{label}_log_likelihoods_{run_id}.npy"
    filepath = LOGPROBS_DIR / filename
    np.save(filepath, data)
    return filepath


def load_logprobs(label: str, run_id: int, env_key: str | None = None) -> np.ndarray:
    """Load log probability data from organized directory.

    Args:
        label: Data label (e.g., 'mbrl_true', 'ql_false')
        run_id: Run identifier
        env_key: Optional environment key prefix used to namespace saved artifacts.

    Returns:
        Array of cumulative log-likelihoods
    """
    filename = f"{_env_prefix(env_key)}{label}_log_likelihoods_{run_id}.npy"
    filepath = LOGPROBS_DIR / filename
    return np.load(filepath, allow_pickle=True)


def list_trajectory_files(
    algo_name: Optional[str] = None,
    env_key: str | None = None,
) -> list:
    """List saved trajectory files.

    Args:
        algo_name: Filter by algorithm name.
        env_key: Optional environment key prefix used to namespace saved artifacts.

    Returns:
        Sorted list of matching file paths.
    """
    if not TRAJECTORIES_DIR.exists():
        return []

    prefix = _env_prefix(env_key)
    pattern = (
        f"{prefix}{algo_name}_trajectories_*.npy"
        if algo_name
        else f"{prefix}*_trajectories_*.npy"
    )
    return sorted(TRAJECTORIES_DIR.glob(pattern))


def list_logprob_files(
    label: Optional[str] = None,
    env_key: str | None = None,
) -> list:
    """List saved log-probability files.

    Args:
        label: Filter by label.
        env_key: Optional environment key prefix used to namespace saved artifacts.

    Returns:
        Sorted list of matching file paths.
    """
    if not LOGPROBS_DIR.exists():
        return []

    prefix = _env_prefix(env_key)
    pattern = (
        f"{prefix}{label}_log_likelihoods_*.npy"
        if label
        else f"{prefix}*_log_likelihoods_*.npy"
    )
    return sorted(LOGPROBS_DIR.glob(pattern))


def _extract_run_id(path: Path, prefix: str) -> int | None:
    """Extract a numeric run id from a saved data filename."""
    stem = path.stem
    if not stem.startswith(prefix):
        return None

    run_id = stem.removeprefix(prefix)
    return int(run_id) if run_id.isdigit() else None


def get_trajectory_run_ids(algo_name: str, env_key: str | None = None) -> list[int]:
    """Return sorted saved trajectory run ids for an algorithm."""
    prefix = f"{_env_prefix(env_key)}{algo_name}_trajectories_"
    run_ids = [
        run_id
        for path in list_trajectory_files(algo_name, env_key=env_key)
        if (run_id := _extract_run_id(path, prefix)) is not None
    ]
    return sorted(run_ids)


def get_logprob_run_ids(label: str, env_key: str | None = None) -> list[int]:
    """Return sorted saved log-probability run ids for a label."""
    prefix = f"{_env_prefix(env_key)}{label}_log_likelihoods_"
    run_ids = [
        run_id
        for path in list_logprob_files(label, env_key=env_key)
        if (run_id := _extract_run_id(path, prefix)) is not None
    ]
    return sorted(run_ids)
