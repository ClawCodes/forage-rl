"""Post-perturbation recovery metrics against benchmark patch residence times."""

from __future__ import annotations

import numpy as np

from forage_rl import RunDataset, Trajectory
from forage_rl.analysis.patch_timing import extract_decision_rows


def _episode_signed_leave_deviations(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    leave_action: int,
    benchmark_prt_by_state: dict[int, int],
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    rows = extract_decision_rows(
        trajectory,
        patch_labels=patch_labels,
        resolved_states=resolved_states,
    )
    deviations = [
        float((row.time_spent + 1) - benchmark_prt_by_state[row.state])
        for row in rows
        if row.action == leave_action and row.state in benchmark_prt_by_state
    ]
    return np.array(deviations, dtype=float)


def episode_prt_deviation_from_benchmark(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    leave_action: int,
    benchmark_prt_by_state: dict[int, int],
    resolved_states: list[int] | None = None,
) -> float:
    """Return the mean absolute leave-dwell deviation for one episode."""
    signed_deviations = _episode_signed_leave_deviations(
        trajectory,
        patch_labels=patch_labels,
        leave_action=leave_action,
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states=resolved_states,
    )
    if signed_deviations.size == 0:
        return float("nan")
    return float(np.mean(np.abs(signed_deviations)))


def _episode_signed_deviation_from_benchmark(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    leave_action: int,
    benchmark_prt_by_state: dict[int, int],
    resolved_states: list[int] | None = None,
) -> float:
    signed_deviations = _episode_signed_leave_deviations(
        trajectory,
        patch_labels=patch_labels,
        leave_action=leave_action,
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states=resolved_states,
    )
    if signed_deviations.size == 0:
        return float("nan")
    return float(np.mean(signed_deviations))


def recovery_curve_for_episode_sequence(
    trajectories: list[Trajectory],
    *,
    patch_labels: dict[int, str],
    leave_action: int,
    benchmark_prt_by_state: dict[int, int],
    resolved_states_by_episode: list[list[int] | None] | None = None,
) -> np.ndarray:
    """Return one mean absolute deviation value per episode trajectory."""
    values = []
    for episode_index, trajectory in enumerate(trajectories):
        resolved_states = (
            None
            if resolved_states_by_episode is None
            else resolved_states_by_episode[episode_index]
        )
        values.append(
            episode_prt_deviation_from_benchmark(
                trajectory,
                patch_labels=patch_labels,
                leave_action=leave_action,
                benchmark_prt_by_state=benchmark_prt_by_state,
                resolved_states=resolved_states,
            )
        )
    return np.array(values, dtype=float)


def signed_recovery_curve_for_episode_sequence(
    trajectories: list[Trajectory],
    *,
    patch_labels: dict[int, str],
    leave_action: int,
    benchmark_prt_by_state: dict[int, int],
    resolved_states_by_episode: list[list[int] | None] | None = None,
) -> np.ndarray:
    """Return one mean signed deviation value per episode trajectory."""
    values = []
    for episode_index, trajectory in enumerate(trajectories):
        resolved_states = (
            None
            if resolved_states_by_episode is None
            else resolved_states_by_episode[episode_index]
        )
        values.append(
            _episode_signed_deviation_from_benchmark(
                trajectory,
                patch_labels=patch_labels,
                leave_action=leave_action,
                benchmark_prt_by_state=benchmark_prt_by_state,
                resolved_states=resolved_states,
            )
        )
    return np.array(values, dtype=float)


def recovery_curve_for_run(
    run_dataset: RunDataset,
    *,
    patch_labels: dict[int, str],
    leave_action: int,
    benchmark_prt_by_state: dict[int, int],
    resolved_states_by_episode: list[list[int] | None] | None = None,
) -> np.ndarray:
    """Return one mean absolute deviation value per episode in a run dataset."""
    return recovery_curve_for_episode_sequence(
        list(run_dataset),
        patch_labels=patch_labels,
        leave_action=leave_action,
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states_by_episode=resolved_states_by_episode,
    )


def signed_recovery_curve_for_run(
    run_dataset: RunDataset,
    *,
    patch_labels: dict[int, str],
    leave_action: int,
    benchmark_prt_by_state: dict[int, int],
    resolved_states_by_episode: list[list[int] | None] | None = None,
) -> np.ndarray:
    """Return one mean signed deviation value per episode in a run dataset."""
    return signed_recovery_curve_for_episode_sequence(
        list(run_dataset),
        patch_labels=patch_labels,
        leave_action=leave_action,
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states_by_episode=resolved_states_by_episode,
    )


def recovery_auc(curve: np.ndarray, window: int) -> float:
    """Return the raw discrete area under the recovery curve over a fixed window."""
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    finite_window = np.asarray(curve[:window], dtype=float)
    finite_window = finite_window[np.isfinite(finite_window)]
    if finite_window.size == 0:
        return float("nan")
    return float(np.sum(finite_window))
