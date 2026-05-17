"""Post-perturbation recovery metrics against benchmark patch residence times."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from forage_rl import RunDataset, Trajectory
from forage_rl.analysis.patch_timing import extract_decision_rows


def _normalize_exit_actions(exit_actions: int | Collection[int]) -> frozenset[int]:
    if isinstance(exit_actions, int):
        return frozenset({int(exit_actions)})
    normalized = frozenset(int(action) for action in exit_actions)
    if not normalized:
        raise ValueError("exit_actions must contain at least one action index.")
    return normalized


def _episode_signed_leave_deviations(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    normalized_exit_actions = _normalize_exit_actions(exit_actions)
    rows = extract_decision_rows(
        trajectory,
        patch_labels=patch_labels,
        resolved_states=resolved_states,
    )
    deviations = [
        float((row.time_spent + 1) - benchmark_prt_by_state[row.state])
        for row in rows
        if row.action in normalized_exit_actions and row.state in benchmark_prt_by_state
    ]
    return np.array(deviations, dtype=float)


def episode_prt_deviation_from_benchmark(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    resolved_states: list[int] | None = None,
) -> float:
    """Return the mean absolute leave-dwell deviation for one episode."""
    signed_deviations = _episode_signed_leave_deviations(
        trajectory,
        patch_labels=patch_labels,
        exit_actions=exit_actions,
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
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    resolved_states: list[int] | None = None,
) -> float:
    signed_deviations = _episode_signed_leave_deviations(
        trajectory,
        patch_labels=patch_labels,
        exit_actions=exit_actions,
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
    exit_actions: int | Collection[int],
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
                exit_actions=exit_actions,
                benchmark_prt_by_state=benchmark_prt_by_state,
                resolved_states=resolved_states,
            )
        )
    return np.array(values, dtype=float)


def signed_recovery_curve_for_episode_sequence(
    trajectories: list[Trajectory],
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
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
                exit_actions=exit_actions,
                benchmark_prt_by_state=benchmark_prt_by_state,
                resolved_states=resolved_states,
            )
        )
    return np.array(values, dtype=float)


def recovery_curve_for_run(
    run_dataset: RunDataset,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    resolved_states_by_episode: list[list[int] | None] | None = None,
) -> np.ndarray:
    """Return one mean absolute deviation value per episode in a run dataset."""
    return recovery_curve_for_episode_sequence(
        list(run_dataset),
        patch_labels=patch_labels,
        exit_actions=exit_actions,
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states_by_episode=resolved_states_by_episode,
    )


def signed_recovery_curve_for_run(
    run_dataset: RunDataset,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    resolved_states_by_episode: list[list[int] | None] | None = None,
) -> np.ndarray:
    """Return one mean signed deviation value per episode in a run dataset."""
    return signed_recovery_curve_for_episode_sequence(
        list(run_dataset),
        patch_labels=patch_labels,
        exit_actions=exit_actions,
        benchmark_prt_by_state=benchmark_prt_by_state,
        resolved_states_by_episode=resolved_states_by_episode,
    )


def _within_episode_leave_deviations(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    perturbation_timestep: int,
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    if perturbation_timestep < 0 or perturbation_timestep > len(trajectory):
        raise ValueError(
            "perturbation_timestep must lie within the trajectory transition range, "
            f"got {perturbation_timestep} for {len(trajectory)} transitions."
        )

    normalized_exit_actions = _normalize_exit_actions(exit_actions)
    rows = extract_decision_rows(
        trajectory,
        patch_labels=patch_labels,
        resolved_states=resolved_states,
    )
    deviations: list[float] = []
    for transition_index, row in enumerate(rows):
        if row.action not in normalized_exit_actions or row.state not in benchmark_prt_by_state:
            continue
        visit_start_index = transition_index - int(row.time_spent)
        if visit_start_index < perturbation_timestep:
            continue
        deviations.append(float((row.time_spent + 1) - benchmark_prt_by_state[row.state]))
    return np.array(deviations, dtype=float)


def within_episode_recovery_curve_for_trajectory(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    perturbation_timestep: int,
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    """Return absolute deviation for each complete post-perturbation leave event."""
    deviations = _within_episode_leave_deviations(
        trajectory,
        patch_labels=patch_labels,
        exit_actions=exit_actions,
        benchmark_prt_by_state=benchmark_prt_by_state,
        perturbation_timestep=perturbation_timestep,
        resolved_states=resolved_states,
    )
    return np.abs(deviations)


def within_episode_signed_recovery_curve_for_trajectory(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    perturbation_timestep: int,
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    """Return signed deviation for each complete post-perturbation leave event."""
    return _within_episode_leave_deviations(
        trajectory,
        patch_labels=patch_labels,
        exit_actions=exit_actions,
        benchmark_prt_by_state=benchmark_prt_by_state,
        perturbation_timestep=perturbation_timestep,
        resolved_states=resolved_states,
    )


def _within_episode_boundary_window_visit_deviations(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    perturbation_timestep: int,
    window: int | None = None,
    window_before: int | None = None,
    window_after: int | None = None,
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    resolved_window_before, resolved_window_after = _resolve_boundary_windows(
        window=window,
        window_before=window_before,
        window_after=window_after,
    )
    if perturbation_timestep < 0 or perturbation_timestep > len(trajectory):
        raise ValueError(
            "perturbation_timestep must lie within the trajectory transition range, "
            f"got {perturbation_timestep} for {len(trajectory)} transitions."
        )

    normalized_exit_actions = _normalize_exit_actions(exit_actions)
    rows = extract_decision_rows(
        trajectory,
        patch_labels=patch_labels,
        resolved_states=resolved_states,
    )
    values = np.full(
        resolved_window_before + resolved_window_after + 1,
        np.nan,
        dtype=float,
    )

    visit_start_index = 0
    for transition_index, row in enumerate(rows):
        next_begins_new_visit = (
            transition_index == len(rows) - 1
            or rows[transition_index + 1].time_spent == 0
        )
        if not next_begins_new_visit:
            continue

        visit_end_index = transition_index
        final_row = row
        if (
            final_row.action not in normalized_exit_actions
            or final_row.state not in benchmark_prt_by_state
        ):
            visit_start_index = transition_index + 1
            continue

        signed_deviation = float(
            (final_row.time_spent + 1) - benchmark_prt_by_state[final_row.state]
        )
        for visit_index in range(visit_start_index, visit_end_index + 1):
            relative_index = visit_index - perturbation_timestep
            if (
                relative_index < -resolved_window_before
                or relative_index > resolved_window_after
            ):
                continue
            values[relative_index + resolved_window_before] = signed_deviation

        visit_start_index = transition_index + 1

    return values


def _resolve_boundary_windows(
    *,
    window: int | None = None,
    window_before: int | None = None,
    window_after: int | None = None,
) -> tuple[int, int]:
    if window_before is None and window_after is None:
        if window is None:
            raise ValueError("Specify window or both window_before/window_after.")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        return window, window

    if window is not None:
        raise ValueError("Specify either window or window_before/window_after, not both.")
    if window_before is None or window_after is None:
        raise ValueError("window_before and window_after must be provided together.")
    if window_before <= 0 or window_after <= 0:
        raise ValueError(
            "window_before and window_after must both be > 0, "
            f"got {window_before} and {window_after}."
        )
    return window_before, window_after


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def boundary_window_segment_means(
    curve: np.ndarray,
    *,
    window: int | None = None,
    window_before: int | None = None,
    window_after: int | None = None,
) -> tuple[float, float]:
    """Return finite means for the pre- and post-perturbation window segments."""
    resolved_before, resolved_after = _resolve_boundary_windows(
        window=window,
        window_before=window_before,
        window_after=window_after,
    )
    expected_length = resolved_before + resolved_after + 1
    arr = np.asarray(curve, dtype=float)
    if arr.shape[0] != expected_length:
        raise ValueError(
            "boundary-window curve length must equal "
            "window_before + window_after + 1, "
            f"got len={arr.shape[0]} expected={expected_length}."
        )

    before_mean = _finite_mean(arr[:resolved_before])
    after_mean = _finite_mean(arr[resolved_before + 1 :])
    return before_mean, after_mean


def within_episode_boundary_window_recovery_curve_for_trajectory(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    perturbation_timestep: int,
    window: int | None = None,
    window_before: int | None = None,
    window_after: int | None = None,
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    """Return absolute visit-level deviation on a perturbation-centered step window.

    Each transition inside the window inherits the final leave deviation of the
    patch visit it belongs to. This produces a dense pre/post curve centered on
    the perturbation boundary instead of a sparse leave-event-only series.
    """
    deviations = _within_episode_boundary_window_visit_deviations(
        trajectory,
        patch_labels=patch_labels,
        exit_actions=exit_actions,
        benchmark_prt_by_state=benchmark_prt_by_state,
        perturbation_timestep=perturbation_timestep,
        window=window,
        window_before=window_before,
        window_after=window_after,
        resolved_states=resolved_states,
    )
    return np.abs(deviations)


def within_episode_signed_boundary_window_recovery_curve_for_trajectory(
    trajectory: Trajectory,
    *,
    patch_labels: dict[int, str],
    exit_actions: int | Collection[int],
    benchmark_prt_by_state: dict[int, int],
    perturbation_timestep: int,
    window: int | None = None,
    window_before: int | None = None,
    window_after: int | None = None,
    resolved_states: list[int] | None = None,
) -> np.ndarray:
    """Return signed visit-level deviation on a perturbation-centered step window."""
    return _within_episode_boundary_window_visit_deviations(
        trajectory,
        patch_labels=patch_labels,
        exit_actions=exit_actions,
        benchmark_prt_by_state=benchmark_prt_by_state,
        perturbation_timestep=perturbation_timestep,
        window=window,
        window_before=window_before,
        window_after=window_after,
        resolved_states=resolved_states,
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
