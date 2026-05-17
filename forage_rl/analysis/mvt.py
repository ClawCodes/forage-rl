"""Analytic Marginal Value Theorem helpers for simple-style fully observed mazes."""

from __future__ import annotations

import numpy as np

from forage_rl.environments import Maze


def _validate_simple_alternating_maze(maze: Maze) -> None:
    """Require the two-state alternating structure used by the true MVT benchmark."""
    if maze.num_states != 2:
        raise ValueError(
            "True MVT helpers currently support only the 2-state simple maze, "
            f"got num_states={maze.num_states}."
        )

    leave_action = maze.action_labels.index("leave")
    stay_action = maze.action_labels.index("stay")
    for state in range(maze.num_states):
        stay_transitions = maze.transition_distribution(state, stay_action)
        leave_transitions = maze.transition_distribution(state, leave_action)
        expected_other_state = 1 - state
        if stay_transitions != [(state, 1.0)]:
            raise ValueError(
                "True MVT helpers require deterministic self-transitions on stay, "
                f"got state={state} stay_transitions={stay_transitions!r}."
            )
        if leave_transitions != [(expected_other_state, 1.0)]:
            raise ValueError(
                "True MVT helpers require deterministic alternation on leave, "
                f"got state={state} leave_transitions={leave_transitions!r}."
            )


def reward_schedule_for_state(maze: Maze, state: int, horizon: int) -> np.ndarray:
    """Return expected exploit-step rewards for one patch visit.

    The returned array is indexed by one-based exploit steps:

    - element ``0`` is the expected reward on exploit step ``1``
    - exploit step ``k`` corresponds to ``time_spent == k - 1``

    The final leave action yields zero reward and is not included.
    """
    _validate_simple_alternating_maze(maze)
    return _reward_schedule_for_patch_state(maze, state, horizon)


def _reward_schedule_for_patch_state(maze: Maze, state: int, horizon: int) -> np.ndarray:
    """Return expected exploit-step rewards for one patch state."""
    if horizon <= 1:
        return np.array([], dtype=float)

    max_exploit_steps = horizon - 1
    return np.array(
        [
            maze.expected_stay_reward(state, time_spent)
            for time_spent in range(max_exploit_steps)
        ],
        dtype=float,
    )


def simple_true_mvt_optimal_exploit_steps(
    maze: Maze,
    horizon: int,
) -> dict[int, int]:
    """Return exact exploit-step counts maximizing cycle reward rate.

    For the simple alternating maze, one cycle is:

    - exploit ``k_upper`` times in the upper patch, then leave
    - exploit ``k_lower`` times in the lower patch, then leave

    The discrete-time MVT benchmark maximizes

    ``(G_upper(k_upper) + G_lower(k_lower)) / (k_upper + 1 + k_lower + 1)``

    over integer exploit counts ``k_upper >= 1`` and ``k_lower >= 1``.
    """
    _validate_simple_alternating_maze(maze)
    upper_schedule = reward_schedule_for_state(maze, 0, horizon)
    lower_schedule = reward_schedule_for_state(maze, 1, horizon)
    if upper_schedule.size == 0 or lower_schedule.size == 0:
        raise ValueError(
            f"horizon must permit at least one exploit step before leaving, got {horizon}."
        )

    upper_cumulative = np.cumsum(upper_schedule)
    lower_cumulative = np.cumsum(lower_schedule)

    best_rate = -np.inf
    best_counts: tuple[int, int] | None = None
    for upper_index, upper_gain in enumerate(upper_cumulative, start=1):
        for lower_index, lower_gain in enumerate(lower_cumulative, start=1):
            total_actions = upper_index + lower_index + 2
            reward_rate = float((upper_gain + lower_gain) / total_actions)
            candidate_counts = (upper_index, lower_index)
            if reward_rate > best_rate + 1e-12:
                best_rate = reward_rate
                best_counts = candidate_counts
                continue
            if np.isclose(reward_rate, best_rate):
                assert best_counts is not None
                if (
                    sum(candidate_counts) < sum(best_counts)
                    or (
                        sum(candidate_counts) == sum(best_counts)
                        and candidate_counts[0] < best_counts[0]
                    )
                ):
                    best_counts = candidate_counts

    assert best_counts is not None
    return {0: best_counts[0], 1: best_counts[1]}


def simple_true_mvt_optimal_prt(
    maze: Maze,
    horizon: int,
) -> dict[int, int]:
    """Return one-based patch residence times for the simple true MVT benchmark."""
    exploit_steps = simple_true_mvt_optimal_exploit_steps(maze, horizon)
    return {state: exploit_steps[state] + 1 for state in exploit_steps}


def _deterministic_leave_successor(maze: Maze, state: int) -> int:
    leave_action = maze.action_labels.index("leave")
    transitions = maze.transition_distribution(state, leave_action)
    if transitions != [(transitions[0][0], 1.0)]:
        raise ValueError(
            "True MVT helpers for one-way mazes require deterministic leave transitions, "
            f"got state={state} transitions={transitions!r}."
        )
    return int(transitions[0][0])


def _patch_states_with_stay_and_leave(maze: Maze) -> tuple[int, int]:
    stay_action = maze.action_labels.index("stay")
    leave_action = maze.action_labels.index("leave")
    patch_states = [
        state
        for state in range(maze.num_states)
        if stay_action in maze.valid_actions(state) and leave_action in maze.valid_actions(state)
    ]
    if len(patch_states) != 2:
        raise ValueError(
            "One-way true MVT helpers require exactly two patch states with stay/leave, "
            f"got {patch_states!r}."
        )
    return tuple(sorted(patch_states))


def _leave_path_length_between_patch_states(
    maze: Maze,
    *,
    start_patch: int,
    target_patch: int,
) -> int:
    stay_action = maze.action_labels.index("stay")
    current_state = start_patch
    visited: set[int] = set()
    travel_steps = 0

    while True:
        next_state = _deterministic_leave_successor(maze, current_state)
        travel_steps += 1
        if next_state == target_patch:
            return travel_steps
        if next_state in visited:
            raise ValueError("Detected a loop while resolving one-way travel time.")
        if stay_action in maze.valid_actions(next_state):
            raise ValueError(
                "Encountered an intermediate patch state before reaching the target patch, "
                f"got next_state={next_state} target_patch={target_patch}."
            )
        visited.add(next_state)
        current_state = next_state


def simple_one_way_true_mvt_optimal_prt(
    maze: Maze,
    horizon: int,
) -> dict[int, int]:
    """Return exact patch residence times for the one-way simple maze."""
    upper_patch, lower_patch = _patch_states_with_stay_and_leave(maze)
    upper_schedule = _reward_schedule_for_patch_state(maze, upper_patch, horizon)
    lower_schedule = _reward_schedule_for_patch_state(maze, lower_patch, horizon)
    if upper_schedule.size == 0 or lower_schedule.size == 0:
        raise ValueError(
            f"horizon must permit at least one exploit step before leaving, got {horizon}."
        )

    travel_upper_to_lower = _leave_path_length_between_patch_states(
        maze,
        start_patch=upper_patch,
        target_patch=lower_patch,
    )
    travel_lower_to_upper = _leave_path_length_between_patch_states(
        maze,
        start_patch=lower_patch,
        target_patch=upper_patch,
    )
    upper_cumulative = np.cumsum(upper_schedule)
    lower_cumulative = np.cumsum(lower_schedule)

    best_rate = -np.inf
    best_counts: tuple[int, int] | None = None
    for upper_index, upper_gain in enumerate(upper_cumulative, start=1):
        for lower_index, lower_gain in enumerate(lower_cumulative, start=1):
            total_actions = (
                upper_index
                + lower_index
                + travel_upper_to_lower
                + travel_lower_to_upper
            )
            reward_rate = float((upper_gain + lower_gain) / total_actions)
            candidate_counts = (upper_index, lower_index)
            if reward_rate > best_rate + 1e-12:
                best_rate = reward_rate
                best_counts = candidate_counts
                continue
            if np.isclose(reward_rate, best_rate):
                assert best_counts is not None
                if (
                    sum(candidate_counts) < sum(best_counts)
                    or (
                        sum(candidate_counts) == sum(best_counts)
                        and candidate_counts[0] < best_counts[0]
                    )
                ):
                    best_counts = candidate_counts

    assert best_counts is not None
    return {upper_patch: best_counts[0] + 1, lower_patch: best_counts[1] + 1}
