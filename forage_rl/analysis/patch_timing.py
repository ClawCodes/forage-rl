"""Shared patch-timing analysis helpers for trajectory summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from forage_rl.agents.value_iteration import ValueIterationSolver
from forage_rl.environments import Maze, load_builtin_maze_spec


@dataclass(frozen=True)
class DecisionRow:
    """One policy decision augmented with reconstructed local context."""

    state: int
    patch_label: str
    time_spent: int
    prev_reward: float
    zero_streak: int
    action: int


@dataclass(frozen=True)
class CurveSummary:
    """Aggregate curve statistics across matched trajectories or runs."""

    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def observation_group_patch_labels(maze_name: str) -> dict[int, str]:
    """Return one patch label for each observation group in a built-in maze."""
    maze_spec = load_builtin_maze_spec(maze_name)
    labels_by_group: dict[int, set[str]] = {}
    for state in maze_spec.states:
        labels_by_group.setdefault(state.observation_group, set()).add(state.label)

    patch_labels: dict[int, str] = {}
    for observation_group, labels in labels_by_group.items():
        if len(labels) != 1:
            raise ValueError(
                "Patch-timing probes require one patch label per observation group, "
                f"got observation_group={observation_group} labels={sorted(labels)}."
            )
        patch_labels[observation_group] = next(iter(labels))
    return patch_labels


def observation_group_state_ids(maze: Maze) -> dict[int, tuple[int, ...]]:
    """Return true-state ids grouped by observation id."""
    groups: dict[int, list[int]] = {}
    for state_spec in maze.maze_spec.states:
        groups.setdefault(state_spec.observation_group, []).append(state_spec.id)
    return {
        observation_group: tuple(sorted(state_ids))
        for observation_group, state_ids in groups.items()
    }


def mvt_optimal_dwell_by_state(
    *,
    maze_name: str,
    horizon: int,
) -> dict[int, int]:
    """Return earliest optimal leave dwell by true state from the full MDP policy."""
    maze = Maze(load_builtin_maze_spec(maze_name), seed=0, horizon=horizon)
    _, policy = ValueIterationSolver(maze).solve(verbose=False)
    leave_action = maze.action_labels.index("leave")

    optimal_dwell_by_state: dict[int, int] = {}
    for state in range(maze.num_states):
        leave_times = np.flatnonzero(policy[state] == leave_action)
        optimal_dwell_by_state[state] = (
            int(leave_times[0]) + 1 if leave_times.size > 0 else horizon
        )
    return optimal_dwell_by_state


def _normalize_probabilities(probabilities: dict[int, float]) -> dict[int, float]:
    """Normalize a sparse probability map, falling back to a uniform distribution."""
    if not probabilities:
        return {}

    clipped = {
        state: max(float(probability), 0.0)
        for state, probability in probabilities.items()
    }
    total = sum(clipped.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(clipped))
        return {state: uniform for state in clipped}
    return {state: probability / total for state, probability in clipped.items()}


def _binary_reward_likelihood(
    maze: Maze,
    *,
    state: int,
    time_spent: int,
    reward: float,
) -> float:
    """Return the Bernoulli likelihood of an observed stay reward."""
    reward_prob = maze.expected_stay_reward(state, time_spent)
    return reward_prob if reward > 0.0 else 1.0 - reward_prob


def infer_hidden_states_for_trajectory(
    trajectory,
    *,
    maze: Maze,
) -> list[int]:
    """Infer visit-level hidden states for PO trajectories from known patch dynamics."""
    observation_group_to_states = observation_group_state_ids(maze)
    observed_group_ids = set(observation_group_to_states)
    transitions = trajectory.transitions
    if not transitions:
        return []

    if any(
        int(transition.state) not in observed_group_ids
        or int(transition.next_state) not in observed_group_ids
        for transition in transitions
    ):
        return [int(transition.state) for transition in transitions]

    state_to_observation_group = {
        state_spec.id: state_spec.observation_group
        for state_spec in maze.maze_spec.states
    }
    leave_action = maze.action_labels.index("leave")
    inferred_states = [maze.initial_state] * len(transitions)

    initial_observation = int(transitions[0].state)
    if state_to_observation_group.get(maze.initial_state) == initial_observation:
        posterior = {
            state: 1.0 if state == maze.initial_state else 0.0
            for state in observation_group_to_states[initial_observation]
        }
    else:
        posterior = {
            state: 1.0 for state in observation_group_to_states[initial_observation]
        }
    posterior = _normalize_probabilities(posterior)

    visit_start = 0
    while visit_start < len(transitions):
        current_observation = int(transitions[visit_start].state)
        candidate_states = observation_group_to_states[current_observation]
        posterior = _normalize_probabilities(
            {state: posterior.get(state, 0.0) for state in candidate_states}
        )

        visit_end = visit_start
        while (
            visit_end + 1 < len(transitions)
            and int(transitions[visit_end].next_state) == current_observation
        ):
            visit_end += 1

        weighted_posterior = dict(posterior)
        for step in range(visit_start, visit_end + 1):
            transition = transitions[step]
            if (
                int(transition.action) == leave_action
                or int(transition.next_state) != current_observation
            ):
                continue
            for state in weighted_posterior:
                weighted_posterior[state] *= _binary_reward_likelihood(
                    maze,
                    state=state,
                    time_spent=int(getattr(transition, "time_spent", 0)),
                    reward=float(transition.reward),
                )
        posterior = _normalize_probabilities(weighted_posterior)

        inferred_state = max(sorted(posterior), key=lambda state: posterior[state])
        for step in range(visit_start, visit_end + 1):
            inferred_states[step] = inferred_state

        final_transition = transitions[visit_end]
        if visit_end == len(transitions) - 1:
            break
        if int(final_transition.next_state) == current_observation:
            break

        next_observation = int(final_transition.next_state)
        next_candidates = observation_group_to_states[next_observation]
        next_posterior = {state: 0.0 for state in next_candidates}
        for state, probability in posterior.items():
            for next_state, transition_probability in maze.transition_distribution(
                state,
                leave_action,
            ):
                if next_state in next_posterior:
                    next_posterior[next_state] += probability * transition_probability
        posterior = _normalize_probabilities(next_posterior)
        visit_start = visit_end + 1

    return inferred_states


def extract_decision_rows(
    trajectory,
    *,
    patch_labels: dict[int, str],
    resolved_states: list[int] | None = None,
) -> list[DecisionRow]:
    """Reconstruct per-decision local context from one saved episode."""
    rows: list[DecisionRow] = []
    prev_reward = 0.0
    zero_streak = 0

    for index, transition in enumerate(trajectory.transitions):
        observed_state = int(transition.state)
        patch_label = patch_labels[observed_state]
        state = (
            int(resolved_states[index])
            if resolved_states is not None
            else observed_state
        )
        time_spent = int(getattr(transition, "time_spent", 0))
        action = int(transition.action)
        reward = float(transition.reward)
        next_state = int(transition.next_state)

        rows.append(
            DecisionRow(
                state=state,
                patch_label=patch_label,
                time_spent=time_spent,
                prev_reward=prev_reward,
                zero_streak=zero_streak,
                action=action,
            )
        )

        if observed_state != next_state or reward > 0.0:
            zero_streak = 0
        else:
            zero_streak += 1
        prev_reward = reward

    return rows


def dwell_lengths_by_patch(
    rows: list[DecisionRow],
    *,
    leave_action: int,
) -> dict[str, list[int]]:
    """Return leave dwell lengths grouped by patch label."""
    dwell_lengths: dict[str, list[int]] = {"Upper Patch": [], "Lower Patch": []}
    for row in rows:
        if row.action == leave_action:
            dwell_lengths.setdefault(row.patch_label, []).append(row.time_spent + 1)
    return dwell_lengths


def leave_probability_curve(
    rows: list[DecisionRow],
    *,
    value_getter: Callable[[DecisionRow], int],
    leave_action: int,
    max_value: int,
    patch_label: str | None = None,
) -> np.ndarray:
    """Return leave probability conditioned on a discrete decision context."""
    counts = np.zeros(max_value, dtype=float)
    leaves = np.zeros(max_value, dtype=float)
    for row in rows:
        if patch_label is not None and row.patch_label != patch_label:
            continue
        value = int(value_getter(row))
        if value < 0 or value >= max_value:
            continue
        counts[value] += 1.0
        if row.action == leave_action:
            leaves[value] += 1.0
    probs = np.full(max_value, np.nan, dtype=float)
    valid = counts > 0
    probs[valid] = leaves[valid] / counts[valid]
    return probs


def mvt_residency_deviation_by_patch(
    rows: list[DecisionRow],
    *,
    leave_action: int,
    optimal_dwell_by_state: dict[int, int],
) -> dict[str, list[int]]:
    """Return signed leave-dwell deviations from the MVT-optimal dwell."""
    deviations: dict[str, list[int]] = {"Upper Patch": [], "Lower Patch": []}
    for row in rows:
        if row.action != leave_action:
            continue
        actual_dwell = row.time_spent + 1
        optimal_dwell = optimal_dwell_by_state[row.state]
        deviations.setdefault(row.patch_label, []).append(actual_dwell - optimal_dwell)
    return deviations


def normalized_curve_auc(curve: np.ndarray) -> float:
    """Return normalized trapezoidal area over the finite observed curve support."""
    finite_x = np.flatnonzero(np.isfinite(curve))
    if finite_x.size == 0:
        return float("nan")

    finite_y = curve[finite_x]
    if finite_x.size == 1:
        return float(finite_y[0])

    x_span = float(finite_x[-1] - finite_x[0])
    if x_span <= 0.0:
        return float(finite_y[0])

    area = np.sum((finite_y[1:] + finite_y[:-1]) * np.diff(finite_x) / 2.0)
    return float(area / x_span)


def aggregate_curves(curves: list[np.ndarray]) -> CurveSummary:
    """Aggregate aligned curves with NaN-aware mean/std at each x position."""
    stacked = np.stack(curves)
    x = np.arange(stacked.shape[1], dtype=int)
    mean = np.full(stacked.shape[1], np.nan, dtype=float)
    std = np.full(stacked.shape[1], np.nan, dtype=float)
    for index in range(stacked.shape[1]):
        column = stacked[:, index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        mean[index] = float(np.mean(finite))
        std[index] = float(np.std(finite, ddof=0))
    return CurveSummary(x=x, mean=mean, std=std)
