"""Fully observed oracle patch-leaving benchmarks derived from the MDP policy."""

from __future__ import annotations

import numpy as np

from forage_rl.agents.value_iteration import ValueIterationSolver
from forage_rl.analysis.action_semantics import patch_exit_action_indices
from forage_rl.environments import Maze


def oracle_patch_optimal_prt_by_state(maze: Maze) -> dict[int, int]:
    """Return oracle-optimal patch residence times for patch states only."""
    _, policy = ValueIterationSolver(maze).solve(verbose=False)
    stay_action = maze.action_labels.index("stay")
    exit_actions = patch_exit_action_indices(maze)

    optimal_prt_by_state: dict[int, int] = {}
    for state in range(maze.num_states):
        valid_actions = maze.valid_actions(state)
        if stay_action not in valid_actions:
            continue
        state_exit_actions = tuple(
            action for action in valid_actions if action in exit_actions and action != stay_action
        )
        if not state_exit_actions:
            continue
        leave_times = np.flatnonzero(np.isin(policy[state], state_exit_actions))
        optimal_prt_by_state[state] = (
            int(leave_times[0]) + 1 if leave_times.size > 0 else maze.horizon
        )
    return optimal_prt_by_state
