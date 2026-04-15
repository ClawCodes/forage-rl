"""Helpers for interpreting patch exit actions across maze variants."""

from __future__ import annotations

from forage_rl.environments import Maze


def patch_exit_action_indices(maze: Maze) -> tuple[int, ...]:
    """Return action indices that exit a patch across all patch states.

    A patch state is any state that permits ``stay``. All other valid actions on
    such states are treated as patch exits, which supports both the standard
    single-``leave`` mazes and detour variants with multiple exit actions.
    """
    stay_action = maze.action_labels.index("stay")
    exit_actions: set[int] = set()
    for state in range(maze.num_states):
        valid_actions = set(maze.valid_actions(state))
        if stay_action not in valid_actions:
            continue
        exit_actions.update(action for action in valid_actions if action != stay_action)

    if not exit_actions:
        raise ValueError("Maze exposes no patch exit actions.")
    return tuple(sorted(exit_actions))
