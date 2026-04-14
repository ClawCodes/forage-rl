"""Q-table data structure with per-state variable action support."""

import numpy as np

from forage_rl.environments import Maze


class InvalidState(Exception):
    def __init__(self, state):
        self.message = state
        super().__init__(self.message)


class QTable:
    """Q-table supporting variable numbers of valid actions per state.

    Internally stores a list of per-state arrays so that only valid
    (state, action) pairs consume memory. Global action indices from
    the maze spec are mapped to compact local indices per state.

    Can operate in timed mode Q(s, t, a) or timeless mode Q(s, a).
    """

    def __init__(self, maze: Maze, timed: bool = True, initial_value: float = 0.0):
        """
        Args:
            maze: Maze instance to model q-table for
            timed: If True, table is shaped (horizon, num_valid_actions) per state;
                   if False, shaped (num_valid_actions,) per state
            initial_value: Initial Q-value for all entries
        """
        self.n_obs = maze.observation_space.n  # type: ignore
        self.timed = timed
        self.horizon = maze.horizon
        self.num_actions = maze.num_actions

        # {<state>: [<state action_1>,...,<state action_N>]}
        if not maze.observable:
            obs_to_rep: dict[int, int] = {}
            for concrete_state, obs_group in maze._state_to_observation_group.items():
                if obs_group not in obs_to_rep:
                    obs_to_rep[obs_group] = concrete_state
            self._valid_actions: dict[int, list[int]] = {
                obs: [
                    a
                    for a in range(maze.num_actions)
                    if (obs_to_rep[obs], a) in maze._transitions_by_state_action
                ]
                for obs in range(self.n_obs)
            }
        else:
            self._valid_actions = {
                s: [
                    a
                    for a in range(maze.num_actions)
                    if (s, a) in maze._transitions_by_state_action
                ]
                for s in range(self.n_obs)
            }
        # Lookup for action dimensions in self._valid_actions and self._data
        # {
        #   (<state>, <action_1>): <state action index 1>,
        #    ...,
        #   (<state>, <action_N>): <state action index N>,
        # }
        self._action_local_idx: dict[tuple[int, int], int] = {
            (s, a): i
            for s, actions in self._valid_actions.items()
            for i, a in enumerate(actions)
        }

        if timed:
            self._data: list[np.ndarray] = [
                np.full((maze.horizon, len(self._valid_actions[s])), initial_value)
                for s in range(self.n_obs)
            ]
        else:
            self._data = [
                np.full((len(self._valid_actions[s]),), initial_value)
                for s in range(self.n_obs)
            ]

    def to_array(self) -> np.ndarray:
        """Return a dense array with np.nan for invalid (state, action) pairs.

        Values are placed at their global action index so the position of an
        action is consistent across all states and time steps.

        Returns:
            Array of shape (num_states, num_actions) if timeless,
            or (num_states, horizon, num_actions) if timed.
        """
        if self.timed:
            result = np.full((self.n_obs, self.horizon, self.num_actions), np.nan)
            for s in range(self.n_obs):
                for local_idx, global_a in enumerate(self.valid_actions(s)):
                    result[s, :, global_a] = self._data[s][:, local_idx]
        else:
            result = np.full((self.n_obs, self.num_actions), np.nan)
            for s in range(self.n_obs):
                for local_idx, global_a in enumerate(self.valid_actions(s)):
                    result[s, global_a] = self._data[s][local_idx]
        return result

    def __getitem__(self, key):
        """Support legacy tuple indexing for row and scalar access."""
        if not isinstance(key, tuple):
            key = (key,)

        if self.timed:
            if len(key) == 2:
                state, time = key
                return self.action_values(state, time)
            if len(key) == 3:
                state, time, action = key
                return self.get(state, action, time)
        else:
            if len(key) == 1:
                (state,) = key
                return self.action_values(state)
            if len(key) == 2:
                state, action = key
                return self.get(state, action)

        raise TypeError(
            "QTable indices must be (state, time), (state, time, action), "
            "(state,), or (state, action) depending on timed mode."
        )

    def __setitem__(self, key, value) -> None:
        """Support legacy tuple indexing for row and scalar assignment."""
        if not isinstance(key, tuple):
            key = (key,)

        if self.timed:
            if len(key) == 2:
                state, time = key
                row = np.asarray(value, dtype=float)
                expected_width = self.num_valid_actions(state)
                if row.shape != (expected_width,):
                    raise ValueError(
                        f"Expected row shape {(expected_width,)} for state={state}, "
                        f"time={time}, got {row.shape}."
                    )
                self._data[state][time, :] = row
                return
            if len(key) == 3:
                state, time, action = key
                self.set(state, action, float(value), time)
                return
        else:
            if len(key) == 1:
                (state,) = key
                row = np.asarray(value, dtype=float)
                expected_width = self.num_valid_actions(state)
                if row.shape != (expected_width,):
                    raise ValueError(
                        f"Expected row shape {(expected_width,)} for state={state}, "
                        f"got {row.shape}."
                    )
                self._data[state][:] = row
                return
            if len(key) == 2:
                state, action = key
                self.set(state, action, float(value))
                return

        raise TypeError(
            "QTable indices must be (state, time), (state, time, action), "
            "(state,), or (state, action) depending on timed mode."
        )

    def valid_actions(self, state: int) -> list[int]:
        """Return valid global action indices for state."""
        try:
            return self._valid_actions[state]
        except KeyError:
            raise InvalidState(state)

    def num_valid_actions(self, state: int) -> int:
        """Return number of valid actions for state."""
        return len(self.valid_actions(state))

    def global_to_local(self, state: int, action: int) -> int:
        """Map global action index to local (state-relative) index."""
        try:
            return self._action_local_idx[state, action]
        except KeyError:
            raise ValueError(f"({state}, {action}) does not exist)")

    def local_to_global(self, state: int, local_idx: int) -> int:
        """Map local action index back to global action index."""
        try:
            actions = self._valid_actions[state]
        except KeyError:
            raise ValueError(f"State {state} does not exist)")

        try:
            return actions[local_idx]
        except IndexError:
            raise ValueError(f"Action index {local_idx} does not exist")

    def action_values(self, state: int, time: int = 0) -> np.ndarray:
        """Return Q-values for all valid actions at (state[, time])."""
        return self._data[state][time] if self.timed else self._data[state]

    def get(self, state: int, action: int, time: int = 0) -> float:
        """Return Q-value for a specific (state, action[, time])."""
        ai = self.global_to_local(state, action)
        return float(
            self._data[state][time, ai] if self.timed else self._data[state][ai]
        )

    def max_value(self, state: int, time: int = 0) -> float:
        """Return the maximum Q-value at (state[, time])."""
        return float(np.max(self.action_values(state, time)))

    def mean(self) -> float:
        """Return mean of all Q-values."""
        return float(np.mean(np.concatenate([d.ravel() for d in self._data])))

    # Value mutation
    def set(self, state: int, action: int, value: float, time: int = 0):
        """Set Q-value for (state, action[, time])."""
        ai = self.global_to_local(state, action)
        if self.timed:
            self._data[state][time, ai] = value
        else:
            self._data[state][ai] = value

    def update(self, state: int, action: int, delta: float, time: int = 0):
        """Add delta to Q-value at (state, action[, time])."""
        new_q = self.get(state, action, time) + delta
        self.set(state, action, new_q, time)

    # Policy extraction
    def best_local_action(self, state: int, time: int = 0) -> int:
        """Return local index of the greedy action at (state[, time])."""
        return int(np.argmax(self.action_values(state, time)))

    def best_global_action(self, state: int, time: int = 0) -> int:
        """Return global action index of the greedy action at (state[, time])."""
        return self.local_to_global(state, self.best_local_action(state, time))

    def policy(self) -> np.ndarray:
        """Return greedy policy as global action indices.

        Returns:
            Array of shape (num_states, horizon) if timed else (num_states,)
        """
        n = len(self._data)
        if self.timed:
            result = np.zeros((n, self.horizon), dtype=int)
            for s in range(n):
                for t in range(self.horizon):
                    result[s, t] = self.best_global_action(s, t)
        else:
            result = np.zeros(n, dtype=int)
            for s in range(n):
                result[s] = self.best_global_action(s)
        return result
