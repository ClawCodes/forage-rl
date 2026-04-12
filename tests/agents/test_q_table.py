"""Tests for QTable — per-state variable-action Q-value storage."""

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from forage_rl.agents.q_table import QTable
from forage_rl.environments import Maze


# Helpers
def simple_maze(horizon: int = 5) -> Maze:
    """2-state, 2-action maze; all (state, action) pairs are valid."""
    return Maze.from_spec("simple", seed=0, horizon=horizon)


class _StubMaze:
    """Minimal maze stub for testing variable-action-count behaviour.

    State 0 has only action 0 valid.
    State 1 has actions 0 and 1 valid.
    """

    def __init__(self):
        self.observation_space = Discrete(2)
        self.num_actions = 2
        self.horizon = 4
        self._transitions_by_state_action = {
            (0, 1): [(1, 1.0)],
            (1, 0): [(1, 1.0)],
            (1, 1): [(0, 1.0)],
        }


class TestQTableInit:
    def test_timeless_data_shape(self):
        """Timeless table stores a 1D array per state with length == num valid actions (i.e. 2 for simple maze)."""
        # Arrange
        maze = simple_maze()

        # Act
        qt = QTable(maze, timed=False)

        # Assert
        assert qt._data[0].shape == (2,)
        assert qt._data[1].shape == (2,)

    def test_timed_data_shape(self):
        """Timed table stores a 2D (horizon, num_valid_actions) array per state."""
        # Arrange
        maze = simple_maze(horizon=5)

        # Act
        qt = QTable(maze, timed=True)

        # Assert
        assert qt._data[0].shape == (5, 2)
        assert qt._data[1].shape == (5, 2)

    def test_initial_value_applied(self):
        """All entries equal initial_value after construction."""
        # Arrange
        maze = simple_maze()

        # Act
        qt = QTable(maze, timed=False, initial_value=3.14)

        # Assert
        assert np.all(qt._data[0] == 3.14)
        assert np.all(qt._data[1] == 3.14)

    def test_valid_actions_matches_transitions(self):
        """valid_actions reflects exactly the (state, action) pairs in maze spec."""
        # Arrange
        maze = simple_maze()

        # Act
        qt = QTable(maze, timed=False)

        # Assert
        assert qt.valid_actions(0) == [0, 1]
        assert qt.valid_actions(1) == [0, 1]

    def test_num_valid_actions(self):
        """num_valid_actions returns the count of valid actions per state."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act / Assert
        assert qt.num_valid_actions(0) == 2
        assert qt.num_valid_actions(1) == 2

    def test_variable_actions_per_state(self):
        """States can have different numbers of valid actions."""
        # Arrange
        stub = _StubMaze()

        # Act
        qt = QTable(stub, timed=False)  # type: ignore

        # Assert
        assert qt.num_valid_actions(0) == 1
        assert qt.num_valid_actions(1) == 2
        assert qt._data[0].shape == (1,)
        assert qt._data[1].shape == (2,)

    def test_variable_actions_per_state_with_horizon(self):
        """States can have different numbers of valid actions."""
        # Arrange
        stub = _StubMaze()

        # Act
        qt = QTable(stub, timed=True)  # type: ignore

        # Assert
        assert qt.num_valid_actions(0) == 1
        assert qt.num_valid_actions(1) == 2
        assert qt._data[0].shape == (4, 1)
        assert qt._data[1].shape == (4, 2)

    def test_horizon_stored(self):
        """The horizon attribute matches the maze horizon."""
        maze = simple_maze(horizon=7)
        qt = QTable(maze, timed=True)
        assert qt.horizon == 7

    def test_builtin_one_way_maze_has_sparse_valid_actions(self):
        maze = Maze.from_spec("simple_one_way", seed=0)

        qt = QTable(maze, timed=False)

        assert qt.valid_actions(0) == [0, 1]
        assert qt.valid_actions(1) == [1]
        assert qt.valid_actions(2) == [1]
        assert qt.valid_actions(3) == [0, 1]


class TestQTableIndexTranslation:
    def test_global_to_local_maps_simple_state_actions_q_table_actions(self):
        """Action 0 is local index 0 for every state."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act/Assert
        assert qt.global_to_local(0, 0) == 0
        assert qt.global_to_local(0, 1) == 1
        assert qt.global_to_local(1, 0) == 0
        assert qt.global_to_local(1, 1) == 1

    def test_local_to_global_roundtrip(self):
        """local_to_global(global_to_local(a)) == a for all valid actions."""
        maze = simple_maze()
        qt = QTable(maze, timed=False)
        assert qt.local_to_global(0, qt.global_to_local(0, 0)) == 0
        assert qt.local_to_global(0, qt.global_to_local(0, 1)) == 1
        assert qt.local_to_global(1, qt.global_to_local(1, 0)) == 0
        assert qt.local_to_global(1, qt.global_to_local(1, 1)) == 1

    def test_variable_actions_local_indices(self):
        """State with one valid action maps it to local index 0."""
        stub = _StubMaze()
        qt = QTable(stub, timed=False)  # type: ignore

        # State 0 has only action 1
        assert qt.global_to_local(0, 1) == 0
        assert qt.local_to_global(0, 0) == 1

        # State 1: action 0 → local 0, action 1 → local 1
        assert qt.global_to_local(1, 0) == 0
        assert qt.global_to_local(1, 1) == 1

    def test_global_to_local_invalid_raises(self):
        """Test ValueError raised when invalid state-action pair accessed"""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False, initial_value=1.0)

        # Act/Assert
        with pytest.raises(ValueError, match=r"(0, 2).*"):
            qt.global_to_local(0, 2)

    def test_local_to_global_invalid_state_raises(self):
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False, initial_value=1.0)

        # Act/Assert
        with pytest.raises(ValueError, match=r"State 2.*"):
            qt.local_to_global(2, 0)

    def test_local_to_global_invalid_action_raises(self):
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False, initial_value=1.0)

        # Act/Assert
        with pytest.raises(ValueError, match=r"Action index 2.*"):
            qt.local_to_global(0, 2)


class TestQTableValueAccess:
    def test_get_timeless_initial(self):
        """get returns initial_value for any valid (state, action) pair."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False, initial_value=1.0)

        # Act/Assert
        assert qt.get(0, 0) == 1.0
        assert qt.get(0, 1) == 1.0
        assert qt.get(1, 0) == 1.0
        assert qt.get(1, 1) == 1.0

    def test_get_timed_initial(self):
        """get returns initial_value for any valid (state, action, time) triple."""
        # Arrange
        maze = simple_maze(horizon=5)
        qt = QTable(maze, timed=True, initial_value=2.5)

        # Act/Assert
        assert qt.get(0, 0, 0) == 2.5
        assert qt.get(0, 1, 0) == 2.5
        assert qt.get(0, 1, 1) == 2.5
        assert qt.get(1, 0, 0) == 2.5
        assert qt.get(1, 0, 1) == 2.5
        assert qt.get(1, 1, 1) == 2.5

    def test_get_invalid_action_raises(self):
        """raises ValueError when Q-value is invalid."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False, initial_value=1.0)

        # Act/Assert
        with pytest.raises(ValueError, match=r"(0, 2).*"):
            qt.get(0, 2)

    def test_action_values_timeless_shape(self):
        """action_values returns a 1D array of length num_valid_actions."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act/Assert
        assert qt.action_values(0).shape == (qt.num_valid_actions(0),)
        assert qt.action_values(1).shape == (qt.num_valid_actions(1),)

    def test_action_values_timed_shape(self):
        """action_values at a given time returns a 1D array of length num_valid_actions."""
        # Arange
        maze = simple_maze(horizon=5)
        qt = QTable(maze, timed=True)

        # Act/Assert
        assert qt.action_values(0, 0).shape == (qt.num_valid_actions(0),)
        assert qt.action_values(0, 1).shape == (qt.num_valid_actions(0),)
        assert qt.action_values(0, 4).shape == (qt.num_valid_actions(0),)
        assert qt.action_values(1, 0).shape == (qt.num_valid_actions(0),)
        assert qt.action_values(1, 1).shape == (qt.num_valid_actions(0),)
        assert qt.action_values(1, 4).shape == (qt.num_valid_actions(0),)

    def test_action_values_reflects_updates(self):
        """action_values returns the live view; mutations are visible."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act
        initial_av = qt.action_values(0).copy()
        qt.set(0, 1, 9.9)
        av = qt.action_values(0)

        # Assert
        assert initial_av[0] == av[0]
        assert initial_av[1] != av[1]
        assert av[1] == 9.9

    def test_max_value_timeless(self):
        """max_value returns the maximum Q-value across valid actions."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)
        expected = 5.0

        # Act
        qt.set(0, 0, 1.0)
        qt.set(0, 1, 5.0)
        actual = qt.max_value(0)

        assert actual == expected

    def test_max_value_timed(self):
        """max_value respects the time dimension."""
        # Arrange
        maze = simple_maze(horizon=5)
        qt = QTable(maze, timed=True)

        # Act
        qt.set(1, 0, 3.0, time=2)
        qt.set(1, 1, 7.0, time=2)
        qt.set(1, 0, 99.0, time=3)  # different time — should not affect t=2

        assert qt.max_value(1, time=2) == 7.0
        assert qt.max_value(1, time=3) == 99.0

    def test_mean_initial_zero(self):
        """mean() returns 0.0 when all values are initialized to zero."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)
        expected = 0.0

        # Act
        actual = qt.mean()

        # Assert
        assert actual == expected

    def test_mean_after_initial_value_change(self):
        """mean() reflects different initial values."""
        # Arrange
        maze = simple_maze(horizon=1)
        qt = QTable(maze, timed=False, initial_value=4.0)

        # Act/Assert
        assert qt.mean() == pytest.approx(4.0)


class TestQTableMutation:
    def test_set_timeless(self):
        """set overwrites the Q-value in timeless mode."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act/Assert
        qt.set(0, 1, 42.0)

        assert qt.get(0, 0) == 0.0  # Unchanged
        assert qt.get(0, 1) == 42.0

    def test_set_timed(self):
        """set overwrites the Q-value at the correct time step."""
        # Arrange
        maze = simple_maze(horizon=5)
        qt = QTable(maze, timed=True)

        # Act
        qt.set(1, 0, 7.5, time=3)

        # Assert
        assert qt.get(1, 0, 3) == 7.5
        assert qt.get(1, 0, 2) == 0.0  # Unchanged

    def test_update_timeless(self):
        """update adds delta to the existing Q-value."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False, initial_value=1.0)

        # Act
        qt.update(0, 0, 2.5)

        # Assert
        assert qt.get(0, 0) == pytest.approx(3.5)
        assert qt.get(0, 1) == 1.0  # Unchanged

    def test_update_timed(self):
        """update adds delta at the correct time step only."""
        # Arrange
        maze = simple_maze(horizon=5)
        qt = QTable(maze, timed=True)

        # Act
        qt.update(0, 1, 3.0, time=2)

        # Assert
        assert qt.get(0, 1, 2) == pytest.approx(3.0)
        assert qt.get(0, 1, 1) == 0.0

    def test_update_accumulates(self):
        """Repeated updates accumulate correctly."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act
        qt.update(1, 0, 1.0)
        qt.update(1, 0, 1.0)
        qt.update(1, 0, 1.0)

        # Assert
        assert qt.get(1, 0) == pytest.approx(3.0)

    def test_set_does_not_affect_other_entries(self):
        """set on one (state, action) leaves all others unchanged."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act
        qt.set(0, 0, 99.0)

        # Assert
        assert qt.get(0, 1) == 0.0
        assert qt.get(1, 0) == 0.0
        assert qt.get(1, 1) == 0.0


class TestQTablePolicy:
    def test_best_local_action_timeless(self):
        """best_local_action returns the argmax over action_values."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act
        qt.set(0, 0, 1.0)
        qt.set(0, 1, 5.0)

        # Assert
        assert qt.best_local_action(0) == 1

    def test_best_local_action_timed(self):
        """best_local_action at time t uses the correct row."""
        # Arrange
        maze = simple_maze(horizon=5)
        qt = QTable(maze, timed=True)

        # Act
        qt.set(1, 0, 10.0, time=2)
        qt.set(1, 1, 3.0, time=2)

        # Assert
        assert qt.best_local_action(1, time=2) == 0

    def test_best_global_action_consistent_with_local(self):
        """best_global_action == local_to_global(best_local_action)."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act
        qt.set(0, 0, 2.0)
        qt.set(0, 1, 9.0)

        # Assert
        assert qt.best_global_action(0) == qt.local_to_global(
            0, qt.best_local_action(0)
        )

    def test_policy_timeless_shape(self):
        """policy() in timeless mode returns shape (num_states,)."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act
        policy = qt.policy()

        # Assert
        assert policy.shape == (maze.num_states,)

    def test_policy_timed_shape(self):
        """policy() in timed mode returns shape (num_states, horizon)."""
        # Arrange
        horizon = 5
        maze = simple_maze(horizon=horizon)
        qt = QTable(maze, timed=True)

        # Act
        policy = qt.policy()

        # Assert
        assert policy.shape == (maze.num_states, horizon)

    def test_policy_timeless_values_are_global_actions(self):
        """Each entry in a timeless policy is a valid global action for that state."""
        # Arrange
        maze = simple_maze()
        qt = QTable(maze, timed=False)

        # Act
        qt.set(0, 1, 1.0)  # state 0 should prefer action 1
        qt.set(1, 0, 1.0)  # state 1 should prefer action 0

        policy = qt.policy()

        # Assert
        assert policy[0] == 1
        assert policy[1] == 0

    def test_policy_timed_values_are_global_actions(self):
        """Each entry in a timed policy is the greedy global action at that (state, time)."""
        # Arrange
        maze = simple_maze(horizon=3)
        qt = QTable(maze, timed=True)

        # Act
        qt.set(0, 0, 5.0, time=1)  # state 0, t=1: prefer action 0
        qt.set(0, 1, 9.0, time=2)  # state 0, t=2: prefer action 1
        policy = qt.policy()

        # Assert
        assert policy[0, 1] == 0
        assert policy[0, 2] == 1

    def test_policy_variable_actions(self):
        """policy() maps to the single valid global action where only one exists."""
        # Arrange
        stub = _StubMaze()
        qt = QTable(stub, timed=False)  # type: ignore

        # Act
        # State 0 only has action 1; set it to a non-zero value to be explicit
        qt.set(0, 1, 1.0)
        policy = qt.policy()

        # Assert
        assert policy[0] == 1  # only valid action
