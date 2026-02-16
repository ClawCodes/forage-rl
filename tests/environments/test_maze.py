from forage_rl.environments import Maze
from forage_rl.environments.spec_loader import load_builtin_maze_spec
import numpy as np
from gymnasium.spaces import Discrete


class TestMazeInitialization:
    def test_maze_from_spec_simple(self):
        """A simple sanity check test ensuring values are assigned correctly in the __init__ method"""
        # Arrange
        expected_spec = load_builtin_maze_spec("simple")
        expected_rng = np.random.default_rng(42)
        expected_transitions = {
            (0, 0): [(0, 1.0)],
            (0, 1): [(1, 1.0)],
            (1, 0): [(1, 1.0)],
            (1, 1): [(0, 1.0)],
        }

        # Act
        maze = Maze.from_spec("simple", rng=expected_rng)

        # Assert
        assert maze.maze_spec == expected_spec
        assert maze.horizon == 100
        assert maze.rng == expected_rng
        assert maze.num_actions == 2
        assert maze.num_actions == 2
        assert maze.decays == expected_spec.decays
        assert maze.state_labels == expected_spec.state_labels
        assert maze.action_labels == expected_spec.maze.action_labels
        assert maze.initial_state == 0
        assert maze.observation_space == Discrete(2)
        assert maze.action_space == Discrete(2)
        assert maze.state == 0
        assert maze._transitions_by_state_action == expected_transitions
        assert [r.decay for r in maze.reward_models] == expected_spec.decays


class TestMazeReset:
    def test_maze_reset_no_seed(self):
        """Test that calling reset without seed preserves original rng"""
        # Arrange
        expected_rng = np.random.default_rng(42)
        maze = Maze.from_spec("simple", rng=expected_rng)

        # Act
        _ = maze.step(1)
        reset_state, _ = maze.reset()

        # Assert
        expected_rng_state = expected_rng.bit_generator.state
        assert reset_state == 0
        assert maze.time == 0
        # Internal rng state should persist without seed
        assert maze.rng.bit_generator.state == expected_rng_state
        assert all(
            [
                model.rng.bit_generator.state == expected_rng_state
                for model in maze.reward_models
            ]
        )

    def test_maze_reset_with_seed(self):
        """Test that calling reset with alters rng state"""
        # Arrange
        initial_rng = np.random.default_rng(42)
        maze = Maze.from_spec("simple", rng=initial_rng)
        new_seed = 0
        expected_rng = np.random.default_rng(new_seed)

        # Act
        _ = maze.step(1)
        reset_state, _ = maze.reset(seed=new_seed)

        # Assert
        expected_rng_state = expected_rng.bit_generator.state
        assert reset_state == 0
        assert maze.time == 0
        # Internal rng state should change with seed
        assert maze.rng.bit_generator.state == expected_rng_state
        assert all(
            [
                model.rng.bit_generator.state == expected_rng_state
                for model in maze.reward_models
            ]
        )
