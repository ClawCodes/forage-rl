from forage_rl.environments import Maze
from forage_rl.environments.spec_loader import load_builtin_maze_spec
from forage_rl.types import Transition
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


class TestMazeStep:
    def test_step_stay_returns_same_state(self):
        """Staying in a patch keeps the agent in the same state"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        obs, _, _, _, _ = maze.step(0)

        # Assert
        assert obs == 0

    def test_step_leave_returns_other_state(self):
        """Leaving a patch moves the agent to the other state in simple maze"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        obs, _, _, _, _ = maze.step(1)

        # Assert
        assert obs == 1

    def test_step_returns_gym_5_tuple_types(self):
        """Step returns (int, float, bool, bool, dict) with expected info keys"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        obs, reward, terminated, truncated, info = maze.step(0)

        # Assert
        assert isinstance(obs, int)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "prev_state" in info
        assert "action" in info

    def test_step_stay_returns_reward(self):
        """Staying in a patch yields a reward in {0.0, 1.0}"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        _, reward, _, _, _ = maze.step(0)

        # Assert
        assert reward in (0.0, 1.0)

    def test_step_leave_returns_zero_reward(self):
        """Leaving a patch always yields zero reward"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        _, reward, _, _, _ = maze.step(1)

        # Assert
        assert reward == 0.0

    def test_step_increments_time(self):
        """Each step increments time by 1"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        maze.step(0)
        maze.step(0)
        maze.step(1)

        # Assert
        assert maze.time == 3

    def test_step_truncates_at_horizon(self):
        """Truncated is True after horizon steps"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42, horizon=5)
        maze.reset()

        # Act
        for _ in range(4):
            _, _, _, truncated, _ = maze.step(0)
            assert truncated is False
        _, _, _, truncated, _ = maze.step(0)

        # Assert
        assert truncated is True

    def test_step_terminated_is_always_false(self):
        """Terminated is always False (no absorbing states)"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42, horizon=5)
        maze.reset()

        # Act & Assert
        for _ in range(5):
            _, _, terminated, _, _ = maze.step(0)
            assert terminated is False

    def test_step_reward_decays_over_stays(self):
        """Average reward decreases over consecutive stays in a high-decay patch"""
        # Arrange
        n_trials = 500
        early_rewards = []
        late_rewards = []

        for trial in range(n_trials):
            maze = Maze.from_spec("simple", seed=trial)
            maze.reset()
            # Move to state 1 (decay=3.0, fast depletion)
            maze.step(1)

            # Act - collect early reward (step 1 in patch)
            _, reward, _, _, _ = maze.step(0)
            early_rewards.append(reward)

            # Stay several more times to deplete
            for _ in range(5):
                maze.step(0)

            # Collect late reward (step 7 in patch)
            _, reward, _, _, _ = maze.step(0)
            late_rewards.append(reward)

        # Assert
        assert np.mean(early_rewards) > np.mean(late_rewards)

    def test_step_info_contains_prev_state_and_action(self):
        """Info dict contains the previous state and action taken"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        _, _, _, _, info = maze.step(1)

        # Assert
        assert info["prev_state"] == 0
        assert info["action"] == 1


class TestMazeStepTransition:
    def test_step_transition_returns_transition_and_done(self):
        """step_transition returns a (Transition, bool) tuple"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42)
        maze.reset()

        # Act
        result = maze.step_transition(0)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Transition)
        assert isinstance(result[1], bool)

    def test_step_transition_fields_match_step(self):
        """step_transition fields match the equivalent step() call"""
        # Arrange
        maze_step = Maze.from_spec("simple", seed=42)
        maze_step.reset()
        maze_trans = Maze.from_spec("simple", seed=42)
        maze_trans.reset()

        # Act
        obs, reward, terminated, truncated, info = maze_step.step(1)
        transition, done = maze_trans.step_transition(1)

        # Assert
        assert transition.state == info["prev_state"]
        assert transition.action == info["action"]
        assert transition.reward == reward
        assert transition.next_state == obs
        assert done == (terminated or truncated)

    def test_step_transition_done_at_horizon(self):
        """step_transition returns done=True after horizon steps"""
        # Arrange
        maze = Maze.from_spec("simple", seed=42, horizon=3)
        maze.reset()

        # Act
        _, done1 = maze.step_transition(0)
        _, done2 = maze.step_transition(0)
        _, done3 = maze.step_transition(0)

        # Assert
        assert done1 is False
        assert done2 is False
        assert done3 is True
