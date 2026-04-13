import numpy as np
import pytest

from forage_rl.agents.value_iteration import ValueIterationSolver
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.environments.specs import (
    MazeMeta,
    MazeSpec,
    StateSpec,
    TransitionStepSpec,
)


def _reward_maze(
    *,
    decay: float | None,
    initial_reward_prob: float = 1.0,
    reward_probs: list[float] | None = None,
) -> Maze:
    spec = MazeSpec(
        maze=MazeMeta(
            name="reward-test",
            horizon=10,
            initial_state=0,
            action_labels=["stay", "leave"],
        ),
        states=[
            StateSpec(
                id=0,
                label="Upper",
                decay=decay,
                initial_reward_prob=initial_reward_prob,
                reward_probs=reward_probs,
                observation_group=0,
            ),
            StateSpec(
                id=1,
                label="Lower",
                decay=0.5,
                observation_group=1,
            ),
        ],
        transitions=[
            TransitionStepSpec(state=0, action=0, next_state=0, prob=1.0),
            TransitionStepSpec(state=0, action=1, next_state=1, prob=1.0),
            TransitionStepSpec(state=1, action=0, next_state=1, prob=1.0),
            TransitionStepSpec(state=1, action=1, next_state=0, prob=1.0),
        ],
    )
    return Maze(spec, seed=0)


def test_value_iteration_rejects_pomdp_mazes():
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=0)

    with pytest.raises(ValueError, match="MazePOMDP"):
        ValueIterationSolver(maze)


def test_value_iteration_uses_plain_exponential_reward_probability():
    solver = ValueIterationSolver(_reward_maze(decay=0.5))

    assert np.isclose(solver._get_expected_reward(0, 2, 0), np.exp(-1.0))


def test_value_iteration_respects_initial_reward_probability_scaling():
    solver = ValueIterationSolver(_reward_maze(decay=0.5, initial_reward_prob=0.35))

    assert np.isclose(
        solver._get_expected_reward(0, 2, 0),
        0.35 * np.exp(-1.0),
    )


def test_value_iteration_uses_reward_schedule_when_present():
    solver = ValueIterationSolver(
        _reward_maze(
            decay=None,
            reward_probs=[1.0, 0.75, 0.4],
        )
    )

    assert solver._get_expected_reward(0, 5, 0) == 0.4
