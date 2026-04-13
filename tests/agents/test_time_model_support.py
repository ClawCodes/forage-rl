import pytest

from forage_rl.agents.dqn import DQNAgent
from forage_rl.agents.model_based import MBRL
from forage_rl.agents.q_learning import QLearningTime
from forage_rl.agents.recurrent import LSTMAgent
from forage_rl.agents.value_iteration import ValueIterationSolver
from forage_rl.environments import Maze
from forage_rl.environments.specs import (
    MazeMeta,
    MazeSpec,
    StateSpec,
    TransitionDurationSpec,
)


def _duration_maze() -> Maze:
    spec = MazeSpec(
        maze=MazeMeta(
            name="duration-test",
            horizon=10,
            initial_state=0,
            action_labels=["stay", "leave"],
        ),
        states=[
            StateSpec(id=0, label="Upper", decay=0.2, observation_group=0),
            StateSpec(id=1, label="Lower", decay=0.5, observation_group=1),
        ],
        transitions=[
            TransitionDurationSpec(
                state=0,
                action=0,
                next_state=0,
                prob=1.0,
                duration=2,
            ),
            TransitionDurationSpec(
                state=0,
                action=1,
                next_state=1,
                prob=1.0,
                duration=3,
            ),
            TransitionDurationSpec(
                state=1,
                action=0,
                next_state=1,
                prob=1.0,
                duration=2,
            ),
            TransitionDurationSpec(
                state=1,
                action=1,
                next_state=0,
                prob=1.0,
                duration=3,
            ),
        ],
    )
    return Maze(spec, seed=0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda maze: MBRL(maze, num_episodes=1, seed=0),
        lambda maze: QLearningTime(maze, num_episodes=1, seed=0),
        lambda maze: DQNAgent(maze, num_episodes=1, device="cpu", seed=0),
        lambda maze: LSTMAgent(maze, num_episodes=1, device="cpu", seed=0),
    ],
)
def test_time_aware_agents_reject_transition_duration_specs(factory):
    maze = _duration_maze()

    with pytest.raises(ValueError, match="transition durations"):
        factory(maze)


def test_value_iteration_rejects_transition_duration_specs():
    maze = _duration_maze()

    with pytest.raises(ValueError, match="transition durations"):
        ValueIterationSolver(maze)
