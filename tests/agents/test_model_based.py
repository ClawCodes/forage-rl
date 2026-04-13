import numpy as np
import pytest

from forage_rl import Trajectory, Transition
from forage_rl.agents.model_based import MBRL
from forage_rl.environments import Maze, MazePOMDP, load_builtin_maze_spec
from forage_rl.environments.specs import (
    MazeMeta,
    MazeSpec,
    StateSpec,
    TransitionStepSpec,
)


def _belief_dependent_pomdp_spec() -> MazeSpec:
    return MazeSpec(
        maze=MazeMeta(
            name="belief-dependent",
            horizon=10,
            initial_state=0,
            action_labels=["stay", "leave"],
        ),
        states=[
            StateSpec(id=0, label="Upper A", decay=0.2, observation_group=0),
            StateSpec(id=1, label="Upper B", decay=0.2, observation_group=0),
            StateSpec(id=2, label="Lower A", decay=0.2, observation_group=1),
            StateSpec(id=3, label="Lower B", decay=0.2, observation_group=1),
        ],
        transitions=[
            TransitionStepSpec(state=0, action=0, next_state=0, prob=1.0),
            TransitionStepSpec(state=0, action=1, next_state=2, prob=1.0),
            TransitionStepSpec(state=1, action=0, next_state=1, prob=1.0),
            TransitionStepSpec(state=1, action=1, next_state=1, prob=1.0),
            TransitionStepSpec(state=2, action=0, next_state=2, prob=1.0),
            TransitionStepSpec(state=2, action=1, next_state=0, prob=1.0),
            TransitionStepSpec(state=3, action=0, next_state=3, prob=1.0),
            TransitionStepSpec(state=3, action=1, next_state=0, prob=1.0),
        ],
    )


def test_mbrl_uses_spec_driven_stochastic_leave_transitions_for_full_maze():
    maze = Maze(load_builtin_maze_spec("full"), seed=0)
    agent = MBRL(
        maze,
        num_episodes=1,
        gamma=1.0,
        num_planning_steps=1,
        seed=0,
    )
    agent.q_table[3, 0] = np.array([1.0, 2.0])
    agent.q_table[4, 0] = np.array([3.0, 4.0])
    agent.q_table[5, 0] = np.array([5.0, 6.0])

    agent.q_value_iteration()

    expected = (2.0 + 4.0 + 6.0) / 3.0
    assert np.isclose(agent.q_table[0, 0, 1], expected)


@pytest.mark.parametrize("maze_name", ["full"])
def test_pomdp_planning_distribution_is_belief_independent_for_benchmark_mazes(
    maze_name: str,
):
    maze = MazePOMDP(load_builtin_maze_spec(maze_name), seed=0)
    agent = MBRL(maze, num_episodes=1, num_planning_steps=1, seed=0)

    assert maze.planning_transition_distribution(0, 1) == [(1, 1.0)]
    assert maze.planning_transition_distribution(1, 1) == [(0, 1.0)]
    agent.q_value_iteration()


def test_pomdp_planning_distribution_rejects_belief_dependent_hidden_dynamics():
    maze = MazePOMDP(_belief_dependent_pomdp_spec(), seed=0)

    with pytest.raises(ValueError, match="Belief-state planning"):
        maze.planning_transition_distribution(0, 1)


def test_mbrl_simulate_requires_timed_transition_inputs():
    maze = Maze.from_spec("simple", seed=0)
    agent = MBRL(maze, num_episodes=1, seed=0)
    trajectory = Trajectory(
        transitions=[
            Transition(
                state=0,
                action=0,
                reward=1.0,
                next_state=0,
            )
        ]
    )

    with pytest.raises(ValueError, match="time_spent"):
        agent.simulate(trajectory)
