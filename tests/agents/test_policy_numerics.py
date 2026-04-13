import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.environments import Maze


class DummyAgent(BaseAgent):
    def simulate(self, trajectory):
        raise NotImplementedError

    def train(self, verbose: bool = True):
        raise NotImplementedError


def test_boltzmann_action_probs_are_finite_and_normalized_for_large_q_values():
    agent = DummyAgent(Maze.from_spec("simple", seed=0), beta=10.0, seed=0)
    q_values = np.array([10_000.0, 10_001.0, 9_999.0])

    probs = agent.boltzmann_action_probs(q_values)

    assert np.all(np.isfinite(probs))
    assert np.isclose(np.sum(probs), 1.0)
    assert int(np.argmax(probs)) == 1


def test_choose_action_boltzmann_handles_large_q_values_without_overflow():
    agent = DummyAgent(Maze.from_spec("simple", seed=0), beta=25.0, seed=7)
    q_values = np.array([10_000.0, 10_001.0, 9_999.0])

    sampled_actions = [agent.choose_action_boltzmann(q_values) for _ in range(50)]

    assert set(sampled_actions) <= {0, 1, 2}
