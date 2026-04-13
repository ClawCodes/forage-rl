import pytest

torch = pytest.importorskip("torch")

from forage_rl.agents import get_agent
from forage_rl.agents.registry import Agent
from forage_rl.environments import Maze


@pytest.mark.parametrize(
    "agent_type",
    [Agent.DQN, Agent.ELMAN, Agent.GRU, Agent.LSTM],
)
def test_neural_agents_only_choose_valid_corridor_action(agent_type):
    maze = Maze.from_spec("simple_one_way", seed=0)
    agent = get_agent(agent_type, maze, num_episodes=1, device="cpu", seed=0)

    action = agent.choose_action(1, 0, prev_action=1, prev_reward=0.0)

    assert action == 1


def test_masked_max_q_values_ignores_invalid_corridor_actions():
    maze = Maze.from_spec("simple_one_way", seed=0)
    agent = get_agent(Agent.DQN, maze, num_episodes=1, device="cpu", seed=0)
    q_values = torch.tensor([[10.0, 1.0]], device=agent.torch_device)

    masked_max = agent.masked_max_q_values(q_values, [1])

    assert masked_max.shape == (1,)
    assert float(masked_max.item()) == pytest.approx(1.0)
