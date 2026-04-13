import pytest

from forage_rl.agents.dqn import DQNAgent
from forage_rl.agents.registry import Agent, get_agent
from forage_rl.agents.recurrent import LSTMAgent
from forage_rl.environments import Maze


def test_dqn_rejects_invalid_neural_hyperparameters():
    maze = Maze.from_spec("simple", seed=0)

    with pytest.raises(ValueError, match="batch_size"):
        DQNAgent(maze, num_episodes=1, device="cpu", seed=0, batch_size=0)
    with pytest.raises(ValueError, match="replay_capacity"):
        DQNAgent(maze, num_episodes=1, device="cpu", seed=0, replay_capacity=0)
    with pytest.raises(ValueError, match="target_update_interval"):
        DQNAgent(
            maze,
            num_episodes=1,
            device="cpu",
            seed=0,
            target_update_interval=0,
        )
    with pytest.raises(ValueError, match="learning_rate"):
        DQNAgent(maze, num_episodes=1, device="cpu", seed=0, learning_rate=0.0)
    with pytest.raises(ValueError, match="gradient_clip"):
        DQNAgent(maze, num_episodes=1, device="cpu", seed=0, gradient_clip=0.0)
    with pytest.raises(ValueError, match="num_episodes"):
        DQNAgent(maze, num_episodes=-1, device="cpu", seed=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        DQNAgent(maze, num_episodes=1, device="cpu", seed=0, warmup_steps=-1)


def test_recurrent_agents_reject_invalid_sequence_hyperparameters():
    maze = Maze.from_spec("simple", seed=0)

    with pytest.raises(ValueError, match="sequence_length"):
        LSTMAgent(maze, num_episodes=1, device="cpu", seed=0, sequence_length=0)
    with pytest.raises(ValueError, match="burn_in"):
        LSTMAgent(maze, num_episodes=1, device="cpu", seed=0, burn_in=-1)
    with pytest.raises(ValueError, match="recurrent_hidden_size"):
        LSTMAgent(
            maze,
            num_episodes=1,
            device="cpu",
            seed=0,
            recurrent_hidden_size=0,
        )
    with pytest.raises(ValueError, match="recurrent_num_layers"):
        LSTMAgent(
            maze,
            num_episodes=1,
            device="cpu",
            seed=0,
            recurrent_num_layers=0,
        )


def test_neural_alpha_alias_sets_learning_rate_when_explicitly_provided():
    maze = Maze.from_spec("simple", seed=0)

    agent = get_agent(Agent.DQN, maze, num_episodes=1, device="cpu", seed=0, alpha=0.02)

    assert agent.learning_rate == pytest.approx(0.02)
    assert agent.alpha == pytest.approx(0.02)


def test_neural_alpha_alias_rejects_conflicting_learning_rate():
    maze = Maze.from_spec("simple", seed=0)

    with pytest.raises(ValueError, match="legacy alias for learning_rate"):
        get_agent(
            Agent.DQN,
            maze,
            num_episodes=1,
            device="cpu",
            seed=0,
            alpha=0.02,
            learning_rate=0.005,
        )
