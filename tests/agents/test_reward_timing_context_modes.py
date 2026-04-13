from pathlib import Path

import numpy as np
import pytest

from forage_rl.agents import get_agent
from forage_rl.agents.registry import Agent
from forage_rl.environments import MazePOMDP, load_builtin_maze_spec
from forage_rl.utils.torch_support import torch_available


torch_required = pytest.mark.skipif(not torch_available(), reason="torch not installed")


@torch_required
def test_prev_reward_context_feature_encoding():
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    agent = get_agent(
        Agent.DQN,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        context_mode="prev_reward",
    )

    assert agent.feature_dim == agent.obs_dim + 1
    assert agent.feature_schema_components == ("observation_one_hot", "prev_reward")
    np.testing.assert_allclose(
        agent.encode_feature_array(1, 3, prev_action=0, prev_reward=-0.5),
        np.array([0.0, 1.0, -0.5], dtype=np.float32),
    )


@torch_required
def test_prev_reward_time_context_feature_encoding():
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    agent = get_agent(
        Agent.DQN,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        context_mode="prev_reward_time",
    )

    assert agent.feature_dim == agent.obs_dim + 2
    assert agent.feature_schema_components == (
        "observation_one_hot",
        "normalized_time_spent",
        "prev_reward",
    )
    np.testing.assert_allclose(
        agent.encode_feature_array(1, 3, prev_action=0, prev_reward=-0.5),
        np.array([0.0, 1.0, 3.0 / 99.0, -0.5], dtype=np.float32),
    )


@torch_required
@pytest.mark.parametrize("agent_type", [Agent.DQN, Agent.LSTM])
@pytest.mark.parametrize("context_mode", ["prev_reward", "prev_reward_time"])
def test_training_and_checkpoint_round_trip_support_new_context_modes(
    tmp_path: Path,
    agent_type: Agent,
    context_mode: str,
):
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5, horizon=20)
    agent = get_agent(
        agent_type,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        context_mode=context_mode,
    )

    run_dataset = agent.train(verbose=False)
    checkpoint_path = tmp_path / f"{agent_type.value}_{context_mode}.pt"
    agent.save_checkpoint(checkpoint_path)

    loaded_agent = get_agent(
        agent_type,
        MazePOMDP(load_builtin_maze_spec("full"), seed=5, horizon=20),
        num_episodes=1,
        device="cpu",
        seed=11,
        init_mode="pretrained",
        checkpoint_path=checkpoint_path,
        context_mode=context_mode,
    )

    assert run_dataset.num_episodes() == 1
    assert run_dataset.num_transitions() == 20
    assert loaded_agent.feature_schema_components == agent.feature_schema_components
    assert loaded_agent.feature_dim == agent.feature_dim
    assert loaded_agent.training_steps == agent.training_steps


@torch_required
def test_checkpoint_context_mode_mismatch_between_prev_reward_and_prev_reward_time(
    tmp_path: Path,
):
    maze = MazePOMDP(load_builtin_maze_spec("full"), seed=5)
    agent = get_agent(
        Agent.DQN,
        maze,
        num_episodes=1,
        device="cpu",
        seed=11,
        context_mode="prev_reward",
    )
    checkpoint_path = tmp_path / "prev_reward_dqn.pt"
    agent.save_checkpoint(checkpoint_path)

    with pytest.raises(ValueError, match="context_mode=prev_reward"):
        get_agent(
            Agent.DQN,
            MazePOMDP(load_builtin_maze_spec("full"), seed=5),
            num_episodes=1,
            device="cpu",
            seed=11,
            init_mode="pretrained",
            checkpoint_path=checkpoint_path,
            context_mode="prev_reward_time",
        )
