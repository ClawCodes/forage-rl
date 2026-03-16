"""Shared Torch-based infrastructure for DQN-style agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from forage_rl import Trajectory
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze

from .base import BaseAgent


TReplaySample = TypeVar("TReplaySample")


def resolve_torch_device(
    requested_device: str | torch.device | None = "auto",
) -> torch.device:
    """Resolve a user-facing device selector to a validated torch device."""
    if isinstance(requested_device, torch.device):
        requested = str(requested_device)
    else:
        requested = (
            "auto" if requested_device is None else str(requested_device).strip()
        )

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available on this machine.")
        device = torch.device(requested)
        device_index = 0 if device.index is None else device.index
        cuda_device_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= cuda_device_count:
            raise ValueError(
                f"CUDA device index {device_index} is invalid; "
                f"available device count is {cuda_device_count}."
            )
        return torch.device("cuda", device_index)

    raise ValueError(
        "device must be one of 'auto', 'cpu', 'cuda', or 'cuda:N'; "
        f"got {requested_device!r}"
    )


@dataclass(frozen=True)
class _TransitionSample:
    """Single transition sample for non-recurrent updates."""

    state_features: torch.Tensor
    action: int
    reward: float
    next_state_features: torch.Tensor
    done: bool


@dataclass(frozen=True)
class _SequenceTransitionSample:
    """Single transition sample for recurrent sequence-based updates."""

    state_sequence: torch.Tensor
    action: int
    reward: float
    next_state_sequence: torch.Tensor
    done: bool


class _ReplayBuffer(Generic[TReplaySample]):
    """Fixed-capacity replay buffer for transition samples."""

    def __init__(self, capacity: int):
        self._buffer: deque[TReplaySample] = deque(maxlen=capacity)

    def add(self, sample: TReplaySample) -> None:
        self._buffer.append(sample)

    def sample(self, rng: np.random.Generator, batch_size: int) -> list[TReplaySample]:
        if batch_size > len(self._buffer):
            raise ValueError(
                "Cannot sample more transitions than are stored: "
                f"requested batch_size={batch_size}, available={len(self._buffer)}"
            )
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class _MLPQNet(nn.Module):
    """Feed-forward Q-network for state-feature to action-value mapping."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.layers(x)


class _LSTMQNet(nn.Module):
    """LSTM-based Q-network for sequence-to-action-value mapping."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_actions)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        outputs, _ = self.lstm(sequence)
        return self.output(outputs[:, -1, :])


class TorchQAgentBase(BaseAgent, ABC):
    """Shared Torch-based base class for DQN-style agents."""

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        epsilon: float = DefaultParams.EPSILON,
        learning_rate: float = DefaultParams.DQN_LEARNING_RATE,
        hidden_dim: int = DefaultParams.DQN_HIDDEN_DIM,
        seed: int | None = None,
        device: str | torch.device | None = "auto",
    ):
        super().__init__(maze=maze, seed=seed)
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.device = resolve_torch_device(device)
        self.log_floor = 1e-12
        self.policy_net: nn.Module = nn.Identity()
        self.target_net: nn.Module = nn.Identity()
        self.optimizer: torch.optim.Optimizer
        self.target_update_interval = 1
        self.update_steps = 0

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.observation_dim = self.maze.agent_num_states
        self.input_dim = self.observation_dim + 1 + maze.num_actions + 1

    def get_policy(self) -> np.ndarray:
        raise NotImplementedError(
            "DQN agents do not expose a tabular q_table policy; "
            "use network predictions instead."
        )

    def print_policy(self, max_time_to_display: int = 6) -> None:
        del max_time_to_display
        raise NotImplementedError(
            "DQN agents do not expose a tabular q_table policy; "
            "use network predictions instead."
        )

    def _encode_features_tensor(
        self,
        state: int,
        time_spent: int,
        prev_action: int,
        prev_reward: float,
    ) -> torch.Tensor:
        observation_vector = F.one_hot(
            torch.tensor(state, device=self.device),
            num_classes=self.observation_dim,
        ).to(torch.float32)

        time_denom = max(self.maze.horizon - 1, 1)
        time_feature = torch.tensor(
            [float(time_spent) / float(time_denom)],
            dtype=torch.float32,
            device=self.device,
        )

        action_vector = F.one_hot(
            torch.tensor(prev_action, device=self.device),
            num_classes=self.maze.num_actions,
        ).to(torch.float32)

        reward_feature = torch.tensor(
            [prev_reward], dtype=torch.float32, device=self.device
        )

        return torch.cat(
            [observation_vector, time_feature, action_vector, reward_feature]
        )

    def _encode_features(
        self,
        state: int,
        time_spent: int,
        prev_action: int,
        prev_reward: float,
    ) -> np.ndarray:
        """Compatibility wrapper for tests and non-performance-sensitive callers."""
        return (
            self._encode_features_tensor(
                state=state,
                time_spent=time_spent,
                prev_action=prev_action,
                prev_reward=prev_reward,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _next_time_spent(
        self,
        *,
        state: int,
        next_state: int,
        time_spent: int,
        action: int,
        true_state: int | None = None,
        true_next_state: int | None = None,
    ) -> int:
        return self.maze.next_time_spent(
            state=state,
            next_state=next_state,
            time_spent=time_spent,
            action=action,
            true_state=true_state,
            true_next_state=true_next_state,
        )

    def _epsilon_greedy_probs(self, q_values: torch.Tensor) -> torch.Tensor:
        q_values = q_values.reshape(-1)
        probs = torch.full(
            (self.maze.num_actions,),
            self.epsilon / self.maze.num_actions,
            dtype=torch.float32,
            device=self.device,
        )
        greedy_actions = torch.isclose(q_values, torch.max(q_values))
        greedy_bonus = (1.0 - self.epsilon) / max(int(greedy_actions.sum().item()), 1)
        probs = probs + greedy_actions.to(torch.float32) * greedy_bonus
        return probs

    def _choose_action_epsilon_greedy(self, q_values: torch.Tensor) -> int:
        probs = self._epsilon_greedy_probs(q_values)
        return int(
            self.rng.choice(
                self.maze.num_actions,
                p=probs.detach().cpu().numpy(),
            )
        )

    def _log_action_probability(self, q_values: torch.Tensor, action: int) -> float:
        probs = self._epsilon_greedy_probs(q_values)
        return float(
            torch.log(
                torch.clamp(
                    probs[action],
                    min=torch.tensor(
                        self.log_floor,
                        dtype=probs.dtype,
                        device=probs.device,
                    ),
                )
            ).item()
        )

    def _make_optimizer(self, policy_net: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate)

    def _initialize_target_network(
        self,
        policy_net: nn.Module,
        network_factory: Callable[[], nn.Module],
    ) -> nn.Module:
        target_net = network_factory().to(self.device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        return target_net

    def _sync_target_network(self) -> None:
        if self.update_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _td_optimize_tensors(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        q_values = self.policy_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q_max

        loss = F.mse_loss(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        self._sync_target_network()

    @abstractmethod
    def _learn_from_sample(self, sample: object) -> None:
        """Apply one learning update from a transition sample."""

    @abstractmethod
    def _predict_q(self, encoded_input: torch.Tensor) -> torch.Tensor:
        """Predict Q-values for a feature vector or sequence."""


__all__ = [
    "TorchQAgentBase",
    "_LSTMQNet",
    "_MLPQNet",
    "_ReplayBuffer",
    "_SequenceTransitionSample",
    "_TransitionSample",
]
