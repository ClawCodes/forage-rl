"""Shared neural-agent infrastructure for DQN-style agents."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.registry import Agent
from forage_rl.config import DefaultParams
from forage_rl.environments import MazePOMDP
from forage_rl.utils.io import checkpoint_path
from forage_rl.utils.torch_support import require_torch, resolve_device

if TYPE_CHECKING:
    from forage_rl.environments import Maze


class NeuralAgentBase(BaseAgent):
    """Shared PyTorch-backed logic for neural agents."""

    agent_name: Agent

    def __init__(
        self,
        maze: "Maze",
        *,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        alpha: float = DefaultParams.ALPHA,
        gamma: float = DefaultParams.GAMMA,
        beta: float = DefaultParams.BETA,
        learning_rate: float = DefaultParams.LEARNING_RATE,
        batch_size: int = DefaultParams.BATCH_SIZE,
        replay_capacity: int = DefaultParams.REPLAY_CAPACITY,
        target_update_interval: int = DefaultParams.TARGET_UPDATE_INTERVAL,
        gradient_clip: float = DefaultParams.GRADIENT_CLIP,
        device: str = "auto",
        init_mode: str = "fresh",
        checkpoint_path_override: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        seed: int | None = None,
    ):
        super().__init__(maze, beta=beta, seed=seed)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.target_update_interval = target_update_interval
        self.gradient_clip = gradient_clip
        self.init_mode = init_mode
        self.device = resolve_device(device)
        self.obs_dim = maze.observation_space.n  # type: ignore[attr-defined]
        self.feature_dim = self.obs_dim + 1
        self.training_steps = 0

        self.torch = require_torch()
        self.nn = self.torch.nn
        self.optim = self.torch.optim
        self.torch_device = self.torch.device(self.device)
        self.loss_fn = self.nn.SmoothL1Loss(reduction="none")

        torch_seed = DefaultParams.FRESH_EVALUATOR_SEED if seed is None else seed
        self.torch.manual_seed(torch_seed)
        if self.device == "cuda":
            self.torch.cuda.manual_seed_all(torch_seed)

        self.q_network = self._build_model().to(self.torch_device)
        self.target_network = self._build_model().to(self.torch_device)
        self.optimizer = self.optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate,
        )
        self.sync_target_network()

        effective_path = checkpoint_path_override or checkpoint_path
        if self.init_mode == "pretrained":
            resolved_path = self._default_checkpoint_path() if effective_path is None else Path(effective_path)
            self.load_checkpoint(resolved_path)

    @abstractmethod
    def _build_model(self):
        """Build the online or target network."""

    def _default_checkpoint_path(self) -> Path:
        maze_name = self.maze.maze_spec.maze.name
        observable = not isinstance(self.maze, MazePOMDP)
        return checkpoint_path(self.agent_name, maze_name, observable)

    def encode_feature_tensor(self, state: int, time_spent: int):
        """Encode an observation and elapsed time into a network input tensor."""
        feature = np.zeros(self.feature_dim, dtype=np.float32)
        feature[int(state)] = 1.0
        horizon = max(self.maze.horizon - 1, 1)
        feature[-1] = float(min(time_spent, self.maze.horizon - 1)) / float(horizon)
        return self.torch.tensor(feature, device=self.torch_device)

    def sync_target_network(self) -> None:
        """Copy online-network weights into the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def maybe_sync_target_network(self) -> None:
        """Update the target network on the configured cadence."""
        if self.training_steps % self.target_update_interval == 0:
            self.sync_target_network()

    def clip_gradients(self) -> None:
        """Apply gradient clipping when training neural agents."""
        self.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)

    def q_values_for_feature(self, feature, hidden: Any = None):
        """Run the online network on a single feature vector."""
        with self.torch.no_grad():
            return self._predict_q_values(self.q_network, feature, hidden)

    @abstractmethod
    def _predict_q_values(self, model, feature, hidden: Any = None):
        """Return q-values (and optional hidden state) for a single feature vector."""

    def action_from_q_values(self, q_values) -> tuple[int, float]:
        """Sample an action and return its log-probability under Boltzmann policy."""
        q_numpy = q_values.detach().cpu().numpy()
        action_probs = self.boltzmann_action_probs(q_numpy)
        action = int(self.rng.choice(len(q_numpy), p=action_probs))
        return action, float(np.log(action_probs[action]))

    def load_checkpoint(self, path: Path) -> None:
        """Load a pretrained checkpoint into the online and target networks."""
        checkpoint = self.torch.load(path, map_location=self.torch_device)
        self.q_network.load_state_dict(checkpoint["model_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = int(checkpoint.get("training_steps", 0))

    def save_checkpoint(self, path: Path) -> None:
        """Persist online-network training state for warm-started evaluators."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "model_state_dict": self.q_network.state_dict(),
                "target_model_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            path,
        )
