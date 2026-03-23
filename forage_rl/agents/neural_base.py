"""Shared neural-agent infrastructure for DQN-style agents."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.registry import Agent
from forage_rl.config import DefaultParams
from forage_rl.environments import MazePOMDP
from forage_rl.utils.io import checkpoint_path
from forage_rl.utils.torch_support import require_torch, resolve_device

if TYPE_CHECKING:
    from forage_rl.environments import Maze
    from forage_rl.types import Trajectory


class NeuralContext(TypedDict):
    prev_action: int | None
    prev_reward: float


class NeuralAgentBase(BaseAgent):
    """Shared PyTorch-backed logic for neural agents."""

    agent_name: Agent
    feature_schema_components = (
        "observation_one_hot",
        "normalized_time_spent",
        "prev_reward",
        "prev_action_one_hot",
    )

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
        self.obs_dim = int(maze.observation_space.n)  # type: ignore[attr-defined]
        self.feature_schema_version = DefaultParams.NEURAL_FEATURE_SCHEMA_VERSION
        action_dim = int(self.maze.num_actions)
        self.feature_dim = int(self.obs_dim + 1 + 1 + action_dim)
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

    def _encode_prev_action(self, prev_action: int | None) -> np.ndarray:
        """Encode the previous action as a one-hot vector or zeros at episode start."""
        action_feature = np.zeros(self.maze.num_actions, dtype=np.float32)
        if prev_action is None:
            return action_feature
        action_feature[int(prev_action)] = 1.0
        return action_feature

    def encode_feature_array(
        self,
        state: int,
        time_spent: int,
        prev_action: int | None = None,
        prev_reward: float = 0.0,
    ) -> np.ndarray:
        """Encode the full neural-agent context into a feature array."""
        feature = np.zeros(self.feature_dim, dtype=np.float32)
        feature[int(state)] = 1.0
        horizon = max(self.maze.horizon - 1, 1)
        feature[self.obs_dim] = float(min(time_spent, self.maze.horizon - 1)) / float(
            horizon
        )
        feature[self.obs_dim + 1] = float(prev_reward)
        feature[self.obs_dim + 2 :] = self._encode_prev_action(prev_action)
        return feature

    def encode_feature_tensor(
        self,
        state: int,
        time_spent: int,
        prev_action: int | None = None,
        prev_reward: float = 0.0,
    ):
        """Encode an observation and context into a network input tensor."""
        feature = self.encode_feature_array(
            state,
            time_spent,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        return self.torch.tensor(feature, device=self.torch_device)

    def initial_context(self) -> NeuralContext:
        """Return the deterministic episode-start context."""
        return {"prev_action": None, "prev_reward": 0.0}

    def next_context(self, action: int, reward: float) -> NeuralContext:
        """Return the context that should be used on the next decision step."""
        return {"prev_action": int(action), "prev_reward": float(reward)}

    def feature_schema_metadata(self) -> dict[str, object]:
        """Return metadata describing the current neural input schema."""
        return {
            "feature_schema_version": int(self.feature_schema_version),
            "feature_dim": int(self.feature_dim),
            "feature_components": list(self.feature_schema_components),
        }

    def context_trace(self, trajectory: "Trajectory") -> list[dict[str, object]]:
        """Return the exact per-step context features used for one episode."""
        rows: list[dict[str, object]] = []
        context = self.initial_context()
        for step_index, transition in enumerate(trajectory.transitions):
            time_spent = getattr(transition, "time_spent", 0)
            encoded_feature = self.encode_feature_array(
                transition.state,
                time_spent,
                prev_action=context["prev_action"],
                prev_reward=context["prev_reward"],
            )
            rows.append(
                {
                    "step_index": step_index,
                    "state": int(transition.state),
                    "time_spent": int(time_spent),
                    "prev_action": context["prev_action"],
                    "prev_reward": float(context["prev_reward"]),
                    "action": int(transition.action),
                    "reward": float(transition.reward),
                    "next_state": int(transition.next_state),
                    "encoded_feature": encoded_feature,
                }
            )
            context = self.next_context(transition.action, transition.reward)
        return rows

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
        checkpoint = self.torch.load(
            path,
            map_location=self.torch_device,
            weights_only=False,
        )
        schema_version = checkpoint.get("feature_schema_version")
        feature_dim = checkpoint.get("feature_dim")
        feature_components = checkpoint.get("feature_components")
        if schema_version is None:
            raise ValueError(
                f"Legacy checkpoint at {path} is incompatible with the current neural "
                "feature schema. Retrain pretrained DQN/DRQN checkpoints with "
                "`python -m forage_rl.experiments.train_pretrained_agents`."
            )
        if (
            int(schema_version) != self.feature_schema_version
            or int(feature_dim) != self.feature_dim
            or list(feature_components) != list(self.feature_schema_components)
        ):
            raise ValueError(
                f"Checkpoint at {path} uses neural feature schema "
                f"(version={schema_version}, feature_dim={feature_dim}, "
                f"components={feature_components}) but this agent expects "
                f"{self.feature_schema_metadata()}. Retrain pretrained DQN/DRQN "
                "checkpoints with the current code."
            )
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
                **self.feature_schema_metadata(),
            },
            path,
        )
