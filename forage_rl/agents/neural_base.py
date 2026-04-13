"""Shared neural-agent infrastructure for DQN-style agents."""

from __future__ import annotations

from abc import abstractmethod
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.base import ensure_time_spent_compatible
from forage_rl.agents.registry import Agent, NeuralContextMode, validate_context_mode
from forage_rl.config import DefaultParams
from forage_rl.environments import MazePOMDP
from forage_rl.utils.torch_support import require_torch, resolve_device

if TYPE_CHECKING:
    from forage_rl.environments import Maze


class NeuralContext(TypedDict):
    prev_action: int | None
    prev_reward: float


class NeuralAgentBase(BaseAgent):
    """Shared PyTorch-backed logic for neural agents."""

    agent_name: Agent
    feature_schema_components_by_context_mode: dict[NeuralContextMode, tuple[str, ...]] = {
        "observation_only": ("observation_one_hot",),
        "prev_reward": ("observation_one_hot", "prev_reward"),
        "prev_reward_time": (
            "observation_one_hot",
            "normalized_time_spent",
            "prev_reward",
        ),
        "legacy_context": (
            "observation_one_hot",
            "normalized_time_spent",
            "prev_reward",
            "prev_action_one_hot",
        ),
    }

    @staticmethod
    def _require_positive_int(name: str, value: int) -> None:
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _require_non_negative_int(name: str, value: int) -> None:
        if int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")

    @staticmethod
    def _require_positive_float(name: str, value: float) -> None:
        numeric_value = float(value)
        if (not np.isfinite(numeric_value)) or numeric_value <= 0.0:
            raise ValueError(f"{name} must be a finite positive float, got {value!r}.")

    @staticmethod
    def _resolve_learning_rate(
        *,
        alpha: float | None,
        learning_rate: float,
    ) -> float:
        resolved_learning_rate = float(learning_rate)
        if alpha is None:
            return resolved_learning_rate

        alpha_value = float(alpha)
        if (not np.isfinite(alpha_value)) or alpha_value <= 0.0:
            raise ValueError(
                f"alpha must be a finite positive float when provided, got {alpha!r}."
            )
        if np.isclose(alpha_value, resolved_learning_rate):
            return alpha_value
        if np.isclose(resolved_learning_rate, float(DefaultParams.LEARNING_RATE)):
            return alpha_value
        raise ValueError(
            "Neural agents treat alpha as a legacy alias for learning_rate; got "
            f"conflicting values alpha={alpha_value!r}, "
            f"learning_rate={resolved_learning_rate!r}."
        )

    def __init__(
        self,
        maze: "Maze",
        *,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        alpha: float | None = None,
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
        context_mode: NeuralContextMode = "legacy_context",
        seed: int | None = None,
    ):
        super().__init__(maze, beta=beta, seed=seed)
        ensure_time_spent_compatible(maze, consumer=type(self).__name__)
        resolved_learning_rate = self._resolve_learning_rate(
            alpha=alpha,
            learning_rate=learning_rate,
        )
        self._require_non_negative_int("num_episodes", num_episodes)
        self._require_positive_float("learning_rate", resolved_learning_rate)
        self._require_positive_int("batch_size", batch_size)
        self._require_positive_int("replay_capacity", replay_capacity)
        self._require_positive_int("target_update_interval", target_update_interval)
        self._require_positive_float("gradient_clip", gradient_clip)
        self.num_episodes = num_episodes
        self.alpha = resolved_learning_rate
        self.gamma = gamma
        self.learning_rate = resolved_learning_rate
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.target_update_interval = target_update_interval
        self.gradient_clip = gradient_clip
        self.init_mode = init_mode
        self.context_mode = validate_context_mode(context_mode)
        self.device = resolve_device(device)
        self.obs_dim = int(maze.observation_space.n)  # type: ignore[attr-defined]
        self.feature_schema_version = DefaultParams.NEURAL_FEATURE_SCHEMA_VERSION
        self.feature_schema_components = self._feature_components_for_context_mode(
            self.context_mode
        )
        action_dim = int(self.maze.num_actions)
        self.feature_dim = self._feature_dim_for_context_mode(action_dim)
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
            self._validate_checkpoint_horizon_metadata(
                resolved_path,
                requires_metadata=effective_path is None,
            )
            self.load_checkpoint(resolved_path)

    @abstractmethod
    def _build_model(self):
        """Build the online or target network."""

    def _default_checkpoint_path(self) -> Path:
        from forage_rl.utils.io import resolve_checkpoint_load_path

        maze_name = self.maze.maze_spec.maze.name
        observable = not isinstance(self.maze, MazePOMDP)
        return resolve_checkpoint_load_path(
            self.agent_name,
            maze_name,
            observable,
            context_mode=self.context_mode,
            horizon=self.maze.horizon,
        )

    def _validate_checkpoint_horizon_metadata(
        self,
        path: Path,
        *,
        requires_metadata: bool,
    ) -> None:
        """Reject pretrained checkpoints whose saved horizon differs from this maze."""
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            if requires_metadata:
                raise ValueError(
                    f"Checkpoint metadata is missing for {path.name}; regenerate pretrained "
                    "neural checkpoints with the current code."
                )
            return

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "horizon" not in metadata:
            raise ValueError(
                f"Checkpoint metadata at {metadata_path} is missing horizon and must be "
                "regenerated."
            )

        checkpoint_horizon = int(metadata["horizon"])
        if checkpoint_horizon != self.maze.horizon:
            raise ValueError(
                f"Checkpoint metadata at {metadata_path} uses horizon={checkpoint_horizon}, "
                f"but this agent expects horizon={self.maze.horizon}."
            )

    def _feature_components_for_context_mode(
        self,
        context_mode: NeuralContextMode,
    ) -> tuple[str, ...]:
        return self.feature_schema_components_by_context_mode[context_mode]

    def _feature_dim_for_context_mode(self, action_dim: int) -> int:
        component_widths = {
            "observation_one_hot": self.obs_dim,
            "normalized_time_spent": 1,
            "prev_reward": 1,
            "prev_action_one_hot": action_dim,
        }
        return int(
            sum(
                component_widths[component]
                for component in self.feature_schema_components
            )
        )

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
        feature_parts: list[np.ndarray] = []
        observation_feature = np.zeros(self.obs_dim, dtype=np.float32)
        observation_feature[int(state)] = 1.0
        feature_parts.append(observation_feature)

        horizon = max(self.maze.horizon - 1, 1)
        if "normalized_time_spent" in self.feature_schema_components:
            feature_parts.append(
                np.array(
                    [
                        float(min(time_spent, self.maze.horizon - 1))
                        / float(horizon)
                    ],
                    dtype=np.float32,
                )
            )
        if "prev_reward" in self.feature_schema_components:
            feature_parts.append(np.array([float(prev_reward)], dtype=np.float32))
        if "prev_action_one_hot" in self.feature_schema_components:
            feature_parts.append(self._encode_prev_action(prev_action))
        return np.concatenate(feature_parts, dtype=np.float32)

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
            "context_mode": self.context_mode,
        }

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
        checkpoint_context_mode = checkpoint.get("context_mode", "legacy_context")
        if schema_version is None:
            raise ValueError(
                f"Legacy checkpoint at {path} is incompatible with the current neural "
                "feature schema. Retrain pretrained neural checkpoints with "
                "`python -m forage_rl.experiments.train_pretrained_agents`."
            )
        if (
            int(schema_version) != self.feature_schema_version
            or int(feature_dim) != self.feature_dim
            or list(feature_components) != list(self.feature_schema_components)
            or checkpoint_context_mode != self.context_mode
        ):
            raise ValueError(
                f"Checkpoint at {path} uses neural feature schema "
                f"(version={schema_version}, feature_dim={feature_dim}, "
                f"components={feature_components}, context_mode={checkpoint_context_mode}) "
                f"but this agent expects "
                f"{self.feature_schema_metadata()}. Retrain pretrained neural "
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
