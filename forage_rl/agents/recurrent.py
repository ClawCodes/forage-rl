"""Shared recurrent Q-learning agents for Elman, GRU, and LSTM cells."""

from __future__ import annotations

from collections import deque
from typing import TypedDict

import numpy as np

from forage_rl.types import RunDataset, TimedTransition, Trajectory
from forage_rl.agents.neural_base import NeuralAgentBase
from forage_rl.config import DefaultParams


class RecurrentQNetwork:
    """Small recurrent Q-network with a recurrent core and linear action head."""

    _CORE_MAP = {
        "elman": "RNN",
        "gru": "GRU",
        "lstm": "LSTM",
    }
    _CORE_ATTR = {
        "elman": "rnn",
        "gru": "gru",
        "lstm": "lstm",
    }
    _MODULE_TYPES: dict[int, type] = {}

    @classmethod
    def _build_module_type(cls, nn) -> type:
        cache_key = id(nn)
        module_type = cls._MODULE_TYPES.get(cache_key)
        if module_type is not None:
            return module_type

        outer_cls = cls

        class RecurrentQNetworkModule(nn.Module):
            def __init__(
                self,
                feature_dim: int,
                num_actions: int,
                *,
                recurrent_core: str,
                hidden_size: int,
                num_layers: int,
            ):
                super().__init__()
                if recurrent_core not in outer_cls._CORE_MAP:
                    raise ValueError(
                        f"Unsupported recurrent core {recurrent_core!r}; expected one of "
                        f"{', '.join(sorted(outer_cls._CORE_MAP))}."
                    )

                recurrent_cls = getattr(nn, outer_cls._CORE_MAP[recurrent_core])
                recurrent = recurrent_cls(
                    feature_dim,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.recurrent = recurrent
                setattr(self, outer_cls._CORE_ATTR[recurrent_core], recurrent)
                self.head = nn.Linear(hidden_size, num_actions)

            def forward(self, inputs, hidden=None):
                outputs, hidden = self.recurrent(inputs, hidden)
                q_values = self.head(outputs)
                return q_values, hidden

        cls._MODULE_TYPES[cache_key] = RecurrentQNetworkModule
        return RecurrentQNetworkModule

    def __new__(
        cls,
        nn,
        feature_dim: int,
        num_actions: int,
        *,
        recurrent_core: str,
        hidden_size: int,
        num_layers: int,
    ):
        module_type = cls._build_module_type(nn)
        return module_type(
            feature_dim,
            num_actions,
            recurrent_core=recurrent_core,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )


class SampledRecurrentWindow(TypedDict):
    """Contiguous replay window with the first loss-bearing timestep index."""

    steps: list[dict[str, object]]
    loss_start_index: int


class RecurrentQAgent(NeuralAgentBase):
    """Shared recurrent Q-agent with sequence replay over full episodes."""

    recurrent_core: str

    def __init__(
        self,
        maze,
        *,
        sequence_length: int = DefaultParams.RECURRENT_SEQUENCE_LENGTH,
        burn_in: int = DefaultParams.RECURRENT_BURN_IN,
        recurrent_hidden_size: int = DefaultParams.RECURRENT_HIDDEN_SIZE,
        recurrent_num_layers: int = DefaultParams.RECURRENT_NUM_LAYERS,
        **kwargs,
    ):
        self._require_positive_int("sequence_length", sequence_length)
        self._require_non_negative_int("burn_in", burn_in)
        self._require_positive_int("recurrent_hidden_size", recurrent_hidden_size)
        self._require_positive_int("recurrent_num_layers", recurrent_num_layers)
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_num_layers = recurrent_num_layers
        super().__init__(maze, **kwargs)
        self.replay_buffer = deque(maxlen=self.replay_capacity)
        self.current_episode: list[dict[str, object]] = []
        self.hidden = None

    def _build_model(self):
        return RecurrentQNetwork(
            self.nn,
            self.feature_dim,
            self.maze.num_actions,
            recurrent_core=self.recurrent_core,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_num_layers,
        )

    def _predict_q_values(self, model, feature, hidden=None):
        q_values, hidden = model(feature.view(1, 1, -1), hidden)
        return q_values.squeeze(0).squeeze(0), hidden

    def _store_step(
        self,
        state: int,
        time_spent: int,
        prev_action: int | None,
        prev_reward: float,
        action: int,
        reward: float,
        next_state: int,
        next_time_spent: int,
        done: bool,
    ) -> None:
        self.current_episode.append(
            {
                "state": int(state),
                "time_spent": int(time_spent),
                "prev_action": None if prev_action is None else int(prev_action),
                "prev_reward": float(prev_reward),
                "action": int(action),
                "reward": float(reward),
                "next_state": int(next_state),
                "next_time_spent": int(next_time_spent),
                "done": bool(done),
            }
        )

    def _finalize_episode(self) -> None:
        if self.current_episode:
            self.replay_buffer.append(list(self.current_episode))
            self.current_episode = []
        self.hidden = None

    def _replay_sources(self) -> list[list[dict[str, object]]]:
        """Return finalized episodes plus the live episode prefix when available."""
        episodes = list(self.replay_buffer)
        if self.current_episode:
            episodes.append(self.current_episode)
        return episodes

    def _sample_window_from_episode(
        self,
        episode: list[dict[str, object]],
        *,
        learn_start: int | None = None,
    ) -> SampledRecurrentWindow:
        """Return one contiguous replay window plus the post-burn-in loss start."""
        if not episode:
            return {"steps": [], "loss_start_index": 0}

        learn_len = self.sequence_length
        episode_len = len(episode)
        if episode_len <= learn_len:
            sampled_learn_start = 0
            learn_end = episode_len
        else:
            max_learn_start = episode_len - learn_len
            if learn_start is None:
                sampled_learn_start = int(self.rng.integers(0, max_learn_start + 1))
            else:
                if learn_start < 0 or learn_start > max_learn_start:
                    raise ValueError(
                        "learn_start must index a full learn window within the episode, "
                        f"got learn_start={learn_start}, episode_len={episode_len}, "
                        f"sequence_length={learn_len}."
                    )
                sampled_learn_start = int(learn_start)
            learn_end = sampled_learn_start + learn_len

        context_start = max(0, sampled_learn_start - self.burn_in)
        return {
            "steps": list(episode[context_start:learn_end]),
            "loss_start_index": sampled_learn_start - context_start,
        }

    def _sample_windows(self) -> list[SampledRecurrentWindow]:
        replay_sources = self._replay_sources()
        if not replay_sources:
            return []

        windows: list[SampledRecurrentWindow] = []
        episode_indices = self.rng.integers(
            0,
            len(replay_sources),
            size=self.batch_size,
        )
        for episode_index in episode_indices:
            episode = replay_sources[int(episode_index)]
            if not episode:
                continue
            windows.append(self._sample_window_from_episode(episode))
        return windows

    def _build_batch_tensors(
        self,
        windows: list[SampledRecurrentWindow] | list[list[dict[str, object]]],
        *,
        include_masks: bool = False,
    ):
        normalized_windows = [
            window
            if isinstance(window, dict)
            else {"steps": window, "loss_start_index": 0}
            for window in windows
        ]
        max_length = max(len(window["steps"]) for window in normalized_windows)
        batch_size = len(normalized_windows)
        features = self.torch.zeros(
            (batch_size, max_length, self.feature_dim),
            dtype=self.torch.float32,
            device=self.torch_device,
        )
        next_features = self.torch.zeros_like(features)
        next_state_ids = self.torch.zeros(
            (batch_size, max_length),
            dtype=self.torch.long,
            device=self.torch_device,
        )
        actions = self.torch.zeros(
            (batch_size, max_length),
            dtype=self.torch.long,
            device=self.torch_device,
        )
        rewards = self.torch.zeros(
            (batch_size, max_length),
            dtype=self.torch.float32,
            device=self.torch_device,
        )
        dones = self.torch.zeros(
            (batch_size, max_length),
            dtype=self.torch.float32,
            device=self.torch_device,
        )
        valid_mask = self.torch.zeros(
            (batch_size, max_length),
            dtype=self.torch.float32,
            device=self.torch_device,
        )
        loss_mask = self.torch.zeros_like(valid_mask)

        for row_index, window in enumerate(normalized_windows):
            steps = window["steps"]
            loss_start_index = int(window["loss_start_index"])
            for step_index, item in enumerate(steps):
                features[row_index, step_index] = self.encode_feature_tensor(
                    int(item["state"]),
                    int(item["time_spent"]),
                    prev_action=item["prev_action"],
                    prev_reward=float(item["prev_reward"]),
                )
                next_features[row_index, step_index] = self.encode_feature_tensor(
                    int(item["next_state"]),
                    int(item["next_time_spent"]),
                    prev_action=int(item["action"]),
                    prev_reward=float(item["reward"]),
                )
                next_state_ids[row_index, step_index] = int(item["next_state"])
                actions[row_index, step_index] = int(item["action"])
                rewards[row_index, step_index] = float(item["reward"])
                dones[row_index, step_index] = float(item["done"])
                valid_mask[row_index, step_index] = 1.0
                if step_index >= loss_start_index:
                    loss_mask[row_index, step_index] = 1.0

        if include_masks:
            return (
                features,
                next_features,
                next_state_ids,
                actions,
                rewards,
                dones,
                valid_mask,
                loss_mask,
            )
        return (
            features,
            next_features,
            next_state_ids,
            actions,
            rewards,
            dones,
            loss_mask,
        )

    def _train_from_replay(self) -> None:
        if self.training_steps < self.batch_size:
            return

        windows = self._sample_windows()
        if not windows:
            return

        (
            features,
            next_features,
            next_state_ids,
            actions,
            rewards,
            dones,
            _valid_mask,
            loss_mask,
        ) = self._build_batch_tensors(windows, include_masks=True)
        q_values, _ = self.q_network(features)
        selected_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with self.torch.no_grad():
            target_q_values, _ = self.target_network(next_features)
            next_max = self.masked_max_q_values(
                target_q_values,
                next_state_ids.detach().cpu().numpy(),
            )
            targets = rewards + self.gamma * (1.0 - dones) * next_max

        if float(loss_mask.sum().item()) <= 0.0:
            return

        losses = self.loss_fn(selected_q, targets) * loss_mask
        denom = self.torch.clamp(loss_mask.sum(), min=1.0)
        loss = losses.sum() / denom
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.optimizer.step()

    def choose_action(
        self,
        state: int,
        time_spent: int,
        prev_action: int | None = None,
        prev_reward: float = 0.0,
    ) -> int:
        """Choose an action using the recurrent policy state."""
        feature = self.encode_feature_tensor(
            state,
            time_spent,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        q_values, self.hidden = self.q_values_for_feature(feature, self.hidden)
        action, _ = self.action_from_q_values(q_values, state)
        return action

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Evaluate one episode while preserving online recurrent learning state.

        This resets within-episode hidden state, but still appends to replay and
        updates network weights so repeated calls form a continuing evaluator.
        """
        log_likelihoods: list[float] = []

        self.hidden = None
        self.current_episode = []
        context = self.initial_context()
        for step_index, transition in enumerate(trajectory.transitions):
            time_spent = getattr(transition, "time_spent", 0)
            feature = self.encode_feature_tensor(
                transition.state,
                time_spent,
                prev_action=context["prev_action"],
                prev_reward=context["prev_reward"],
            )
            q_values, self.hidden = self.q_values_for_feature(feature, self.hidden)
            action_probs = self.action_probabilities_for_state(
                q_values,
                transition.state,
            )
            log_likelihoods.append(float(np.log(action_probs[transition.action])))

            next_time_spent = (
                min(time_spent + 1, self.maze.horizon - 1)
                if transition.state == transition.next_state
                else 0
            )
            self._store_step(
                transition.state,
                time_spent,
                context["prev_action"],
                context["prev_reward"],
                transition.action,
                transition.reward,
                transition.next_state,
                next_time_spent,
                step_index == len(trajectory.transitions) - 1,
            )
            self.training_steps += 1
            self._train_from_replay()
            self.maybe_sync_target_network()
            context = self.next_context(transition.action, transition.reward)

        self._finalize_episode()

        return log_likelihoods

    def train(self, verbose: bool = True) -> RunDataset:
        """Train the recurrent agent and return one run dataset."""
        trajectories: list[Trajectory[TimedTransition]] = []

        for episode_idx in range(self.num_episodes):
            state, _ = self.maze.reset()
            self.hidden = None
            self.current_episode = []
            time_spent = 0
            context = self.initial_context()
            done = False
            episode_transitions: list[TimedTransition] = []

            while not done:
                action = self.choose_action(
                    state,
                    time_spent,
                    prev_action=context["prev_action"],
                    prev_reward=context["prev_reward"],
                )
                transition, done = self.maze.step_transition(action)
                timed_transition = TimedTransition.from_transition_time(
                    transition,
                    time_spent,
                )
                episode_transitions.append(timed_transition)

                next_time_spent = (
                    min(time_spent + 1, self.maze.horizon - 1)
                    if state == timed_transition.next_state
                    else 0
                )
                self._store_step(
                    timed_transition.state,
                    time_spent,
                    context["prev_action"],
                    context["prev_reward"],
                    timed_transition.action,
                    timed_transition.reward,
                    timed_transition.next_state,
                    next_time_spent,
                    done,
                )
                self.training_steps += 1
                self._train_from_replay()
                self.maybe_sync_target_network()

                state = timed_transition.next_state
                time_spent = next_time_spent
                context = self.next_context(
                    timed_transition.action,
                    timed_transition.reward,
                )

            self._finalize_episode()
            trajectories.append(Trajectory(transitions=episode_transitions))
            if verbose and episode_idx % 100 == 0:
                print(
                    f"Episode {episode_idx}, recurrent_replay={len(self.replay_buffer)}"
                )

        return RunDataset(trajectories=trajectories)


class ElmanAgent(RecurrentQAgent):
    """Recurrent Q-agent with a simple Elman RNN core."""

    agent_name = "elman"
    recurrent_core = "elman"


class GRUAgent(RecurrentQAgent):
    """Recurrent Q-agent with a GRU core."""

    agent_name = "gru"
    recurrent_core = "gru"


class LSTMAgent(RecurrentQAgent):
    """Recurrent Q-agent with an LSTM core."""

    agent_name = "lstm"
    recurrent_core = "lstm"
