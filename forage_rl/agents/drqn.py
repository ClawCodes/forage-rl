"""Recurrent DRQN agent with an LSTM core."""

from __future__ import annotations

from collections import deque

import numpy as np

from forage_rl import RunDataset, TimedTransition, Trajectory
from forage_rl.agents.neural_base import NeuralAgentBase
from forage_rl.agents.registry import Agent
from forage_rl.config import DefaultParams


class DRQNNetwork:
    """Small wrapper around an LSTM and linear action head."""

    def __init__(self, nn, feature_dim: int, num_actions: int):
        self.model = nn.Module()
        self.model.lstm = nn.LSTM(feature_dim, 64, batch_first=True)
        self.model.head = nn.Linear(64, num_actions)

    def __call__(self, inputs, hidden=None):
        outputs, hidden = self.model.lstm(inputs, hidden)
        q_values = self.model.head(outputs)
        return q_values, hidden

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class DRQNAgent(NeuralAgentBase):
    """DRQN with episode replay over contiguous sequences."""

    agent_name = Agent.DRQN

    def __init__(
        self,
        maze,
        *,
        sequence_length: int = DefaultParams.DRQN_SEQUENCE_LENGTH,
        **kwargs,
    ):
        self.sequence_length = sequence_length
        super().__init__(maze, **kwargs)
        self.replay_buffer = deque(maxlen=self.replay_capacity)
        self.current_episode: list[dict[str, object]] = []
        self.hidden = None

    def _build_model(self):
        return DRQNNetwork(self.nn, self.feature_dim, self.maze.num_actions)

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

    def _sample_sequences(self) -> list[list[dict[str, object]]]:
        if not self.replay_buffer:
            return []

        sequences: list[list[dict[str, object]]] = []
        episode_indices = self.rng.integers(
            0,
            len(self.replay_buffer),
            size=self.batch_size,
        )
        for episode_index in episode_indices:
            episode = self.replay_buffer[int(episode_index)]
            if not episode:
                continue
            if len(episode) <= self.sequence_length:
                sequences.append(list(episode))
                continue
            start = int(self.rng.integers(0, len(episode) - self.sequence_length + 1))
            sequences.append(episode[start : start + self.sequence_length])
        return sequences

    def _build_batch_tensors(self, sequences: list[list[dict[str, object]]]):
        max_length = max(len(sequence) for sequence in sequences)
        batch_size = len(sequences)
        features = self.torch.zeros(
            (batch_size, max_length, self.feature_dim),
            dtype=self.torch.float32,
            device=self.torch_device,
        )
        next_features = self.torch.zeros_like(features)
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
        mask = self.torch.zeros(
            (batch_size, max_length),
            dtype=self.torch.float32,
            device=self.torch_device,
        )

        for row_index, sequence in enumerate(sequences):
            for step_index, item in enumerate(sequence):
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
                actions[row_index, step_index] = int(item["action"])
                rewards[row_index, step_index] = float(item["reward"])
                dones[row_index, step_index] = float(item["done"])
                mask[row_index, step_index] = 1.0

        return features, next_features, actions, rewards, dones, mask

    def _train_from_replay(self) -> None:
        if not self.replay_buffer:
            return

        sequences = self._sample_sequences()
        if not sequences:
            return

        features, next_features, actions, rewards, dones, mask = self._build_batch_tensors(
            sequences
        )
        q_values, _ = self.q_network(features)
        selected_q = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with self.torch.no_grad():
            target_q_values, _ = self.target_network(next_features)
            next_max = target_q_values.max(dim=2).values
            targets = rewards + self.gamma * (1.0 - dones) * next_max

        losses = self.loss_fn(selected_q, targets) * mask
        denom = self.torch.clamp(mask.sum(), min=1.0)
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
        action, _ = self.action_from_q_values(q_values)
        return action

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Evaluate log-likelihood of one episode under online DRQN updates."""
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
            action_probs = self.boltzmann_action_probs(q_values.detach().cpu().numpy())
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
        """Train the DRQN agent and return one run dataset."""
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
                print(f"Episode {episode_idx}, recurrent_replay={len(self.replay_buffer)}")

        return RunDataset(trajectories=trajectories)
