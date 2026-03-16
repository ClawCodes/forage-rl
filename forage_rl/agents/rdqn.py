"""Recurrent DQN agent using replay and a target network."""

from __future__ import annotations

from collections import deque

import torch

from forage_rl import ObservedTimedTransition, TimedTransition, Trajectory
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze

from .torch_q import (
    TorchQAgentBase,
    _LSTMQNet,
    _ReplayBuffer,
    _SequenceTransitionSample,
)


class RDQNAgent(TorchQAgentBase):
    """LSTM-based recurrent DQN agent with replay and a target network."""

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        epsilon: float = DefaultParams.EPSILON,
        learning_rate: float = DefaultParams.DQN_LEARNING_RATE,
        hidden_dim: int = DefaultParams.DQN_HIDDEN_DIM,
        sequence_length: int = DefaultParams.RDQN_SEQUENCE_LENGTH,
        replay_size: int = DefaultParams.DQN_REPLAY_SIZE,
        batch_size: int = DefaultParams.RDQN_BATCH_SIZE,
        target_update_interval: int = DefaultParams.DQN_TARGET_UPDATE_INTERVAL,
        replay_warmup: int = DefaultParams.DQN_REPLAY_WARMUP,
        seed: int | None = None,
        device: str | torch.device | None = "auto",
    ):
        super().__init__(
            maze=maze,
            num_episodes=num_episodes,
            gamma=gamma,
            epsilon=epsilon,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            seed=seed,
            device=device,
        )

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.replay_warmup = replay_warmup
        self.update_steps = 0

        self.policy_net = _LSTMQNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_actions=self.maze.num_actions,
        ).to(self.device)
        self.optimizer = self._make_optimizer(self.policy_net)
        self.target_net = self._initialize_target_network(
            self.policy_net,
            lambda: _LSTMQNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_actions=self.maze.num_actions,
            ),
        )
        self.replay_buffer = _ReplayBuffer[_SequenceTransitionSample](replay_size)

    def _sequence_from_history(self, history: deque[torch.Tensor]) -> torch.Tensor:
        sequence = torch.zeros(
            (self.sequence_length, self.input_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if not history:
            return sequence

        recent = list(history)[-self.sequence_length :]
        recent_tensor = torch.stack(recent)
        sequence[-recent_tensor.shape[0] :] = recent_tensor
        return sequence

    def _predict_q(self, encoded_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(encoded_input).squeeze(0)

    def _td_update_batch(self, batch: list[_SequenceTransitionSample]) -> None:
        states = torch.stack([sample.state_sequence for sample in batch])
        actions = torch.as_tensor(
            [sample.action for sample in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [sample.reward for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.stack([sample.next_state_sequence for sample in batch])
        dones = torch.as_tensor(
            [sample.done for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        self._td_optimize_tensors(states, actions, rewards, next_states, dones)

    def _learn_from_sample(self, sample: object) -> None:
        if not isinstance(sample, _SequenceTransitionSample):
            raise TypeError("RDQNAgent expects _SequenceTransitionSample inputs")

        self.replay_buffer.add(sample)
        if len(self.replay_buffer) < max(self.replay_warmup, self.batch_size):
            return

        replay_batch = self.replay_buffer.sample(self.rng, self.batch_size)
        self._td_update_batch(replay_batch)

    def train(self, verbose: bool = True) -> Trajectory:
        transitions: list[TimedTransition] = []

        for episode in range(self.num_episodes):
            state, _ = self.maze.reset()
            time_spent = 0
            prev_action = 0
            prev_reward = 0.0
            done = False
            history: deque[torch.Tensor] = deque(maxlen=self.sequence_length)
            history.append(
                self._encode_features_tensor(
                    state=state,
                    time_spent=time_spent,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
            )

            while not done:
                state_sequence = self._sequence_from_history(history)
                q_values = self._predict_q(state_sequence)
                action = self._choose_action_epsilon_greedy(q_values)

                details = self.maze.step_transition_details(action)
                if self.maze.agent_num_states == self.maze.true_num_states:
                    transitions.append(
                        TimedTransition.from_transition_time(
                            transition=details.transition,
                            time=time_spent,
                            done=details.done,
                        )
                    )
                else:
                    transitions.append(
                        ObservedTimedTransition.from_transition_time_truth(
                            transition=details.transition,
                            time=time_spent,
                            true_state=details.true_state,
                            true_next_state=details.true_next_state,
                            done=details.done,
                        )
                    )

                next_time_spent = self._next_time_spent(
                    state=state,
                    next_state=details.transition.next_state,
                    time_spent=time_spent,
                    action=action,
                    true_state=details.true_state,
                    true_next_state=details.true_next_state,
                )
                next_features = self._encode_features_tensor(
                    state=details.transition.next_state,
                    time_spent=next_time_spent,
                    prev_action=action,
                    prev_reward=details.transition.reward,
                )

                next_history = deque(history, maxlen=self.sequence_length)
                next_history.append(next_features)
                next_sequence = self._sequence_from_history(next_history)

                self._learn_from_sample(
                    _SequenceTransitionSample(
                        state_sequence=state_sequence,
                        action=action,
                        reward=details.transition.reward,
                        next_state_sequence=next_sequence,
                        done=details.done,
                    )
                )

                history = next_history
                state = details.transition.next_state
                time_spent = next_time_spent
                prev_action = action
                prev_reward = details.transition.reward
                done = details.done

            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}")

        if verbose:
            print("Training completed.")

        return Trajectory(transitions=transitions)

    def simulate(self, trajectory: Trajectory) -> list[float]:
        self.validate_replay_trajectory(trajectory)
        log_likelihoods: list[float] = []

        history: deque[torch.Tensor] = deque(maxlen=self.sequence_length)
        prev_action = 0
        prev_reward = 0.0

        for transition in trajectory.transitions:
            if not isinstance(transition, TimedTransition):
                raise TypeError("RDQN agents expect TimedTransition trajectory data")

            if not history:
                history.append(
                    self._encode_features_tensor(
                        state=transition.state,
                        time_spent=transition.time_spent,
                        prev_action=prev_action,
                        prev_reward=prev_reward,
                    )
                )

            state_sequence = self._sequence_from_history(history)
            q_values = self._predict_q(state_sequence)
            log_likelihoods.append(
                self._log_action_probability(
                    q_values=q_values,
                    action=transition.action,
                )
            )

            next_time_spent = self._next_time_spent(
                state=transition.state,
                next_state=transition.next_state,
                time_spent=transition.time_spent,
                action=transition.action,
                true_state=getattr(transition, "true_state", None),
                true_next_state=getattr(transition, "true_next_state", None),
            )
            next_features = self._encode_features_tensor(
                state=transition.next_state,
                time_spent=next_time_spent,
                prev_action=transition.action,
                prev_reward=transition.reward,
            )

            next_history = deque(history, maxlen=self.sequence_length)
            next_history.append(next_features)
            next_sequence = self._sequence_from_history(next_history)

            self._learn_from_sample(
                _SequenceTransitionSample(
                    state_sequence=state_sequence,
                    action=transition.action,
                    reward=transition.reward,
                    next_state_sequence=next_sequence,
                    done=transition.done,
                )
            )

            if transition.done:
                history.clear()
                prev_action = 0
                prev_reward = 0.0
            else:
                history = next_history
                prev_action = transition.action
                prev_reward = transition.reward

        return log_likelihoods


__all__ = ["RDQNAgent"]
