"""Feed-forward DQN agent using replay and a target network."""

from __future__ import annotations

import torch

from forage_rl import ObservedTimedTransition, TimedTransition, Trajectory
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze

from .torch_q import TorchQAgentBase, _MLPQNet, _ReplayBuffer, _TransitionSample


class DQNAgent(TorchQAgentBase):
    """DQN agent with replay-buffer updates and a target network."""

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        epsilon: float = DefaultParams.EPSILON,
        learning_rate: float = DefaultParams.DQN_LEARNING_RATE,
        hidden_dim: int = DefaultParams.DQN_HIDDEN_DIM,
        replay_size: int = DefaultParams.DQN_REPLAY_SIZE,
        batch_size: int = DefaultParams.DQN_BATCH_SIZE,
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

        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.replay_warmup = replay_warmup
        self.update_steps = 0

        self.policy_net = _MLPQNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_actions=self.maze.num_actions,
        ).to(self.device)
        self.optimizer = self._make_optimizer(self.policy_net)
        self.target_net = self._initialize_target_network(
            self.policy_net,
            lambda: _MLPQNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_actions=self.maze.num_actions,
            ),
        )
        self.replay_buffer = _ReplayBuffer[_TransitionSample](replay_size)

    def _predict_q(self, encoded_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(encoded_input).squeeze(0)

    def _td_update_batch(self, batch: list[_TransitionSample]) -> None:
        states = torch.stack([sample.state_features for sample in batch])
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
        next_states = torch.stack([sample.next_state_features for sample in batch])
        dones = torch.as_tensor(
            [sample.done for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )
        self._td_optimize_tensors(states, actions, rewards, next_states, dones)

    def _learn_from_sample(self, sample: object) -> None:
        if not isinstance(sample, _TransitionSample):
            raise TypeError("DQNAgent expects _TransitionSample inputs")

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

            while not done:
                state_features = self._encode_features_tensor(
                    state=state,
                    time_spent=time_spent,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                q_values = self._predict_q(state_features)
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

                self._learn_from_sample(
                    _TransitionSample(
                        state_features=state_features,
                        action=action,
                        reward=details.transition.reward,
                        next_state_features=next_features,
                        done=details.done,
                    )
                )

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
        prev_action = 0
        prev_reward = 0.0

        for transition in trajectory.transitions:
            if not isinstance(transition, TimedTransition):
                raise TypeError("DQN agents expect TimedTransition trajectory data")

            state_features = self._encode_features_tensor(
                state=transition.state,
                time_spent=transition.time_spent,
                prev_action=prev_action,
                prev_reward=prev_reward,
            )
            q_values = self._predict_q(state_features)
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

            self._learn_from_sample(
                _TransitionSample(
                    state_features=state_features,
                    action=transition.action,
                    reward=transition.reward,
                    next_state_features=next_features,
                    done=transition.done,
                )
            )

            if transition.done:
                prev_action = 0
                prev_reward = 0.0
            else:
                prev_action = transition.action
                prev_reward = transition.reward

        return log_likelihoods


__all__ = ["DQNAgent"]
