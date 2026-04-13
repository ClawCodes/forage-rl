"""Feed-forward DQN agent."""

from __future__ import annotations

from collections import deque

import numpy as np

from forage_rl import RunDataset, TimedTransition, Trajectory
from forage_rl.agents.neural_base import NeuralAgentBase
from forage_rl.agents.registry import Agent

class DQNAgent(NeuralAgentBase):
    """DQN with replay, target networks, and Boltzmann action selection."""

    agent_name = Agent.DQN

    def __init__(
        self,
        maze,
        *,
        warmup_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(maze, **kwargs)
        self.replay_buffer = deque(maxlen=self.replay_capacity)
        resolved_warmup_steps = self.batch_size if warmup_steps is None else warmup_steps
        self._require_non_negative_int("warmup_steps", resolved_warmup_steps)
        self.warmup_steps = int(resolved_warmup_steps)

    def _build_model(self):
        return self.nn.Sequential(
            self.nn.Linear(self.feature_dim, 64),
            self.nn.ReLU(),
            self.nn.Linear(64, 64),
            self.nn.ReLU(),
            self.nn.Linear(64, self.maze.num_actions),
        )

    def _predict_q_values(self, model, feature, hidden=None):
        del hidden
        q_values = model(feature.unsqueeze(0)).squeeze(0)
        return q_values

    def _store_transition(
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
        self.replay_buffer.append(
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

    def _build_batch_tensors(self, batch: list[dict[str, object]]):
        states = self.torch.stack(
            [
                self.encode_feature_tensor(
                    int(item["state"]),
                    int(item["time_spent"]),
                    prev_action=item["prev_action"],
                    prev_reward=float(item["prev_reward"]),
                )
                for item in batch
            ]
        )
        next_states = self.torch.stack(
            [
                self.encode_feature_tensor(
                    int(item["next_state"]),
                    int(item["next_time_spent"]),
                    prev_action=int(item["action"]),
                    prev_reward=float(item["reward"]),
                )
                for item in batch
            ]
        )
        actions = self.torch.tensor(
            [item["action"] for item in batch],
            device=self.torch_device,
            dtype=self.torch.long,
        )
        rewards = self.torch.tensor(
            [item["reward"] for item in batch],
            device=self.torch_device,
            dtype=self.torch.float32,
        )
        dones = self.torch.tensor(
            [item["done"] for item in batch],
            device=self.torch_device,
            dtype=self.torch.float32,
        )
        return states, next_states, actions, rewards, dones

    def _train_from_replay(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return
        if self.training_steps < self.warmup_steps:
            return

        batch_indices = self.rng.choice(
            len(self.replay_buffer),
            size=self.batch_size,
            replace=False,
        )
        batch = [self.replay_buffer[int(idx)] for idx in batch_indices]

        states, next_states, actions, rewards, dones = self._build_batch_tensors(batch)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with self.torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = self.loss_fn(q_values, targets).mean()
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
        """Choose an action from the current network policy."""
        feature = self.encode_feature_tensor(
            state,
            time_spent,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        q_values = self.q_values_for_feature(feature)
        action, _ = self.action_from_q_values(q_values)
        return action

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Evaluate one episode while preserving online DQN learning semantics.

        This intentionally mutates replay and optimizer state so repeated calls
        behave like a continuing evaluator rather than a pure forward pass.
        """
        log_likelihoods: list[float] = []
        context = self.initial_context()

        for step_index, transition in enumerate(trajectory.transitions):
            time_spent = getattr(transition, "time_spent", 0)
            feature = self.encode_feature_tensor(
                transition.state,
                time_spent,
                prev_action=context["prev_action"],
                prev_reward=context["prev_reward"],
            )
            q_values = self.q_values_for_feature(feature)
            action_probs = self.boltzmann_action_probs(q_values.detach().cpu().numpy())
            log_likelihoods.append(float(np.log(action_probs[transition.action])))

            next_time_spent = (
                min(time_spent + 1, self.maze.horizon - 1)
                if transition.state == transition.next_state
                else 0
            )
            self._store_transition(
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

        return log_likelihoods

    def train(self, verbose: bool = True) -> RunDataset:
        """Train the DQN agent and return one run dataset."""
        trajectories: list[Trajectory[TimedTransition]] = []

        for episode_idx in range(self.num_episodes):
            state, _ = self.maze.reset()
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
                self._store_transition(
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

            trajectories.append(Trajectory(transitions=episode_transitions))
            if verbose and episode_idx % 100 == 0:
                print(f"Episode {episode_idx}, replay={len(self.replay_buffer)}")

        return RunDataset(trajectories=trajectories)
