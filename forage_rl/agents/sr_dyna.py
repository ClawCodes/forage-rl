"""SR-Dyna: Off-policy experience replay over the state-action successor representation."""

import numpy as np

from forage_rl.types import RunDataset, TimedTransition, Trajectory
from forage_rl.agents.successor_base import BaseSRAgent
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class SRDynaAgent(BaseSRAgent):
    """
    SR-Dyna (Algorithm 3 in Russek et al. 2017).

    Uses the state-action successor representation H instead of the state SR M.
    H(sa, s'a') = expected discounted future visits to state-action pair (s'a')
    starting from (sa). Q-values decompose as:

        Q(sa) = Σ_{s'a'} H(sa, s'a') · w_sa(s'a')     [Eq 13]

    H is updated online (on-policy, Eq 17) and refined via off-policy experience
    replay (Eq 18), using the greedy action a* at each replayed next state.
    This off-policy update makes H converge toward the optimal policy rather
    than the behavioral policy, resolving SR-MB's policy-dependence limitation.

    With sufficient replay (k_replay large), SR-Dyna matches full model-based
    value iteration. With limited replay it degrades gracefully toward SR-TD.
    """

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.TRAINING_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        alpha_sr: float = DefaultParams.ALPHA_SR,
        alpha_w: float = DefaultParams.ALPHA_W,
        beta: float = DefaultParams.BETA,
        k_replay: int = DefaultParams.K_REPLAY,
        seed: int | None = None,
    ):
        super().__init__(maze, num_episodes, gamma, alpha_sr, alpha_w, beta, seed)
        self.k_replay = k_replay

        self.sa_to_flat, self.flat_to_sa, n_sa = self._build_sa_index()
        self.H = np.eye(n_sa)  # (n_sa, n_sa) state-action SR matrix
        self.w_sa = np.zeros(n_sa)  # reward weights over state-action pairs
        self.replay_buffer: list[
            tuple[int, int, int]
        ] = []  # (state, action, next_state)

    def _build_sa_index(self) -> tuple[dict, list, int]:
        """Build a flat index over all valid (state, global_action) pairs."""
        idx = 0
        sa_to_flat: dict[tuple[int, int], int] = {}
        flat_to_sa: list[tuple[int, int]] = []
        for s in range(self.maze.observation_space.n):  # type: ignore
            for a in self.q_table.valid_actions(s):
                sa_to_flat[(s, a)] = idx
                flat_to_sa.append((s, a))
                idx += 1
        return sa_to_flat, flat_to_sa, idx

    def compute_q_sa(self, state: int) -> np.ndarray:
        """Q(s,a) for all valid actions at state, derived from H and w_sa."""
        valid = self.q_table.valid_actions(state)
        return np.array(
            [self.H[self.sa_to_flat[(state, a)]] @ self.w_sa for a in valid]
        )

    def _greedy_sa_flat(self, state: int) -> int:
        """Flat index of the greedy (highest-Q) action at state."""
        q = self.compute_q_sa(state)
        best_local = int(np.argmax(q))
        best_action = self.q_table.valid_actions(state)[best_local]
        return self.sa_to_flat[(state, best_action)]

    def _update_H(self, sa_flat: int, next_sa_flat: int) -> None:
        """TD update to H row (Eq 17/18).

        H(sa,:) += α_SR · (e_sa + γ·H(next_sa,:) − H(sa,:))
        """
        n_sa = len(self.flat_to_sa)
        e_sa = np.zeros(n_sa)
        e_sa[sa_flat] = 1.0
        self.H[sa_flat] += self.alpha_sr * (
            e_sa + self.gamma * self.H[next_sa_flat] - self.H[sa_flat]
        )

    def _update_w_sa(self, sa_flat: int, reward: float) -> None:
        """Track immediate reward for state-action pair as exponential moving average.

        w_sa(sa) = E[R | taking action a in state s]. Combined with H, this gives
        Q(sa) = Σ H(sa, s'a') * w_sa(s'a') = expected discounted future reward.
        The EMA form is numerically stable; the TD-error form diverges when H and
        w_sa are co-updated simultaneously.
        """
        self.w_sa[sa_flat] += self.alpha_w * (reward - self.w_sa[sa_flat])

    def _replay_step(self) -> None:
        """Off-policy H update from one replayed experience (Eq 18)."""
        if not self.replay_buffer:
            return
        s, a, ns = self.replay_buffer[np.random.randint(len(self.replay_buffer))]
        sa_flat = self.sa_to_flat[(s, a)]
        next_sa_flat = self._greedy_sa_flat(ns)  # off-policy: greedy a* at ns
        self._update_H(sa_flat, next_sa_flat)

    def _sync_q_cache(self, state: int) -> None:
        """Write Q(s,a) into q_table so print_policy and simulate work correctly."""
        for a in self.q_table.valid_actions(state):
            q_val = self.H[self.sa_to_flat[(state, a)]] @ self.w_sa
            for t in range(self.maze.horizon):
                self.q_table.set(state, a, q_val, t)

    def train(self, verbose: bool = True) -> RunDataset:
        """Train the agent and return one trajectory per episode."""
        trajectories: list[Trajectory[TimedTransition]] = []

        for episode in range(self.num_episodes):
            state, _ = self.maze.reset()
            time_spent = 0
            done = False
            prev_sa_flat: int | None = None

            episode_transitions: list[TimedTransition] = []

            while not done:
                local_idx = self.choose_action_boltzmann(self.compute_q_sa(state))
                action = self.q_table.local_to_global(state, local_idx)
                sa_flat = self.sa_to_flat[(state, action)]

                # Online H update: now that we know (s', a'), complete the previous step
                if prev_sa_flat is not None:
                    self._update_H(prev_sa_flat, sa_flat)

                transition, done = self.maze.step_transition(action)
                timed_t = TimedTransition.from_transition_time(transition, time_spent)
                episode_transitions.append(timed_t)

                self._update_w_sa(sa_flat, timed_t.reward)

                self.replay_buffer.append((state, action, timed_t.next_state))
                for _ in range(self.k_replay):
                    self._replay_step()

                self._sync_q_cache(state)

                prev_sa_flat = sa_flat
                time_spent = time_spent + 1 if state == timed_t.next_state else 0
                state = timed_t.next_state

            trajectories.append(Trajectory(transitions=episode_transitions))

            if verbose and episode % 50 == 0:
                print(f"Episode {episode}")

        if verbose:
            print("Training completed.")
            self.print_policy()

        return RunDataset(trajectories=trajectories)

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Return per-transition log-likelihoods and update internal state."""
        log_likelihoods = []
        prev_sa_flat: int | None = None

        for t in trajectory.transitions:
            q_vals = self.compute_q_sa(t.state)
            action_probs = self.boltzmann_action_probs(q_vals)
            local_idx = self.q_table.global_to_local(t.state, t.action)
            log_likelihoods.append(np.log(action_probs[local_idx]))

            sa_flat = self.sa_to_flat[(t.state, t.action)]

            if prev_sa_flat is not None:
                self._update_H(prev_sa_flat, sa_flat)

            self._update_w_sa(sa_flat, t.reward)

            self.replay_buffer.append((t.state, t.action, t.next_state))
            for _ in range(self.k_replay):
                self._replay_step()

            self._sync_q_cache(t.state)
            prev_sa_flat = sa_flat

        return log_likelihoods
