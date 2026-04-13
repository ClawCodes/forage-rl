"""SR-TD: Successor Representation learned via temporal difference updates."""

import numpy as np

from forage_rl import TimedTransition, Trajectory
from forage_rl.agents.successor_base import BaseSRAgent
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class SRTDAgent(BaseSRAgent):
    """
    SR-TD (Algorithm 1 in Russek et al. 2017).

    Learns the successor matrix M via TD updates applied directly to future
    state occupancy predictions. After each s → s' transition:

        M(s,:) += α_SR · (e_s + γ·M(s',:) − M(s,:))     [Eq 10]

    M only propagates via direct experience: it cannot infer that upstream
    states have changed without visiting them. This makes SR-TD capable of
    reward revaluation (latent learning) but not transition revaluation
    (detour / one-way corridor tasks).
    """

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        alpha_sr: float = DefaultParams.ALPHA_SR,
        alpha_w: float = DefaultParams.ALPHA_W,
        beta: float = DefaultParams.BETA,
    ):
        super().__init__(maze, num_episodes, gamma, alpha_sr, alpha_w, beta)

    def update_M(self, state: int, next_state: int) -> None:
        """TD update to row M(s,:) after observing transition s → s' (Eq 10)."""
        n = self.maze.observation_space.n  # type: ignore
        e_s = np.zeros(n)
        e_s[state] = 1.0
        self.M[state] += self.alpha_sr * (
            e_s + self.gamma * self.M[next_state] - self.M[state]
        )

    def _step_update(self, t: TimedTransition) -> None:
        """Apply all updates for a single observed transition."""
        self.update_r(t.state, t.action, t.reward, t.time_spent)
        self.update_M(t.state, t.next_state)
        self.update_w(t.state, t.reward)
        self.update_q_cache()

    def train(self, verbose: bool = True) -> Trajectory:
        """Train the agent and return the collected trajectory."""
        transitions = []

        for episode in range(self.num_episodes):
            state, _ = self.maze.reset()
            time_spent = 0
            done = False

            while not done:
                local_idx = self.choose_action_boltzmann(
                    self.q_table.action_values(state, time_spent)
                )
                action = self.q_table.local_to_global(state, local_idx)
                transition, done = self.maze.step_transition(action)

                timed_t = TimedTransition.from_transition_time(transition, time_spent)
                transitions.append(timed_t)
                self._step_update(timed_t)

                time_spent = time_spent + 1 if state == timed_t.next_state else 0
                state = timed_t.next_state

            if verbose and episode % 50 == 0:
                print(f"Episode {episode}")

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Return per-transition log-likelihoods and update internal state."""
        log_likelihoods = []

        for t in trajectory.transitions:
            action_probs = self.boltzmann_action_probs(
                self.q_table.action_values(t.state, t.time_spent)
            )
            local_idx = self.q_table.global_to_local(t.state, t.action)
            log_likelihoods.append(np.log(action_probs[local_idx]))
            self._step_update(t)

        return log_likelihoods
