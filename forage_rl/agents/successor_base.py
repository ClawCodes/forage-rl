"""Base class shared by all Successor Representation agents."""

import numpy as np

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.q_table import QTable
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class BaseSRAgent(BaseAgent):
    """
    Base class for Successor Representation agents.

    Holds the successor matrix M, reward weights w, and running-average
    reward estimates. Provides shared utilities for computing values,
    updating the Q-value cache, and updating reward weights.

    Value decomposition:
        V(s)     = M(s,:) · w
        Q(s,a,t) = r(s,a,t) + γ · Σ_{s'} P(s'|s,a) · V[s']

    M is initialized to the identity: before any experience, each state
    predicts only itself in its future trajectory. No pretraining phase
    is needed — M and w are learned concurrently during normal training.
    """

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.TRAINING_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        alpha_sr: float = DefaultParams.ALPHA_SR,
        alpha_w: float = DefaultParams.ALPHA_W,
        beta: float = DefaultParams.BETA,
        seed: int | None = None,
    ):
        super().__init__(maze, beta, seed=seed)
        self.q_table = QTable(maze, timed=True)  # Q-value cache; overrides base default

        n = maze.observation_space.n  # type: ignore
        self.M = np.eye(n)  # (n, n) successor matrix
        self.w = np.zeros(n)  # (n,) reward weights
        self.r_table = QTable(maze, timed=True)  # running-average immediate rewards
        self.count = QTable(maze, timed=True)  # visit counts for running average

        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha_sr = alpha_sr
        self.alpha_w = alpha_w

    def _get_transitions(self, state: int, action: int) -> list[tuple[int, float]]:
        """Return transition distribution, handling FO and PO mazes."""
        if not self.maze.observable:
            return self.maze.obs_transition_distribution(state, action)
        return self.maze.transition_distribution(state, action)

    def compute_v(self) -> np.ndarray:
        """V(s) = M(s,:) · w,  shape (n,)."""
        return self.M @ self.w

    def update_q_cache(self) -> None:
        """Recompute Q(s,a,t) = r(s,a,t) + γ·Σ P(s'|s,a)·V[s'] and store in q_table."""
        V = self.compute_v()
        for s in range(self.maze.observation_space.n):  # type: ignore
            for t in range(self.maze.horizon):
                for a in self.q_table.valid_actions(s):
                    r_sa = self.r_table.get(s, a, t)
                    ev = sum(p * V[ns] for ns, p in self._get_transitions(s, a))
                    self.q_table.set(s, a, r_sa + self.gamma * ev, t)

    def update_r(self, state: int, action: int, reward: float, time_spent: int) -> None:
        """Incremental running-average reward update (same pattern as MBRL)."""
        self.count.update(state, action, 1.0, time_spent)
        n = self.count.get(state, action, time_spent)
        delta = (reward - self.r_table.get(state, action, time_spent)) / n
        self.r_table.update(state, action, delta, time_spent)

    def update_w(self, state: int, reward: float) -> None:
        """Track immediate reward at state as exponential moving average.

        w(s) represents E[R | visiting state s] under the current policy.
        V(s) = M(s,:) · w then gives the expected cumulative future reward,
        which is the correct interpretation of the SR decomposition.

        The TD-error form of this update (Russek et al. Eq 8) is equivalent when M
        is fixed, but is numerically unstable when M and w are co-updated
        simultaneously. The EMA form is stable and converges to the same
        fixed point: w(s) = average immediate reward observed at state s.
        """
        self.w[state] += self.alpha_w * (reward - self.w[state])
