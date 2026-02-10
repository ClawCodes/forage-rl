"""Gymnasium-compatible foraging maze environments."""

from typing import Any, List, Optional, Dict

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from forage_rl.config import DefaultParams, MazeParams, EnvConfig
from forage_rl.environments.maze import ForagingReward


class BaseEnv(gym.Env):
    """Base class for gym maze environments."""

    def __init__(self, config: EnvConfig):
        self._observation_space = config.observation_space
        self._action_space = config.action_space
        self._decays = config.decays
        self._rewards = config.rewards
        self._horizon = config.horizon
        self._transition_probs = config.transition_probs

        # TODO: should agent location and time go in config?
        self._agent_location = 0
        self._time = 0

        # TODO: Add handling of state and action labels

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        return {"time": self._time, **kwargs}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._agent_location = 0
        self._time = 0
        for r in self._rewards:
            r.reset()
        return self._agent_location, self._get_info()

    def _get_reward(self, new_state: int) -> float:
        if self._state == new_state:
            return self._rewards[self._state].sample_reward()
        else:
            for r in self._rewards:
                r.reset()
            return 0.0

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        prev_state = self._agent_location
        new_state = self._get_transition(action)
        reward = self._get_reward(new_state)

        self._state = new_state
        self._time += 1
        truncated = self._time >= self._horizon

        return (
            self._state,
            reward,
            False,
            truncated,
            self._get_info(prev_state=prev_state, action=action),
        )

    def _get_transition(self, action: int) -> int:
        if self._transition_probs is not None:
            return self._action_space.sample(
                probability=np.array(self._transition_probs)
            )

        return self._action_space.sample()


class ForagingMazeEnv(gym.Env):
    """Gymnasium version of the full 6-state foraging maze.

    States 0-2: Upper patch (reward decay decreases as state number increases)
    States 3-5: Lower patch (reward decay increases as state number increases)
    Actions: 0 = stay, 1 = leave

    Leaving triggers a stochastic transition to a state in the opposite patch.
    Rewards decay exponentially the longer the agent stays in a patch.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        decays: Optional[List[float]] = None,
        horizon: int = DefaultParams.HORIZON,
    ) -> None:
        super().__init__()
        self.decays = decays or MazeParams.FULL_MAZE_DECAYS
        self.horizon = horizon

        self.observation_space = Discrete(6)
        self.action_space = Discrete(2)

        self._state = 0
        self._time = 0
        self._rewards = [ForagingReward(d) for d in self.decays]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._state = 0
        self._time = 0
        for r in self._rewards:
            r.reset()
        return self._state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        prev_state = self._state
        new_state = self._get_transition(action)
        reward = self._get_reward(new_state)

        self._state = new_state
        self._time += 1
        truncated = self._time >= self.horizon

        info = {
            "prev_state": prev_state,
            "action": action,
        }
        return self._state, reward, False, truncated, info

    def _get_reward(self, new_state: int) -> float:
        if self._state == new_state:
            return self._rewards[self._state].sample_reward()
        else:
            for r in self._rewards:
                r.reset()
            return 0.0

    def _get_transition(self, action: int) -> int:
        if action == 0:
            return self._state

        rand = self.np_random.random()
        probs = MazeParams.TRANSITION_PROBS

        if self._state in (0, 1, 2):  # Upper -> Lower
            if rand < probs[0]:
                return 3
            elif rand < probs[0] + probs[1]:
                return 4
            else:
                return 5
        else:  # Lower -> Upper
            if rand < probs[0]:
                return 0
            elif rand < probs[0] + probs[1]:
                return 1
            else:
                return 2


class ForagingMazePOMDPEnv(ForagingMazeEnv):
    """Partially observable version of the foraging maze.

    Observations are reduced to 2 states (0 = upper patch, 1 = lower patch),
    hiding the specific sub-state identity within each patch.
    """

    def __init__(
        self,
        decays: Optional[List[float]] = None,
        horizon: int = DefaultParams.HORIZON,
    ) -> None:
        super().__init__(decays, horizon)
        self.observation_space = Discrete(2)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options)
        obs = 0 if self._state in (0, 1, 2) else 1
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = super().step(action)
        obs = 0 if self._state in (0, 1, 2) else 1
        info["true_state"] = self._state
        return obs, reward, terminated, truncated, info


class SimpleForagingMazeEnv(gym.Env):
    """Gymnasium version of the simplified 2-state foraging maze.

    State 0: Upper patch
    State 1: Lower patch
    Actions: 0 = stay, 1 = leave

    Transitions are deterministic: leaving always moves to the other patch.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        decays: Optional[List[float]] = None,
        horizon: int = DefaultParams.HORIZON,
    ) -> None:
        super().__init__()
        self.decays = decays or MazeParams.SIMPLE_MAZE_DECAYS
        self.horizon = horizon

        self.observation_space = Discrete(2)
        self.action_space = Discrete(2)

        self._state = 0
        self._time = 0
        self._rewards = [ForagingReward(d) for d in self.decays]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._state = 0
        self._time = 0
        for r in self._rewards:
            r.reset()
        return self._state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        prev_state = self._state
        new_state = self._get_transition(action)
        reward = self._get_reward(new_state)

        self._state = new_state
        self._time += 1
        truncated = self._time >= self.horizon

        info = {
            "prev_state": prev_state,
            "action": action,
        }
        return self._state, reward, False, truncated, info

    def _get_reward(self, new_state: int) -> float:
        if self._state == new_state:
            return self._rewards[self._state].sample_reward()
        else:
            for r in self._rewards:
                r.reset()
            return 0.0

    def _get_transition(self, action: int) -> int:
        if action == 0:
            return self._state
        return 1 - self._state
