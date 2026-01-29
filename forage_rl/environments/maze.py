"""Foraging maze environments for reinforcement learning experiments."""

from typing import List, Optional

import numpy as np

from forage_rl.config import DefaultParams, MazeParams
from forage_rl import SignedInteger


class ForagingReward:
    """Models time-dependent reward depletion in a foraging patch.

    Reward probability decays exponentially over time spent in the patch,
    simulating food depletion as the agent forages.
    """

    def __init__(self, decay: float):
        self.decay = decay
        self.counter = 0

    def reset(self):
        """Reset the counter when leaving the patch."""
        self.counter = 0

    def sample_reward(self) -> float:
        """Sample a stochastic reward based on current depletion level."""
        prob = np.exp(-self.decay * self.counter)
        self.counter += 1
        return 1.0 if np.random.rand() < prob else 0.0


# TODO: Need a more flexible way to define state-action-reward configs
# States can be bound to a different set of actions and reward decays
# States can have the same label, but be bound to a different reward decay
class Maze:
    """
    Base Maze environment for reinforcement learning experiments.

    Default Environment:
    Full 6-state foraging maze with stochastic transitions.

    States 0-2: Upper patch (reward decay decreases as state number increases)
    States 3-5: Lower patch (reward decay increases as state number increases)
    Actions: 0 = stay, 1 = leave
    """

    def __init__(
        self,
        decays: Optional[List[float]] = None,
        horizon: Optional[int] = None,
        num_states: int = 2,
        num_actions: int = 2,
        state: int = 0,
        time: int = 0,
        rewards: Optional[List[ForagingReward]] = None,
        state_labels: Optional[List[str]] = None,
        action_labels: Optional[List[str]] = None,
    ) -> None:
        self.decays = decays or MazeParams.FULL_MAZE_DECAYS
        self.horizon = horizon or DefaultParams.HORIZON
        self.num_states = num_states
        self.num_actions = num_actions
        self.action_labels = self._set_action_labels(
            action_labels or MazeParams.BASE_ACTIONS
        )
        self.state = state
        self.state_labels = self._set_state_labels(
            state_labels or MazeParams.BASE_STATES
        )
        self.time = time
        self.rewards = rewards or self._init_rewards()

    def _set_state_labels(self, state_labels: List[str]) -> List[str]:
        if len(state_labels) != self.num_states:
            raise ValueError(f"Number of states {self.num_states} does not match")

        return state_labels

    def _set_action_labels(self, action_labels: List[str]) -> List[str]:
        if len(set(action_labels)) != self.num_actions:
            raise ValueError(
                f"Action labels must have {self.num_actions} unique actions"
            )

        return action_labels

    def _init_rewards(self) -> List[ForagingReward]:
        return [ForagingReward(d) for d in self.decays]

    def get_state_label(self, state_idx: int) -> str:
        return self.state_labels[state_idx]

    def get_action_label(self, action_idx: int) -> str:
        return self.action_labels[action_idx]

    def reset(self) -> int:
        """Reset environment to initial state."""
        self.state = 0
        self.time = 0
        for r in self.rewards:
            r.reset()
        return self.state

    def step(self, action: SignedInteger) -> tuple:
        """Execute action and return (next_state, reward, done)."""
        new_state = self._get_transition(action)
        reward = self._get_reward(new_state)
        self.state = new_state
        self.time += 1
        done = self.time >= self.horizon
        return new_state, reward, done

    def _get_reward(self, new_state: int) -> float:
        """Get reward based on state transition."""
        if self.state == new_state:
            return self.rewards[self.state].sample_reward()
        else:
            # Mouse left patch - reset all patches, no reward during travel
            for r in self.rewards:
                r.reset()
            return 0.0

    def _get_transition(self, action: SignedInteger) -> int:
        """Get next state based on action."""
        if action == 0:  # Stay
            return self.state

        # Leave - stochastic transition to other patch
        rand = np.random.rand()
        probs = MazeParams.TRANSITION_PROBS

        if self.state in [0, 1, 2]:  # Upper patch -> Lower patch
            if rand < probs[0]:
                return 3
            elif rand < probs[0] + probs[1]:  # 0.5
                return 4
            else:
                return 5
        else:  # Lower patch -> Upper patch
            if rand < probs[0]:
                return 0
            elif rand < probs[0] + probs[1]:  # 0.5
                return 1
            else:
                return 2


class MazePOMDP(Maze):
    """Partially observable version of the maze.

    Observations are reduced to 2 states (upper/lower patch),
    hiding the specific patch identity.
    """

    def __init__(self, decays: Optional[list] = None, horizon: Optional[int] = None):
        super().__init__(decays, horizon)
        self.num_states = 2  # Observation space

    def step(self, action: SignedInteger) -> tuple:
        """Execute action and return (observation, reward, done)."""
        true_state, reward, done = super().step(action)
        obs = 0 if true_state in [0, 1, 2] else 1
        return obs, reward, done


class SimpleMaze(Maze):
    """Simplified 2-state foraging maze with deterministic transitions.

    State 0: Upper patch
    State 1: Lower patch
    Actions: 0 = stay, 1 = leave
    """

    def __init__(self, decays: Optional[list] = None, horizon: Optional[int] = None):
        super().__init__(
            decays or MazeParams.SIMPLE_MAZE_DECAYS,
            horizon or DefaultParams.HORIZON,
            num_states=2,
            num_actions=2,
            state=0,
            state_labels=MazeParams.SIMPLE_STATES,
            time=0,
            action_labels=MazeParams.SIMPLE_ACTIONS,
        )

    def reset(self) -> int:
        """Reset environment to initial state."""
        self.state = 0
        self.time = 0
        for r in self.rewards:
            r.reset()
        return self.state

    def step(self, action: SignedInteger) -> tuple:
        """Execute action and return (next_state, reward, done)."""
        new_state = self._get_transition(action)
        reward = self._get_reward(new_state)
        self.state = new_state
        self.time += 1
        done = self.time >= self.horizon
        return new_state, reward, done

    def _get_reward(self, new_state: int) -> float:
        """Get reward based on state transition."""
        if self.state == new_state:
            return self.rewards[self.state].sample_reward()
        else:
            for r in self.rewards:
                r.reset()
            return 0.0

    def _get_transition(self, action: SignedInteger) -> int:
        """Get next state based on action (deterministic)."""
        if action == 0:  # Stay
            return self.state
        else:  # Leave - deterministic transition
            return 1 - self.state


if __name__ == "__main__":
    # Simple test
    print("Testing SimpleMaze")
    maze = SimpleMaze()
    print(f"Initial state: {maze.state}")

    actions = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    for step, action in enumerate(actions):
        action_name = "stay" if action == 0 else "leave"
        obs, reward, done = maze.step(action)
        print(
            f"Step {step}: action={action_name}, state={obs}, reward={reward:.1f}, done={done}"
        )
