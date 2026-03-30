import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import gymnasium as gym

from forage_rl import Trajectory
from forage_rl import TimedTransition
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze

from base import BaseAgent


# Small enough observation space that an EncoderCNN is not needed.
# DecoderCNN, on the other hand, converts the hidden and latent states into back to observation space.
# In our case, it does not need to be a CNN, but we do need some model that does this conversion.
class ObservationModel(nn.Module):
    """
    o_t = f(h_t, s_t)
    """
    def __init__(self, observation_dim, hidden_dim, latent_dim, hidden_layer_size: int):
        super(ObservationModel, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, observation_dim)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor):
        x = torch.cat([h_t, s_t], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RewardModel(nn.Module):
    """
    r_t ~ p(r_t | h_t, s_t)

    The Reward Model is defined as a mapping from the deterministic state h
    and the stochastic state s to parameters for a Gaussian distribution to sample the reward from.
    """
    def __init__(self, hidden_dim: int, latent_dim: int, hidden_layer_size: int):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 2)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor):
        x = torch.cat([h_t, s_t], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeterministicStateModel(nn.Module):
    """
    h_t = f(h_{t-1}, s_{t-1}, a_{t-1})

    The Deterministic State Model is defined as a recurrent neural network that
    takes the last deterministic state h_{t-1}, the last stochastic state s_{t-1},
    and the last action a_{t-1}, mapping it to the next deterministic state in latent space.
    """
    def __init__(self, hidden_dim: int, latent_dim: int, action_dim: int):
        super(DeterministicStateModel, self).__init__()

        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

    def forward(self, h_prev: torch.Tensor, s_prev: torch.Tensor, a_t: torch.Tensor):
        x = torch.cat([s_prev, a_t], dim=-1)
        h_t = self.rnn(x, h_prev)
        return h_t


class PriorStochasticStateModel(nn.Module):
    """
    s_t ~ p(s_t | h_t)
    """
    def __init__(self, latent_dim: int, hidden_dim: int):
        super(PriorStochasticStateModel, self).__init__()

        self.prior = nn.Linear(hidden_dim, 2 * latent_dim)
    
    def forward(self, h_t):
        s_t = self.prior(h_t)
        return s_t


class PosteriorStochasticStateModel(nn.Module):
    """
    s_t ~ p(s_t | h_t, o_t)
    """
    def __init__(self, latent_dim: int, hidden_dim: int, observation_dim: int):
        super(PosteriorStochasticStateModel, self).__init__()

        self.posterior = nn.Linear(hidden_dim + observation_dim, 2 * latent_dim)
    
    def forward(self, h_t, o_t):
        x = torch.cat([h_t, o_t], dim=-1)
        s_t = self.posterior(x)
        return s_t


class ReplayBuffer:
    def __init__(self, buffer_size: int, obs_shape: tuple, action_shape: tuple, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.obs_buffer = np.zeros((buffer_size, *obs_shape))
        self.action_buffer = np.zeros((buffer_size, *action_shape), dtype=np.int32)
        self.reward_buffer = np.zeros((buffer_size, 1))
        self.done_buffer = np.zeros((buffer_size, 1), dtype=np.bool)

        self.device = device

        self.idx = 0

    def add(self, obs: torch.Tensor, action: int, reward: float, done: bool):
        self.obs_buffer[self.idx] = obs
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.done_buffer[self.idx] = done

        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, batch_size: int, sequence_length: int):
        starting_idxs = np.random.randint(0, (self.idx % self.buffer_size) - sequence_length, (batch_size,))

        index_tensor = np.stack([np.arange(start, start + sequence_length) for start in starting_idxs])
        obs_sequence = self.obs_buffer[index_tensor]
        action_sequence = self.action_buffer[index_tensor]
        reward_sequence = self.reward_buffer[index_tensor]
        done_sequence = self.done_buffer[index_tensor]

        return obs_sequence, action_sequence, reward_sequence, done_sequence

    def save(self, path: str):
        np.savez(path, obs_buffer=self.obs_buffer, action_buffer=self.action_buffer,
                 reward_buffer=self.reward_buffer, done_buffer=self.done_buffer, idx=self.idx)

    def load(self, path: str):
        data = np.load(path)
        self.obs_buffer = data["obs_buffer"]
        self.action_buffer = data["action_buffer"]
        self.reward_buffer = data["reward_buffer"]
        self.done_buffer = data["done_buffer"]
        self.idx = data["idx"]


class RSSMAgent(BaseAgent):
    def __init__(
            self,
            maze: Maze,
            replay_buffer: ReplayBuffer,
            hidden_dim: int,
            latent_dim: int,
            hidden_layer_size: int = 64,
            device: str = "cpu",
            num_episodes = DefaultParams.NUM_EPISODES):

        super().__init__(maze)

        self.num_episodes = num_episodes

        self.replay_buffer = replay_buffer

        observation_dim = maze.observation_space.n
        action_dim = maze.action_space.n

        self.observation_model         = ObservationModel(observation_dim, hidden_dim, latent_dim, hidden_layer_size)
        self.reward_model              = RewardModel(hidden_dim, latent_dim, hidden_layer_size)
        self.deterministic_state_model = DeterministicStateModel(hidden_dim, latent_dim, action_dim)
        self.prior_model               = PriorStochasticStateModel(latent_dim, hidden_dim)
        self.posterior_model           = PosteriorStochasticStateModel(latent_dim, hidden_dim, observation_dim)

        # shift to device
        self.observation_model.to(device)
        self.reward_model.to(device)
        self.deterministic_state_model.to(device)
        self.prior_model.to(device)
        self.posterior_model.to(device)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device


    def simulate(self, trajectory) -> list[float]:
        """Evaluate log-likelihood of each transition under this agent's learning rule."""


    def train(self, verbose: bool = True) -> Trajectory:
        """Train the agent on the provided Maze"""
        transitions = []

        for episode in range(self.num_episodes):
            obs, _ = self.maze.reset()
            time_spent = 0
            done = False
            max_time_spent = 0
            prev_reward = 0

            h = torch.zeros(self.hidden_dim)
            s = torch.zeros(self.hidden_dim)

            while not done:
                action = self.choose_action(obs, time_spent, prev_reward, h, s)
                transition, done = self.maze.step_transition(action)

                timed_transition = TimedTransition.from_transition_time(
                    transition, time_spent
                )

                transitions.append(timed_transition)

                # update

                next_state = timed_transition.next_state
                if obs == next_state:
                    time_spent += 1
                else:
                    time_spent = 0

                obs = next_state
                max_time_spent = max(max_time_spent, time_spent)

            if verbose:
                print(f"Episode {episode}, max time spent: {max_time_spent}")
                # if episode % 100 == 0:
                #     avg_q = np.mean(self.q_table)
                #     print(f"Episode {episode}, Average Q-value: {avg_q:.4f}")
        
        return Trajectory(transitions=transitions)

    def choose_action(self, obs, time_spent, prev_reward, h, s):
        pass

    def imagine_step(self, h_prev, s_prev, action):
        h = self.deterministic_state_model(h_prev, s_prev, action)
        params = self.prior_model(h)
        mean, logvar = torch.chunk(params, 2, dim=-1)
        dist = torch.distributions.Normal(mean, torch.exp(logvar))
        s = dist.rsample()
        return h, s, mean, logvar
    
    def observe_step(self, h_prev, s_prev, action, obs):
        h = self.deterministic_state_model(h_prev, s_prev, action)
        params = self.posterior_model(h, obs)
        mean, logvar = torch.chunk(params, 2, dim=-1)
        dist = torch.distributions.Normal(mean, torch.exp(logvar))
        s = dist.rsample()
        return h, s, mean, logvar

    def _reconstruction_loss(self, decoded_obs, obs):
        return F.mse_loss(decoded_obs, obs)

    def _kl_loss(self, prior_means, prior_logvars, posterior_means, posterior_logvars):
        prior_dist = torch.distributions.Normal(prior_means, torch.exp(prior_logvars))
        posterior_dist = torch.distributions.Normal(posterior_means, torch.exp(posterior_logvars))

        return torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()

    def _reward_loss(self, rewards, predicted_rewards):
        return F.mse_loss(predicted_rewards, rewards)