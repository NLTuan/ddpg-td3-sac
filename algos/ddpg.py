import gymnasium as gym 

import torch
from torch import nn, optim
import torch.nn.functional as F  

import numpy as np


class QNetwork(nn.Module):
    def __init__(self, env, hidden_dim=64):
        super().__init__()

        obs_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)  # cat(obs, action)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Q-value is scalar

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Policy(nn.Module):
    def __init__(self, env, hidden_dim=64):
        super().__init__()

        obs_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)  # policy takes obs only
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # use single_action_space

        # action_space.high/low on a vectorized env has shape (num_envs, action_dim)
        # use single_action_space for per-env bounds
        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_scale + self.action_bias

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


learning_rate = 2e-4
num_envs = 4
# Initialize continuous action space env
env = gym.vector.SyncVectorEnv([make_env("HalfCheetah-v5", 1, 0, True, "test") for i in range(num_envs)])

q1 = QNetwork(env)
q2 = QNetwork(env)
actor = Policy(env)

# env.observation_space.sample() -> (num_envs, obs_dim), keep the batch dim
sample_obs = torch.tensor(env.observation_space.sample(), dtype=torch.float32)    # (num_envs, obs_dim)
sample_action = torch.tensor(env.action_space.sample(), dtype=torch.float32)      # (num_envs, action_dim)

print("Obs batch shape:   ", sample_obs.shape)     # expect (4, 17)
print("Action batch shape:", sample_action.shape)  # expect (4, 6)

print("Q1 output shape:", q1(sample_obs, sample_action).shape)    # expect (4, 1)
print("Q2 output shape:", q2(sample_obs, sample_action).shape)    # expect (4, 1)
print("Actor output shape:", actor(sample_obs).shape)             # expect (4, 6)
