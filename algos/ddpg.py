import gymnasium as gym

import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

from utils.replay_buffer import ReplayBuffer


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

class Actor(nn.Module):
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Initialize continuous action space env
envs = gym.vector.SyncVectorEnv([make_env("HalfCheetah-v5", 1, 0, True, "test") for i in range(num_envs)])

qf1 = QNetwork(envs).to(device)
qf1_target = QNetwork(envs).to(device)
actor = Actor(envs).to(device)
target_actor = Actor(envs).to(device)

qf1_target.load_state_dict(qf1.state_dict())
target_actor.load_state_dict(actor.state_dict())

envs.single_action_space.dtype = np.float32
rb = ReplayBuffer(
    buffer_size=int(1e6),
    obs_dim=int(np.prod(envs.single_observation_space.shape)),
    action_dim=int(np.prod(envs.single_action_space.shape)),
    n_envs=num_envs,
    device=device,
)