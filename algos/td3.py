import os
from dataclasses import dataclass

import gymnasium as gym
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from utils.replay_buffer import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True
    capture_video: bool = True

    # Env args
    env_id: str = "HalfCheetah-v5"
    num_envs: int = 1
    total_timesteps: int = 1_000_000
    buffer_size: int = int(1e6)
    
    # TD3 Hyperparameters (You might need more here!)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 25_000
    policy_frequency: int = 2
    exploration_noise: float = 0.1
    # TODO: Add policy_noise and noise_clip for Target Policy Smoothing

    policy_noise: float = 0.3
    noise_clip: float = 0.3

args = Args()
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_scale + self.action_bias

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

# ── Setup ──────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])

actor = Actor(envs).to(device)
target_actor = Actor(envs).to(device)
target_actor.load_state_dict(actor.state_dict())

# TODO: TD3 uses "Twin" Q-networks. Initialize them here.
qf1 = QNetwork(envs).to(device)
qf1_target = QNetwork(envs).to(device)
qf1_target.load_state_dict(qf1.state_dict())

qf2 = QNetwork(envs).to(device)
qf2_target = QNetwork(envs).to(device)
qf2_target.load_state_dict(qf2.state_dict())

q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate) # TODO: Adjust for twin Q
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

rb = ReplayBuffer(args.buffer_size, np.prod(envs.single_observation_space.shape), np.prod(envs.single_action_space.shape), args.num_envs, device)

# ── Training Loop ──────────────────────────────────────────────────────────────
obs, _ = envs.reset(seed=args.seed)
for global_step in range(args.total_timesteps):
    if global_step < args.learning_starts:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.randn_like(actions) * actor.action_scale * args.exploration_noise
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    
    # Handle Final Observation for truncations
    real_next_obs = next_obs.copy()
    if "final_observation" in infos:
        for idx, trunc in enumerate(truncations):
            if trunc: real_next_obs[idx] = infos["final_observation"][idx]

    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
    obs = next_obs

    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)

        with torch.no_grad():
            # TODO: Implement Target Policy Smoothing. 
            # 1. Get next_state_actions from target_actor
            # 2. Add clipped noise to these actions
            # 3. Clip the final actions to the env's valid range

            next_state_actions = target_actor(data.next_observations)
            smoothing_noise = torch.randn()


            # TODO: Implement Clipped Double Q-Learning for the target.
            # Use the minimum of the two target Q-networks.
            qf1_next_target = qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = qf2_target(data.next_observations, next_state_actions)

            min_qf_target = torch.min(qf1_next_target, qf2_next_target)

            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_target.view(-1)

        # TODO: Update the Twin Q-networks
        qf1_values = qf1(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_values, next_q_value)
        
        qf2_values = qf2(data.observations, data.actions).view(-1)
        

        q_optimizer.zero_grad()
        qf1_loss.backward()
        q_optimizer.step()

        # TODO: Implement Delayed Policy Updates
        if global_step % args.policy_frequency == 0:
            # Update Actor
            actor_loss = -qf1(data.observations, actor(data.observations)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update Target Networks (Soft Update)
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

envs.close()
