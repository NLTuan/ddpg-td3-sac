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
    env_id: str = "Hopper-v5"
    num_envs: int = 4
    total_timesteps: int = 1_000_000
    buffer_size: int = int(1e6)
    
    # TD3 Hyperparameters (Paper Defaults)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 25_000
    policy_frequency: int = 2
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5

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
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % 50 == 0)
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

qf1 = QNetwork(envs).to(device)
qf2 = QNetwork(envs).to(device)
qf1_target = QNetwork(envs).to(device)
qf2_target = QNetwork(envs).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

rb = ReplayBuffer(args.buffer_size, np.prod(envs.single_observation_space.shape), np.prod(envs.single_action_space.shape), args.num_envs, device)

# Pre-calculate bounds for efficiency
action_low = torch.tensor(envs.single_action_space.low).to(device)
action_high = torch.tensor(envs.single_action_space.high).to(device)

# ── Training Loop ──────────────────────────────────────────────────────────────
obs, _ = envs.reset(seed=args.seed)
for global_step in range(args.total_timesteps):
    if global_step < args.learning_starts:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.randn_like(actions) * actor.action_scale * args.exploration_noise
            actions = actions.clamp(action_low, action_high).cpu().numpy()

    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    
    if "episode" in infos:
        for idx in np.where(infos["_episode"])[0]:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r'][idx]:.2f}")

    real_next_obs = next_obs.copy()
    if "final_observation" in infos:
        for idx, trunc in enumerate(truncations):
            if trunc: real_next_obs[idx] = infos["final_observation"][idx]

    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
    obs = next_obs

    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)

        with torch.no_grad():
            # Target Policy Smoothing
            noise = (torch.randn_like(data.actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
            next_state_actions = (target_actor(data.next_observations) + noise).clamp(action_low, action_high)
            
            # Clipped Double Q-Learning
            qf1_next_target = qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

        qf1_values = qf1(data.observations, data.actions).view(-1)
        qf2_values = qf2(data.observations, data.actions).view(-1)
        qf_loss = F.mse_loss(qf1_values, next_q_value) + F.mse_loss(qf2_values, next_q_value)
        
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:
            # Delayed Actor Update
            actor_loss = -qf1(data.observations, actor(data.observations)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft Update Targets
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

envs.close()

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\nRunning final evaluation...")
eval_env = gym.make(args.env_id, render_mode="rgb_array")
eval_env = gym.wrappers.RecordVideo(eval_env, f"videos/{run_name}/eval", episode_trigger=lambda x: True)
eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

obs, _ = eval_env.reset(seed=args.seed)
done = False
while not done:
    with torch.no_grad():
        actions = actor(torch.Tensor(obs).to(device))
        actions = actions.cpu().numpy()
    next_obs, reward, terminations, truncations, infos = eval_env.step(actions)
    obs = next_obs
    done = terminations or truncations

print(f"Evaluation finished. Episode return: {infos['episode']['r']:.2f}")
eval_env.close()
