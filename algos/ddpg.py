import os

import gymnasium as gym

import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

from utils.replay_buffer import ReplayBuffer


class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the environment id"""
    total_timesteps: int = 100_000
    """total timesteps of the experiment"""
    learning_rate: float = 2e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 200
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    num_envs: int = 4
    """number of parallel environments"""


args = Args()
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"


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


# ── Setup ──────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)]
)

qf1         = QNetwork(envs).to(device)
qf1_target  = QNetwork(envs).to(device)
actor       = Actor(envs).to(device)
target_actor = Actor(envs).to(device)

qf1_target.load_state_dict(qf1.state_dict())
target_actor.load_state_dict(actor.state_dict())

q_optimizer     = optim.Adam(qf1.parameters(),   lr=args.learning_rate)
actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

envs.single_action_space.dtype = np.dtype("float32")
rb = ReplayBuffer(
    buffer_size=args.buffer_size,
    obs_dim=int(np.prod(envs.single_observation_space.shape)),
    action_dim=int(np.prod(envs.single_action_space.shape)),
    n_envs=args.num_envs,
    device=device,
)
# ──────────────────────────────────────────────────────────────────────────────


# ── Training loop ──────────────────────────────────────────────────────────────
obs, info = envs.reset()
for global_step in range(args.total_timesteps):
    if global_step < args.learning_starts:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

    else:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)  # keep obs as ndarray for rb.add()
            actions = actor(obs_tensor)
            actions += torch.normal(0, actor.action_scale * args.exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    if "episode" in infos:
        for idx in np.where(infos["_episode"])[0]:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r'][idx]:.2f}")

    real_next_obs = next_obs.copy()
    if "final_observation" in infos:
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    obs = next_obs

    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)

        with torch.no_grad():
            next_state_actions = target_actor(data.next_observations)
            qf1_next_target = qf1_target(data.next_observations, next_state_actions)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * qf1_next_target.view(-1)

        qf1_values = qf1(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_values, next_q_value)

        q_optimizer.zero_grad()
        qf1_loss.backward()
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:
            pi = actor(data.observations)
            qf1_pi_values = qf1(data.observations, pi)
            actor_loss = -qf1_pi_values.mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)