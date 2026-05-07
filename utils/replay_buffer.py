from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """
    A simple CleanRL-style replay buffer for off-policy algorithms (DDPG, TD3, SAC).

    Stores transitions as flat numpy arrays and supports vectorized environments
    by writing a block of n_envs transitions per call to add().

    Args:
        buffer_size: Maximum number of transitions to store.
        obs_dim: Dimensionality of a single observation.
        action_dim: Dimensionality of a single action.
        n_envs: Number of parallel environments (transitions added per step).
        device: PyTorch device to send sampled tensors to.
        handle_timeout_termination: If True, extract ``TimeLimit.truncated`` from
            infos and store it separately so that time-limit truncations are not
            treated as true terminal states during critic updates.
            See https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        n_envs: int = 1,
        device: torch.device | str = "cpu",
        handle_timeout_termination: bool = True,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.device = torch.device(device)

        self.pos = 0       # next write position
        self.size = 0      # number of valid transitions currently stored

        # Pre-allocate storage as flat (buffer_size, dim) arrays.
        # Each row is one transition regardless of which env it came from.
        self.handle_timeout_termination = handle_timeout_termination

        self.observations      = np.zeros((buffer_size, obs_dim),    dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim),    dtype=np.float32)
        self.actions           = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards           = np.zeros((buffer_size, 1),          dtype=np.float32)
        self.dones             = np.zeros((buffer_size, 1),          dtype=np.float32)
        # Stores whether a transition ended due to a time limit (truncation).
        # Kept separate so the critic can treat them as non-terminal.
        self.timeouts          = np.zeros((buffer_size, 1),          dtype=np.float32)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,                    # (n_envs, obs_dim)
        next_obs: np.ndarray,               # (n_envs, obs_dim)
        action: np.ndarray,                 # (n_envs, action_dim)
        reward: np.ndarray,                 # (n_envs,)
        done: np.ndarray,                   # (n_envs,)
        infos: list[dict[str, Any]] | None = None,  # per-env info dicts from env.step()
    ) -> None:
        """Write one step of experience from all parallel environments."""
        # Number of transitions we're inserting (== n_envs in normal use)
        n = len(obs)

        # Indices into the circular buffer where we'll write.
        # np.arange + modulo handles the wrap-around case automatically.
        idxs = np.arange(self.pos, self.pos + n) % self.buffer_size

        self.observations[idxs]      = obs
        self.next_observations[idxs] = next_obs
        self.actions[idxs]           = action
        self.rewards[idxs]           = reward.reshape(-1, 1)
        self.dones[idxs]             = done.reshape(-1, 1)

        if self.handle_timeout_termination and infos is not None:
            # Extract the TimeLimit.truncated flag set by gymnasium's TimeLimit wrapper.
            # A truncated episode ended due to a time limit, NOT a true terminal state,
            # so the critic should bootstrap from next_obs rather than treating it as done.
            #
            # gymnasium's SyncVectorEnv returns infos as a dict of arrays, e.g.:
            #   infos["TimeLimit.truncated"] -> ndarray of bool, shape (n_envs,)
            # Older / non-vectorized envs return a list[dict] instead.
            if isinstance(infos, dict):
                truncated = np.array(
                    infos.get("TimeLimit.truncated", np.zeros(n, dtype=bool)),
                    dtype=np.float32,
                )
            else:
                truncated = np.array(
                    [info.get("TimeLimit.truncated", False) for info in infos],
                    dtype=np.float32,
                )
            self.timeouts[idxs] = truncated.reshape(-1, 1)

        # Advance the circular pointer and track how full the buffer is.
        self.pos = (self.pos + n) % self.buffer_size
        self.size = min(self.size + n, self.buffer_size)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a random mini-batch of transitions."""
        assert self.size >= batch_size, (
            f"Cannot sample {batch_size} transitions from a buffer with only {self.size} stored."
        )
        idxs = np.random.randint(0, self.size, size=batch_size)

        return ReplayBufferSamples(
            observations=self._to_tensor(self.observations[idxs]),
            actions=self._to_tensor(self.actions[idxs]),
            next_observations=self._to_tensor(self.next_observations[idxs]),
            # Mask out timeout truncations: treat them as non-terminal for the critic.
            dones=self._to_tensor(
                self.dones[idxs] * (1 - self.timeouts[idxs])
                if self.handle_timeout_termination
                else self.dones[idxs]
            ),
            rewards=self._to_tensor(self.rewards[idxs]),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self.size}/{self.buffer_size}, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"n_envs={self.n_envs})"
        )