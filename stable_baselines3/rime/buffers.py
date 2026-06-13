"""RIME-PPO Rollout Buffer: Extended buffer with editing state and chunk BPTT support.

Stores additional fields needed by RIME-PPO:
- edit_states: editing state e_t at each step
- psis: surprise signal psi_t
- rhos: standardized TD error rho_t
- obs_emas: EMA of observations
- prev_actions: previous action for context
- chunk_size: for BPTT chunking
"""

from collections.abc import Generator
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import get_device


class RIMERolloutBufferSamples:
    """Container for a chunk of RIME rollout data for BPTT training."""

    def __init__(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        old_values: th.Tensor,
        old_log_prob: th.Tensor,
        advantages: th.Tensor,
        returns: th.Tensor,
        # RIME-specific fields
        edit_states: th.Tensor,  # e_t: (chunk_size, n_envs, S)
        psis: th.Tensor,  # psi_t: (chunk_size, n_envs)
        rhos: th.Tensor,  # rho_t: (chunk_size, n_envs)
        obs_emas: th.Tensor,  # obs EMA: (chunk_size, n_envs, obs_dim)
        prev_actions: th.Tensor,  # prev action: (chunk_size, n_envs, act_dim)
        episode_starts: th.Tensor,  # for resetting e_t at boundaries
        e_init: th.Tensor,  # initial e_t for this chunk: (n_envs, S)
    ):
        self.observations = observations
        self.actions = actions
        self.old_values = old_values
        self.old_log_prob = old_log_prob
        self.advantages = advantages
        self.returns = returns
        self.edit_states = edit_states
        self.psis = psis
        self.rhos = rhos
        self.obs_emas = obs_emas
        self.prev_actions = prev_actions
        self.episode_starts = episode_starts
        self.e_init = e_init


class RIMERolloutBuffer(RolloutBuffer):
    """Extended RolloutBuffer for RIME-PPO with editing state tracking and chunk BPTT.

    Additional fields beyond standard RolloutBuffer:
    - edit_states: (buffer_size, n_envs, S)
    - psis: (buffer_size, n_envs)
    - rhos: (buffer_size, n_envs)
    - obs_emas: (buffer_size, n_envs, obs_dim)
    - prev_actions: (buffer_size, n_envs, act_dim)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        # RIME-specific
        edit_dim: int = 32,
        chunk_size: int = 32,
    ):
        self.edit_dim = edit_dim
        self.chunk_size = chunk_size
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self) -> None:
        super().reset()
        self.edit_states = np.zeros((self.buffer_size, self.n_envs, self.edit_dim), dtype=np.float32)
        self.psis = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rhos = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.obs_emas = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.prev_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype
        )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        # RIME-specific
        edit_state: np.ndarray | None = None,
        psi: np.ndarray | None = None,
        rho: np.ndarray | None = None,
        obs_ema: np.ndarray | None = None,
        prev_action: np.ndarray | None = None,
    ) -> None:
        """Add transition to buffer with RIME-specific fields."""
        # Call parent add for standard fields
        super().add(obs, action, reward, episode_start, value, log_prob)

        # Store RIME-specific fields
        pos = self.pos - 1  # super().add already incremented pos
        if edit_state is not None:
            self.edit_states[pos] = np.array(edit_state)
        if psi is not None:
            self.psis[pos] = np.array(psi)
        if rho is not None:
            self.rhos[pos] = np.array(rho)
        if obs_ema is not None:
            self.obs_emas[pos] = np.array(obs_ema)
        if prev_action is not None:
            self.prev_actions[pos] = np.array(prev_action)

    def get_chunks(self, chunk_size: int | None = None) -> Generator[RIMERolloutBufferSamples, None, None]:
        """Generate chunks of sequential data for BPTT training.

        Unlike standard get() which shuffles, this preserves temporal order
        and provides initial e_t for each chunk.

        Args:
            chunk_size: Size of each chunk. Defaults to self.chunk_size.

        Yields:
            RIMERolloutBufferSamples with sequential data for BPTT.
        """
        assert self.full, "Buffer must be full before getting chunks"

        if chunk_size is None:
            chunk_size = self.chunk_size

        device = get_device(self.device)

        # Process each environment separately to maintain temporal order
        for env_idx in range(self.n_envs):
            e_init = th.zeros(self.edit_dim, device=device)

            for start in range(0, self.buffer_size, chunk_size):
                end = min(start + chunk_size, self.buffer_size)
                actual_chunk = end - start

                # Get sequential slice for this env
                obs_chunk = th.tensor(self.observations[start:end, env_idx], device=device, dtype=th.float32)
                actions_chunk = th.tensor(self.actions[start:end, env_idx], device=device, dtype=th.float32)
                values_chunk = th.tensor(self.values[start:end, env_idx], device=device, dtype=th.float32)
                log_probs_chunk = th.tensor(self.log_probs[start:end, env_idx], device=device, dtype=th.float32)
                advantages_chunk = th.tensor(self.advantages[start:end, env_idx], device=device, dtype=th.float32)
                returns_chunk = th.tensor(self.returns[start:end, env_idx], device=device, dtype=th.float32)

                edit_states_chunk = th.tensor(self.edit_states[start:end, env_idx], device=device, dtype=th.float32)
                psis_chunk = th.tensor(self.psis[start:end, env_idx], device=device, dtype=th.float32)
                rhos_chunk = th.tensor(self.rhos[start:end, env_idx], device=device, dtype=th.float32)
                obs_emas_chunk = th.tensor(self.obs_emas[start:end, env_idx], device=device, dtype=th.float32)
                prev_actions_chunk = th.tensor(self.prev_actions[start:end, env_idx], device=device, dtype=th.float32)
                episode_starts_chunk = th.tensor(self.episode_starts[start:end, env_idx], device=device, dtype=th.float32)

                yield RIMERolloutBufferSamples(
                    observations=obs_chunk,
                    actions=actions_chunk,
                    old_values=values_chunk,
                    old_log_prob=log_probs_chunk,
                    advantages=advantages_chunk,
                    returns=returns_chunk,
                    edit_states=edit_states_chunk,
                    psis=psis_chunk,
                    rhos=rhos_chunk,
                    obs_emas=obs_emas_chunk,
                    prev_actions=prev_actions_chunk,
                    episode_starts=episode_starts_chunk,
                    e_init=e_init.clone(),
                )

                # Update e_init for next chunk (detach to prevent gradient flow across chunks)
                with th.no_grad():
                    e_init = edit_states_chunk[-1].detach().clone()

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        """Standard shuffled get (for compatibility, but RIME-PPO uses get_chunks)."""
        # Fall back to parent implementation for non-BPTT use
        return super().get(batch_size)
