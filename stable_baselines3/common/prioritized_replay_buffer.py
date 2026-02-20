from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import PrioritizedReplayBufferSamples
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import VecNormalize


class SumTree:
    """
    SumTree data structure for prioritized replay.

    This structure supports O(log n) updates and O(log n) sampling
    and stores transition priorities over leaves.

    :param buffer_size: Maximum number of transitions stored in the tree.
    """

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.nodes = np.zeros(2 * buffer_size - 1, dtype=np.float64)
        self.data = np.zeros(buffer_size, dtype=np.int64)
        self.pos = 0
        self.full = False

    @property
    def total_sum(self) -> float:
        """
        Return the sum of all priorities in the tree.
        """
        return float(self.nodes[0])

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Update a leaf-tree index priority.

        :param tree_idx: Leaf-tree index in ``[buffer_size - 1, 2*buffer_size - 1)``.
        :param priority: New priority value.
        """
        delta = priority - self.nodes[tree_idx]
        self.nodes[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.nodes[tree_idx] += delta

    def add(self, idx: int, priority: float) -> int:
        """
        Add or overwrite a transition priority.

        :param idx: Position used for next insertion.
        :param priority: Transition priority.
        """
        self.data[self.pos] = idx
        leaf_idx = self.pos + self.buffer_size - 1
        self.update(leaf_idx, priority)
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
        return leaf_idx

    def get(self, cumulative_sum: float) -> tuple[int, float, int]:
        """
        Get a leaf index and its corresponding data index from a cumulative sum.

        :param cumulative_sum: cumulative priority.
        :return: leaf index, priority and data index.
        """
        assert 0 <= cumulative_sum <= self.total_sum

        idx = 0
        while idx < self.buffer_size - 1:
            left = 2 * idx + 1
            if cumulative_sum <= self.nodes[left]:
                idx = left
            else:
                cumulative_sum -= self.nodes[left]
                idx = left + 1
        data_idx = idx - self.buffer_size + 1
        return idx, float(self.nodes[idx]), int(self.data[data_idx])


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Proportional Prioritized Replay Buffer (PER).

    Paper: https://arxiv.org/abs/1511.05952
    This implementation is inspired by the proportional variant from
    ``prioritized_experience_replay``.

    :param buffer_size: Maximum number of element in the buffer.
    :param observation_space: Observation space.
    :param action_space: Action space.
    :param device: Device on which tensors are stored and returned.
    :param n_envs: Number of parallel environments.
    :param alpha: Prioritization exponent (0: uniform sampling, 1: full prioritization).
    :param beta: Importance sampling initial exponent.
    :param final_beta: Final importance sampling exponent.
    :param optimize_memory_usage: Not supported (not compatible with PER).
    :param min_priority: Minimum transition priority.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        final_beta: float = 1.0,
        optimize_memory_usage: bool = False,
        min_priority: float = 1e-6,
    ):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs)

        if optimize_memory_usage:
            raise AssertionError("PrioritizedReplayBuffer doesn't support optimize_memory_usage=True")

        self.min_priority = min_priority
        self.alpha = alpha
        self.initial_beta = beta
        self.final_beta = final_beta
        self._max_priority = min_priority
        # Training progress in SB3 starts from 1 and goes to 0.
        self._current_progress_remaining = 1.0
        self.beta_schedule = get_linear_fn(self.initial_beta, self.final_beta, end_fraction=1.0)
        # SumTree stores per-env transition priorities.
        self.tree = SumTree(self.buffer_size * self.n_envs)
        self._transition_to_leaf = np.full(self.buffer_size * self.n_envs, -1, dtype=np.int64)

    @property
    def beta(self) -> float:
        return self.beta_schedule(self._current_progress_remaining)

    @property
    def _n_stored(self) -> int:
        return self.size() * self.n_envs

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """
        Add a transition to the buffer.
        """
        for env_idx in range(self.n_envs):
            transition_idx = self.pos * self.n_envs + env_idx
            leaf_idx = self.tree.add(transition_idx, self._max_priority)
            self._transition_to_leaf[transition_idx] = leaf_idx

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> PrioritizedReplayBufferSamples:
        """
        Sample from the prioritized buffer.
        """
        assert self._n_stored >= batch_size, "The buffer contains fewer transitions than the batch size requires."

        # Divide [0, total_sum] into equal ranges for stable stratified sampling.
        segment = self.tree.total_sum / batch_size
        priorities = np.zeros((batch_size, 1), dtype=np.float64)
        transition_data_indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)

        for sample_idx in range(batch_size):
            cumulative_sum = np.random.uniform(segment * sample_idx, segment * (sample_idx + 1))
            _, priority, transition_data_idx = self.tree.get(cumulative_sum)
            tree_indices[sample_idx] = transition_data_idx
            priorities[sample_idx] = priority
            transition_data_indices[sample_idx] = transition_data_idx

        env_indices = transition_data_indices % self.n_envs
        transition_indices = transition_data_indices // self.n_envs

        probs = priorities / self.tree.total_sum
        # Importance sampling correction.
        weights = np.power(self._n_stored * probs, -self.beta)
        weights = weights / weights.max()

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(transition_indices + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(
                self.next_observations[transition_indices, env_indices, :],
                env,
            )

        batch = (
            self._normalize_obs(self.observations[transition_indices, env_indices, :], env),
            self.actions[transition_indices, env_indices, :],
            next_obs,
            # Ignore time truncation when using done signal.
            (
                self.dones[transition_indices, env_indices]
                * (1 - self.timeouts[transition_indices, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(self.rewards[transition_indices, env_indices].reshape(-1, 1), env),
            self.to_torch(weights),
        )
        return PrioritizedReplayBufferSamples(*tuple(map(self.to_torch, batch)), leaf_nodes_indices=tree_indices)

    def update_priorities(
        self,
        leaf_nodes_indices: np.ndarray,
        td_errors: np.ndarray | th.Tensor,
        progress_remaining: float,
    ) -> None:
        """
        Update transition priorities after computing TD-errors.
        """
        self._current_progress_remaining = progress_remaining
        if isinstance(td_errors, th.Tensor):
            td_errors = td_errors.detach().cpu().numpy().flatten()

        for transition_idx, td_error in zip(leaf_nodes_indices, td_errors, strict=True):
            tree_idx = self._transition_to_leaf[int(transition_idx)]
            if tree_idx == -1:
                continue
            priority = (abs(float(td_error)) + self.min_priority) ** self.alpha
            self.tree.update(int(tree_idx), priority)
            self._max_priority = max(self._max_priority, priority)
