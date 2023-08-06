import random
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class SumTree:
    """
    SumTree data structure for Prioritized Replay Buffer.
    This code is inspired by: https://github.com/Howuhh/prioritized_experience_replay

    :param size: Max number of element in the buffer.
    """

    def __init__(self, size: int):
        self.nodes = th.zeros(2 * size - 1)
        self.data = th.empty(size)
        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def p_total(self) -> float:
        """
        Returns the root node value, which represents the total sum of all priorities in the tree.

        :return: Total sum of all priorities in the tree.
        """
        return self.nodes[0].item()

    def update(self, data_idx: int, value: float):
        """
        Update the priority of a leaf node.

        :param data_idx: Index of the leaf node to update.
        :param value: New priority value.
        """
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]
        self.nodes[idx] = value
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value: float, data: int):
        """
        Add a new transition with priority value.

        :param value: Priority value.
        :param data: Transition data.
        """
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum) -> tuple[int, float, th.Tensor]:
        """
        Get a leaf node index, its priority value and transition data by cumsum value.

        :param cumsum: Cumulative sum value.
        :return: Leaf node index, its priority value and transition data.
        """
        assert cumsum <= self.p_total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2
            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx].item(), self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer.
    Paper: https://arxiv.org/abs/1511.05952
    This code is inspired by: https://github.com/Howuhh/prioritized_experience_replay

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization)
    :param beta: To what degree to use importance weights (0 - no corrections, 1 - full correction)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        optimize_memory_usage: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)

        # TODO: check this
        if optimize_memory_usage:
            warnings.warn("PrioritizedReplayBuffer does not support optimize_memory_usage=True during sampling")

        # PER params
        self.eps = 1e-8  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, alpha = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction
        self.max_priority = self.eps  # priority for new samples, init as eps

        # SumTree: data structure to store priorities
        self.tree = SumTree(size=buffer_size)

        self.real_size = 0
        self.count = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new transition to the buffer.

        :param obs: Starting observation of the transition to be stored.
        :param next_obs: Destination observation of the transition to be stored.
        :param action: Action performed in the transition to be stored.
        :param reward: Reward received in the transition to be stored.
        :param done: Whether the episode was finished after the transition to be stored.
        :param infos: Eventual information given by the environment.
        """
        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # update counters
        self.count = (self.count + 1) % self.buffer_size
        self.real_size = min(self.buffer_size, self.real_size + 1)

        # store transition in the buffer
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the prioritized replay buffer.

        :param batch_size: Number of element to sample
        :param env:associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: a batch of sampled experiences from the buffer.
        """
        assert self.buffer_size >= batch_size, "The buffer contains less samples than the batch size requires."

        sample_idxs, tree_idxs = [], []
        priorities = th.empty(batch_size, 1, dtype=th.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree.
        segment = self.tree.p_total / batch_size
        for i in range(batch_size):
            # extremes of the current segment
            a, b = segment * i, segment * (i + 1)

            # uniformely sample a value from the current segment
            cumsum = random.uniform(a, b)

            # tree_idx is a index of a sample in the tree, needed further to update priorities
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(int(sample_idx.item()))

        # probability of sampling transition i as P(i) = p_i^alpha / \sum_{k} p_k^alpha
        # where p_i > 0 is the priority of transition i.
        probs = priorities / self.tree.p_total

        # Importance sampling weights.
        # All weights w_i were scaled so that max_i w_i = 1.
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        env_indices = np.random.randint(0, high=self.n_envs, size=(len(sample_idxs),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(np.array(sample_idxs) + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[sample_idxs, env_indices, :], env)

        batch = (
            self._normalize_obs(self.observations[sample_idxs, env_indices, :], env),
            self.actions[sample_idxs, env_indices, :],
            next_obs,
            self.dones[sample_idxs],
            self.rewards[sample_idxs],
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, batch)))  # type: ignore
