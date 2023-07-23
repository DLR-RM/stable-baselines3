import random
from typing import Any, Dict, List, Optional, Union
import warnings
from gymnasium import spaces
import numpy as np
import torch as th
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
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size
        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def p_total(self):
        return self.nodes[0]

    def update(self, data_idx: int, value: float):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value: float, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.p_total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

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
        self.beta = beta  # determines the amount of importance-sampling correction, beta = 1 fully compensate for the non-uniform probabilities
        self.max_priority = self.eps  # priority for new samples, init as eps

        # SumTree: data structure to store priorities
        self.tree = SumTree(size=buffer_size)

        self.real_size = 0
        self.count = 0
    
    def add(self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
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
        :return:
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
            sample_idxs.append(sample_idx)

        # probability of sampling transition i as P(i) = p_i^alpha / \sum_{k} p_k^alpha
        # where p_i > 0 is the priority of transition i.
        probs = priorities / self.tree.p_total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).

        # Importance sampling weights.
        # All weights w_i were scaled so that max_i w_i = 1.
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = ReplayBufferSamples(
            self.observations[sample_idxs],
            self.actions[sample_idxs],
            self.next_observations[sample_idxs],
            self.dones[sample_idxs],
            self.rewards[sample_idxs],
        )
        return batch, weights, tree_idxs
