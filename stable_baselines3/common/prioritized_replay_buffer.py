from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class SumTree:
    """
    SumTree data structure for Prioritized Replay Buffer.
    This code is inspired by: https://github.com/Howuhh/prioritized_experience_replay

    :param buffer_size: Max number of element in the buffer.
    """

    def __init__(self, buffer_size: int) -> None:
        self.nodes = np.zeros(2 * buffer_size - 1)
        # The data array stores transition indices
        self.data = np.zeros(buffer_size)
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

    @property
    def total_sum(self) -> float:
        """
        Returns the root node value, which represents the total sum of all priorities in the tree.

        :return: Total sum of all priorities in the tree.
        """
        return self.nodes[0].item()

    def update(self, leaf_node_idx: int, value: float) -> None:
        """
        Update the priority of a leaf node.

        :param leaf_node_idx: Index of the leaf node to update.
        :param value: New priority value.
        """
        idx = leaf_node_idx + self.buffer_size - 1  # child index in tree array
        change = value - self.nodes[idx]
        self.nodes[idx] = value
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value: float, data: int) -> None:
        """
        Add a new transition with priority value,
        it adds a new leaf node and update cumulative sum.

        :param value: Priority value.
        :param data: Data for the new leaf node, storing transition index
            in the case of the prioritized replay buffer.
        """
        # Note: transition_indices should be constant
        # as the replay buffer already updates a pointer
        self.data[self.pos] = data
        self.update(self.pos, value)
        self.pos = (self.pos + 1) % self.buffer_size

    def get(self, cumulative_sum: float) -> Tuple[int, float, th.Tensor]:
        """
        Get a leaf node index, its priority value and transition index by cumulative_sum value.

        :param cumulative_sum: Cumulative sum value.
        :return: Leaf node index, its priority value and transition index.
        """
        assert cumulative_sum <= self.total_sum

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2
            if cumulative_sum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumulative_sum = cumulative_sum - self.nodes[left]

        leaf_node_idx = idx - self.buffer_size + 1
        return leaf_node_idx, self.nodes[idx].item(), self.data[leaf_node_idx]

    def __repr__(self) -> str:
        return f"SumTree(nodes={self.nodes!r}, data={self.data!r})"


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer (proportional priorities version).
    Paper: https://arxiv.org/abs/1511.05952
    This code is inspired by: https://github.com/Howuhh/prioritized_experience_replay

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param alpha: How much prioritization is used (0 - no prioritization aka uniform case, 1 - full prioritization)
    :param beta: To what degree to use importance weights (0 - no corrections, 1 - full correction)
    :param final_beta: Value of beta at the end of training.
        Linear annealing is used to interpolate between initial value of beta and final beta.
    :param min_priority: Minimum priority, prevents zero probabilities, so that all samples
        always have a non-zero probability to be sampled.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.5,
        beta: float = 0.4,
        final_beta: float = 1.0,
        optimize_memory_usage: bool = False,
        min_priority: float = 1e-8,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)

        assert optimize_memory_usage is False, "PrioritizedReplayBuffer doesn't support optimize_memory_usage=True"

        self.min_priority = min_priority
        self.alpha = alpha
        self.max_priority = self.min_priority  # priority for new samples, init as eps
        # Track the training progress remaining (from 1 to 0)
        # this is used to update beta
        self._current_progress_remaining = 1.0
        self.inital_beta = beta
        self.final_beta = final_beta
        self.beta_schedule = get_linear_fn(
            self.inital_beta,
            self.final_beta,
            end_fraction=1.0,
        )
        # SumTree: data structure to store priorities
        self.tree = SumTree(buffer_size=buffer_size)

    @property
    def beta(self) -> float:
        # Linear schedule
        return self.beta_schedule(self._current_progress_remaining)

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
        self.tree.add(self.max_priority, self.pos)

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

        leaf_nodes_indices = np.zeros(batch_size, dtype=np.uint32)
        priorities = np.zeros((batch_size, 1))
        sample_indices = np.zeros(batch_size, dtype=np.uint32)

        # To sample a minibatch of size k, the range [0, total_sum] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree.
        segment_size = self.tree.total_sum / batch_size
        for batch_idx in range(batch_size):
            # extremes of the current segment
            start, end = segment_size * batch_idx, segment_size * (batch_idx + 1)

            # uniformely sample a value from the current segment
            cumulative_sum = np.random.uniform(start, end)

            # leaf_node_idx is a index of a sample in the tree, needed further to update priorities
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            leaf_node_idx, priority, sample_idx = self.tree.get(cumulative_sum)

            leaf_nodes_indices[batch_idx] = leaf_node_idx
            priorities[batch_idx] = priority
            sample_indices[batch_idx] = sample_idx

        # probability of sampling transition i as P(i) = p_i^alpha / \sum_{k} p_k^alpha
        # where p_i > 0 is the priority of transition i.
        probs = priorities / self.tree.total_sum

        # Importance sampling weights.
        # All weights w_i were scaled so that max_i w_i = 1.
        weights = (self.size() * probs) ** -self.beta
        weights = weights / weights.max()

        # TODO: add proper support for multi env
        # env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size,))
        env_indices = np.zeros(batch_size, dtype=np.uint32)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(sample_indices + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[sample_indices, env_indices, :], env)

        batch = (
            self._normalize_obs(self.observations[sample_indices, env_indices, :], env),
            self.actions[sample_indices, env_indices, :],
            next_obs,
            self.dones[sample_indices],
            self.rewards[sample_indices],
            weights,
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, batch)), leaf_nodes_indices)  # type: ignore[arg-type,call-arg]

    def update_priorities(self, leaf_nodes_indices: np.ndarray, td_errors: th.Tensor, progress_remaining: float) -> None:
        """
        Update transition priorities.

        :param leaf_nodes_indices: Indices for the leaf nodes to update
            (correponding to the transitions)
        :param td_errors: New priorities, td error in the case of
            proportional prioritized replay buffer.
        :param progress_remaining: Current progress remaining (starts from 1 and ends to 0)
            to linearly anneal beta from its start value to 1.0 at the end of training
        """
        # Update beta schedule
        self._current_progress_remaining = progress_remaining
        td_errors = td_errors.detach().cpu().numpy().flatten()

        for leaf_node_idx, td_error in zip(leaf_nodes_indices, td_errors):
            # Proportional prioritization priority = (abs(td_error) + eps) ^ alpha
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.tree.update(leaf_node_idx, priority)
            # Update max priority for new samples
            self.max_priority = max(self.max_priority, priority)
