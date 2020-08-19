from typing import Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.vec_env.dict_obs_wrapper import ObsWrapper
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


class HerReplayBuffer(BaseBuffer):
    """
    Replay Buffer for online Hindsight Experience Replay (HER)

    :param env: (VecEnv) The training environment
    :param buffer_size: (int) The size of the buffer measured in transitions.
    :param goal_selection_strategy: (GoalSelectionStrategy ) Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future', 'random']
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[th.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    :her_ratio: (float) The ratio between HER replays and regular replays in percent (between 0 and 1, for online sampling)

    """

    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        goal_selection_strategy: GoalSelectionStrategy,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        her_ratio: float = 0.6,
    ):

        super(HerReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs)

        self.env = env
        self.buffer_size = buffer_size

        # buffer with episodes
        self.buffer = []
        self.goal_selection_strategy = goal_selection_strategy
        # percentage of her indices
        self.her_ratio = her_ratio

        # memory management
        self._n_episodes_stored = 0
        self._n_transitions_stored = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (ReplayBufferSamples)
        """
        return self._sample_transitions(batch_size)

    def _sample_transitions(self, batch_size: int) -> ReplayBufferSamples:
        """
        :param batch_size: (int) Number of element to sample
        :return: (ReplayBufferSamples)
        """
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, self.n_episodes_stored, batch_size)
        buffer = np.array(self.buffer, dtype=object)
        # get episode lengths for selecting timesteps
        episode_lengths = np.array([len(ep) for ep in buffer[episode_idxs]])
        # select timesteps of episodes
        t_samples = np.array([np.random.choice(np.arange(ep_len)) for ep_len in episode_lengths])
        # get selected timesteps
        transitions = np.array([buffer[ep][trans] for ep, trans in zip(episode_idxs, t_samples)], dtype=object)
        # get her samples indices with her_ratio
        her_idxs = np.random.choice(np.arange(batch_size), int(self.her_ratio * batch_size), replace=False)

        # if we sample goals from future delete indices from her_idxs where we have no transition after current one
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            her_idxs = her_idxs[t_samples[her_idxs] != episode_lengths[her_idxs] - 1]

        # get new goals with goal selection strategy
        her_new_goals = [
            self.sample_goal(self.goal_selection_strategy, trans_idx, episode, self.buffer, online_sampling=True)
            for episode, trans_idx in zip(buffer[episode_idxs[her_idxs]], t_samples[her_idxs])
        ]

        # assign new goals as desired_goals
        for idx, goal in enumerate(her_new_goals):
            transitions[her_idxs][:, 0][idx]["desired_goal"] = goal

        observations, actions, rewards, next_observations, dones = list(zip(*transitions))

        # compute new rewards with new goal
        achieved_goals = [new_obs["achieved_goal"] for new_obs in np.array(next_observations)[her_idxs]]
        new_rewards = np.array(rewards)
        new_rewards[her_idxs] = [
            self.env.env_method("compute_reward", achieved_goal, new_goal, None)
            for achieved_goal, new_goal in zip(achieved_goals, her_new_goals)
        ]

        # concatenate observation with (desired) goal
        obs = [ObsWrapper.convert_dict(obs_) for obs_ in observations]
        new_obs = [ObsWrapper.convert_dict(new_obs_) for new_obs_ in next_observations]

        data = (
            np.array(obs)[:, 0, :],
            np.array(actions, dtype=self.action_space.dtype)[:, 0, :],
            np.array(new_obs)[:, 0, :],
            np.array(dones, dtype=np.int8),
            new_rewards,
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def sample_goal(
        goal_selection_strategy: GoalSelectionStrategy,
        sample_idx: int,
        episode: list,
        observations: Union[list, np.ndarray],
        obs_dim: int = None,
        online_sampling: bool = False,
    ) -> Union[np.ndarray, None]:
        """
        Sample a goal based on goal_selection_strategy.

        :param goal_selection_strategy: (GoalSelectionStrategy ) Strategy for sampling goals for replay.
            One of ['episode', 'final', 'future', 'random']
        :param sample_idx: (int) Index of current transition.
        :param episode: (list) Current episode.
        :param observations: (list or np.ndarray)
        :param obs_dim: (int) Dimension of real observation without goal. It is needed for the random strategy.
        :param online_sampling: (bool) Sample HER transitions online.
        :return: (np.ndarray or None) Return sampled goal.
        """
        if goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            return episode[-1][0]["achieved_goal"]
        elif goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            # we have no transition after last transition of episode
            if (sample_idx + 1) < len(episode):
                index = np.random.choice(np.arange(sample_idx + 1, len(episode)))
                return episode[index][0]["achieved_goal"]
        elif goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            index = np.random.choice(np.arange(len(episode)))
            return episode[index][0]["achieved_goal"]
        elif goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            if online_sampling:
                # replay with random state from the entire replay buffer
                ep_idx = np.random.choice(np.arange(len(observations)))
                trans_idx = np.random.choice(np.arange(len(observations[ep_idx])))
                return observations[ep_idx][trans_idx][0]["achieved_goal"]
            else:
                # replay with random state from the entire replay buffer
                index = np.random.choice(np.arange(len(observations)))
                obs = observations[index]
                # get only the observation part
                obs_array = obs[:, :obs_dim]
                return obs_array
        else:
            raise ValueError("Strategy for sampling goals not supported!")

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        """
        Add episode to replay buffer

        :param obs:
        :param next_obs:
        :param action:
        :param reward:
        :param done:

        :param episode: (list) Episode to store.
        """
        episode = list(zip(obs, action, reward, next_obs, done))

        episode_length = len(episode)

        # check if replay buffer has enough space for all transitions of episode
        if self.n_transitions_stored + episode_length <= self.buffer_size:
            self.buffer.append(episode)
            # update replay size
            self.n_episodes_stored += 1
            self.n_transitions_stored += episode_length
        elif self.full:
            # if replay buffer is full take random stored episode and replace it
            idx = np.random.randint(0, self.n_episodes_stored)

            if len(self.buffer[idx]) == episode_length:
                self.buffer[idx] = episode
            elif len(self.buffer[idx]) > episode_length:
                self.buffer[idx] = episode
                self.n_transitions_stored -= len(self.buffer[idx]) - episode_length

        if self.n_transitions_stored == self.buffer_size:
            self.full = True

    @property
    def n_episodes_stored(self):
        return self._n_episodes_stored

    @n_episodes_stored.setter
    def n_episodes_stored(self, n):
        self._n_episodes_stored = n

    @property
    def n_transitions_stored(self):
        return self._n_transitions_stored

    @n_transitions_stored.setter
    def n_transitions_stored(self, n):
        self._n_transitions_stored = n

    def clear_buffer(self):
        self.buffer = []
        self.n_episodes_stored = 0
        self.n_transitions_stored = 0

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer in transitions.
        """
        return self.n_transitions_stored
