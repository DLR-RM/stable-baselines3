from typing import Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from gym.spaces import Discrete

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
    :param max_episode_length: (int) The length of an episode. (time horizon)
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
        max_episode_length: int,
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
        self.max_episode_length = max_episode_length

        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        n_episodes = self.buffer_size // self.max_episode_length
        # input dimensions for buffer initialization
        input_shape = {
            "observation": (self.env.num_envs, self.env.obs_dim),
            "achieved_goal": (self.env.num_envs, self.env.goal_dim),
            "desired_goal": (self.env.num_envs, self.env.goal_dim),
            "action": (self.action_dim,),
            "reward": (1,),
            "next_obs": (self.env.num_envs, self.env.obs_dim),
            "next_achieved_goal": (self.env.num_envs, self.env.goal_dim),
            "next_desired_goal": (self.env.num_envs, self.env.goal_dim),
            "done": (1,),
        }
        self.buffer = {
            key: np.empty([n_episodes, self.max_episode_length, *dim], dtype=np.float32) for key, dim in input_shape.items()
        }
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.empty(n_episodes)

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
        return self._sample_transitions(batch_size, env)

    def _sample_transitions(self, batch_size: int, env: Optional[VecNormalize]) -> ReplayBufferSamples:
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (ReplayBufferSamples)
        """
        # Select which episodes to use
        episode_idxs = np.random.randint(0, self.n_episodes_stored, batch_size)
        # select timesteps of episodes
        max_timestep_idx = self.episode_lengths[episode_idxs]
        # transition_idxs = np.random.randint(self.max_episode_length, size=batch_size)
        transition_idxs = np.random.randint(max_timestep_idx)
        # get selected timesteps
        transitions = {key: self.buffer[key][episode_idxs, transition_idxs].copy() for key in self.buffer.keys()}
        # get her samples indices with her_ratio
        her_idxs = np.random.choice(np.arange(batch_size), int(self.her_ratio * batch_size), replace=False)

        # if we sample goals from future delete indices from her_idxs where we have no transition after current one
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            her_idxs = her_idxs[transition_idxs[her_idxs] != max_timestep_idx[her_idxs] - 1]

        # get new goals with goal selection strategy
        her_new_goals = [
            self.sample_goal(self.goal_selection_strategy, trans, episode, self.buffer["achieved_goal"], online_sampling=True)
            for episode, trans in zip(self.buffer["achieved_goal"][episode_idxs[her_idxs]], transition_idxs[her_idxs])
        ]

        # assign new goals as desired_goals
        for idx, goal in enumerate(her_new_goals):
            # observation
            transitions["desired_goal"][her_idxs][idx] = goal
            # next observation
            transitions["next_desired_goal"][her_idxs][idx] = goal

        # compute new rewards with new goal
        achieved_goals = transitions["next_achieved_goal"][her_idxs]
        new_rewards = transitions["reward"].copy()
        new_rewards[her_idxs] = [
            self.env.env_method("compute_reward", achieved_goal, new_goal, None)
            for achieved_goal, new_goal in zip(achieved_goals, her_new_goals)
        ]

        # concatenate observation with (desired) goal
        obs = [
            np.concatenate([obs, desired_goal], axis=1)
            for obs, desired_goal in zip(transitions["observation"], transitions["desired_goal"])
        ]
        next_obs = [
            np.concatenate([obs, desired_goal], axis=1)
            for obs, desired_goal in zip(transitions["next_obs"], transitions["next_desired_goal"])
        ]

        data = (
            self._normalize_obs(np.asarray(obs, dtype=np.int8), env),
            transitions["action"],
            self._normalize_obs(np.asarray(next_obs, dtype=np.int8), env),
            transitions["done"],
            self._normalize_obs(new_rewards, env),
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
            if online_sampling:
                return episode[-1]
            return episode[-1][0]["achieved_goal"]
        elif goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            # we have no transition after last transition of episode
            if (sample_idx + 1) < len(episode):
                index = np.random.choice(np.arange(sample_idx + 1, len(episode)))
                if online_sampling:
                    return episode[index]
                return episode[index][0]["achieved_goal"]
        elif goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            index = np.random.choice(np.arange(len(episode)))
            if online_sampling:
                return episode[index]
            return episode[index][0]["achieved_goal"]
        elif goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            if online_sampling:
                # replay with random state from the entire replay buffer
                ep_idx = np.random.choice(np.arange(len(observations)))
                trans_idx = np.random.choice(np.arange(len(observations[ep_idx])))
                return observations[ep_idx][trans_idx]
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

        :param obs: (np.ndarray) Observation.
        :param next_obs: (np.ndarray) Next observation.
        :param action: (np.ndarray) Action.
        :param reward: (np.ndarray) Reward.
        :param done: (np.ndarray) Done.
        """
        episode_length = len(action)
        episode = self._get_episode_dict(obs, next_obs, action, reward, done)

        # check if replay buffer has enough space for all transitions of episode
        if self.n_transitions_stored + episode_length <= self.buffer_size:
            for key in self.buffer.keys():
                self.buffer[key][self._n_episodes_stored][:episode_length] = episode[key]
            # add episode length to length storage
            self.episode_lengths[self._n_episodes_stored] = episode_length
            # update replay size
            self.n_episodes_stored += 1
            self.n_transitions_stored += episode_length
        elif self.full:
            # if replay buffer is full take random stored episode and replace it
            idx = np.random.randint(0, self.n_episodes_stored)

            for key in self.buffer.keys():
                self.buffer[key][idx][:episode_length] = episode[key]
            # add episode length to length storage
            self.episode_lengths[idx] = episode_length

        if self.n_transitions_stored == self.buffer_size:
            self.full = True

    def _get_episode_dict(self, obs, next_obs, action, reward, done) -> dict:
        """
        Convert episode to dictionary.

        :param obs: (np.ndarray) Observation.
        :param next_obs: (np.ndarray) Next observation.
        :param action: (np.ndarray) Action.
        :param reward: (np.ndarray) Reward.
        :param done: (np.ndarray) Done.
        """

        observations = []
        achieved_goals = []
        desired_goals = []

        for obs_ in obs:
            observations.append(obs_["observation"])
            achieved_goals.append(obs_["achieved_goal"])
            desired_goals.append(obs_["desired_goal"])

        next_observations = []
        next_achieved_goals = []
        next_desired_goals = []

        for next_obs_ in next_obs:
            next_observations.append(next_obs_["observation"])
            next_achieved_goals.append(next_obs_["achieved_goal"])
            next_desired_goals.append(next_obs_["desired_goal"])

        episode = {
            "observation": np.array(observations),
            "achieved_goal": np.array(achieved_goals),
            "desired_goal": np.array(desired_goals),
            "action": action,
            "reward": reward,
            "next_obs": np.array(next_observations),
            "next_achieved_goal": np.array(next_achieved_goals),
            "next_desired_goal": np.array(next_desired_goals),
            "done": done,
        }

        return episode

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
