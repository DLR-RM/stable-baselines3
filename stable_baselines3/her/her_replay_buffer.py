from typing import Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
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
        env: ObsDictWrapper,
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
        self.n_episodes = n_episodes

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
            key: np.empty((n_episodes, self.max_episode_length, *dim), dtype=np.float32) for key, dim in input_shape.items()
        }
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.empty(n_episodes, dtype=np.uint64)

        self.goal_selection_strategy = goal_selection_strategy
        # percentage of her indices
        self.her_ratio = her_ratio

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
        episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
        her_episode_indices = episode_indices[: int(self.her_ratio * batch_size)]

        observations = np.zeros((batch_size, self.env.obs_dim + self.env.goal_dim), dtype=self.observation_space.dtype)
        actions = np.zeros((batch_size, self.action_dim), dtype=self.action_space.dtype)
        next_observations = np.zeros((batch_size, self.env.obs_dim + self.env.goal_dim), dtype=self.observation_space.dtype)
        dones = np.zeros((batch_size, 1), dtype=np.float32)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)

        for idx, ep_length in enumerate(self.episode_lengths[episode_indices]):
            skip_her_sampling = False
            if episode_indices[idx] in her_episode_indices and self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
                max_timestep = ep_length - 1
                # handle the case of 1 step episode: we must use a normal transition then
                if max_timestep == 0:
                    max_timestep = ep_length
                    skip_her_sampling = True
            else:
                max_timestep = ep_length

            transition_idx = np.random.randint(max_timestep)
            transition = {key: self.buffer[key][episode_indices[idx], transition_idx].copy() for key in self.buffer.keys()}

            if episode_indices[idx] in her_episode_indices and not skip_her_sampling:
                episode = self.buffer["achieved_goal"][episode_indices[idx]]
                new_goal = self.sample_goal(
                    self.goal_selection_strategy, transition_idx, episode, self.buffer["achieved_goal"], online_sampling=True
                )
                # observation
                transition["desired_goal"] = new_goal
                # next observation
                transition["next_desired_goal"] = new_goal
                transition["reward"] = self.env.env_method("compute_reward", transition["next_achieved_goal"], new_goal, None)
                # TODO: check that it does not change anything
                # transition["done"] = False

            # concatenate observation with (desired) goal
            obs = ObsDictWrapper.convert_dict(transition)
            next_obs = ObsDictWrapper.convert_dict(transition, observation_key="next_obs")
            observations[idx] = obs
            next_observations[idx] = next_obs
            actions[idx] = transition["action"]
            dones[idx] = transition["done"]
            rewards[idx] = transition["reward"]

        data = (
            self._normalize_obs(observations, env),
            actions,
            self._normalize_obs(next_observations, env),
            dones,
            self._normalize_reward(rewards, env),
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

        for key in self.buffer.keys():
            self.buffer[key][self.pos][:episode_length] = episode[key]
        # add episode length to length storage
        self.episode_lengths[self.pos] = episode_length

        # update current pointer
        self.pos += 1
        if self.pos == self.n_episodes:
            self.full = True
            self.pos = 0

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
        if self.full:
            return self.n_episodes
        return self.pos

    def clear_buffer(self):
        self.buffer = {}

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer in transitions.
        """
        return int(np.sum(self.episode_lengths))
