from collections import deque
from typing import Dict, List, Optional, Tuple, Union

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
    Replay Buffer for sampling HER (Hindsight Experience Replay) transitions online.
    These transitions will not be saved in the Buffer.

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
        self.max_episode_stored = self.buffer_size // self.max_episode_length
        self.current_idx = 0

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
            key: np.empty((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

        self.goal_selection_strategy = goal_selection_strategy
        # percentage of her indices
        self.her_ratio = her_ratio

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
        online_sampling: bool = True,
        n_sampled_goal: int = None,
        replay_observations: np.ndarray = None,
    ) -> Union[ReplayBufferSamples, Tuple]:
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :param online_sampling: (bool) Using online_sampling for HER or not.
        :param n_sampled_goal: (int) Number of sampled goals for replay. (offline sampling)
        :param replay_observations: (np.ndarray) Observations of the offline replay buffer. Needed for 'RANDOM' goal strategy.
        :return: (ReplayBufferSamples or Tuple)
        """
        return self._sample_transitions(batch_size, env, online_sampling, n_sampled_goal, replay_observations)

    def vectorized_sample_goal(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
        online_sampling: bool = True,
        replay_observations: np.ndarray = None,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.

        :param episode_indices: (np.ndarray) Episode indices to use.
        :param her_indices: (np.ndarray) HER indices.
        :param transitions_indices: (np.ndarray) Transition indices to use.
        :param online_sampling: (bool) Using online_sampling for HER or not.
        :param replay_observations: (np.ndarray) Observations of the offline replay buffer. Needed for 'RANDOM' goal strategy.
        :return: (np.ndarray) Return sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transitions_indices = self.episode_lengths[her_episode_indices] - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            transitions_indices = np.random.randint(
                transitions_indices[her_indices] + 1, self.episode_lengths[her_episode_indices]
            )

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])

        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            if online_sampling:
                # replay with random state from the entire replay buffer
                her_episode_indices = np.random.randint(self.n_episodes_stored, size=len(her_indices))
                transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])
            else:
                # replay with random state from the entire replay buffer
                index = np.random.choice(np.arange(len(replay_observations)), len(her_indices))
                obs = replay_observations[index]
                # get only the observation part of the state
                obs_dim = self.env.obs_dim
                obs_array = obs[:, :, :obs_dim]
                return obs_array
        else:
            raise ValueError("Strategy for sampling goals not supported!")

        return self.buffer["achieved_goal"][her_episode_indices, transitions_indices]

    def _sample_transitions(
        self,
        batch_size: int,
        env: Optional[VecNormalize],
        online_sampling: bool = True,
        n_sampled_goal: int = None,
        replay_observations: np.ndarray = None,
    ) -> Union[ReplayBufferSamples, Tuple]:
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :param online_sampling: (bool) Using online_sampling for HER or not.
        :param n_sampled_goal: (int) Number of sampled goals for replay. (offline sampling)
        :param replay_observations: (np.ndarray) Observations of the offline replay buffer. Needed for 'RANDOM' goal strategy.
        :return: (ReplayBufferSamples or Tuple)
        """
        # Select which episodes to use
        if online_sampling:
            episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
            ep_length = self.episode_lengths[episode_indices]
        else:
            episode_length = self.episode_lengths[0]
            episode_indices = np.array(list(range(self.n_episodes_stored)) * episode_length * n_sampled_goal)
            her_indices = np.arange(len(episode_indices))
            # repeat every transition index n_sampled_goals times
            transitions_indices = np.array(list(range(episode_length)) * n_sampled_goal)

        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # restrict the sampling domain when ep_length > 1
            # otherwise filter out the indices
            if online_sampling:
                her_indices = her_indices[ep_length[her_indices] > 1]
                ep_length[her_indices] -= 1
            else:
                her_indices = her_indices[episode_length > 1 and transitions_indices < episode_length - 1]

        if online_sampling:
            # Select which transitions to use
            transitions_indices = np.random.randint(ep_length)
        # get selected transitions
        transitions = {key: self.buffer[key][episode_indices, transitions_indices].copy() for key in self.buffer.keys()}

        new_goals = self.vectorized_sample_goal(
            episode_indices, her_indices, transitions_indices, online_sampling, replay_observations
        )
        transitions["desired_goal"][her_indices] = new_goals

        # Convert to numpy array
        # TODO: disable if not needed for faster computation
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Vectorized computation
        transitions["reward"][her_indices] = self.env.env_method(
            "compute_reward",
            transitions["next_achieved_goal"][her_indices],
            transitions["desired_goal"][her_indices],
            transitions["info"][her_indices],
        )

        # concatenate observation with (desired) goal
        observations = ObsDictWrapper.convert_dict(transitions)
        next_observations = ObsDictWrapper.convert_dict(transitions, observation_key="next_obs")

        if online_sampling:
            data = (
                self._normalize_obs(observations, env),
                transitions["action"],
                self._normalize_obs(next_observations, env),
                transitions["done"],
                self._normalize_reward(transitions["reward"], env),
            )

            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        else:
            return observations, next_observations, transitions, her_indices

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[dict],
    ) -> None:

        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

        self.buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self.buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self.buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self.buffer["action"][self.pos][self.current_idx] = action
        self.buffer["done"][self.pos][self.current_idx] = done
        self.buffer["reward"][self.pos][self.current_idx] = reward
        self.buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        self.buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        self.buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]

        self.info_buffer[self.pos].append(infos)

        # update current pointer
        self.current_idx += 1

    def store_episode(self):
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx

        # update current episode pointer
        # Note: in the OpenAI implementation
        # when the buffer is full, the episode replaced
        # is randomly chosen
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0

    @property
    def n_episodes_stored(self):
        if self.full:
            return self.max_episode_stored
        return self.pos

    def clear_buffer(self):
        self.buffer = {}

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer in transitions.
        """
        return int(np.sum(self.episode_lengths))
