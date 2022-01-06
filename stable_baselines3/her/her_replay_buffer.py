import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy


def get_time_limit(env: VecEnv, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.

    :param env: Environment from which we want to get the time limit.
    :param current_max_episode_length: Current value for max_episode_length.
    :return: max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
            # Raise the error because the attribute is present but is None
            if current_max_episode_length is None:
                raise AttributeError
        # if not available check if a valid value was passed as an argument
        except AttributeError:
            raise ValueError(
                "The max episode length could not be inferred.\n"
                "You must specify a `max_episode_steps` when registering the environment,\n"
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the model constructor"
            )
    return current_max_episode_length


class HerReplayBuffer(DictReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.

    :param buffer_size: The size of the buffer measured in transitions.
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment. Used to computed hindsight reward.
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param online_sampling:
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: VecEnv,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))

        if isinstance(goal_selection_strategy, str):
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        # check if goal_selection_strategy is valid
        assert isinstance(
            goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"
        self.goal_selection_strategy = goal_selection_strategy
        self.online_sampling = online_sampling

        # Assigns a unique identifier to each episode.
        self.episode_uid = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_episode_uid = np.arange(self.n_envs)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.episode_uid[self.pos] = self._current_episode_uid.copy()
        super().add(obs, next_obs, action, reward, done, infos)

        # When done, start a new episode by assigning a new episode uid.
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._current_episode_uid[env_idx] = np.max(self._current_episode_uid) + 1

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        all_trans_coord = np.mgrid[0:upper_bound, 0 : self.n_envs].transpose(1, 2, 0)

        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            episode_uid = self.episode_uid[:upper_bound]
            is_last = np.zeros((upper_bound, self.n_envs)).astype(bool)
            # if a transition is the last one of the episode, then, the next transition has a different episode uid
            is_last[:-1] = episode_uid[:-1] != episode_uid[1:]
            # the last transition stored is always the last within its episode (truncate last trajectory)
            is_last[-1][:] = True
            # remove all last transitions
            all_trans_coord = all_trans_coord[np.logical_not(is_last)]

        # Uniform sampling on all transitions.
        sampled_indices = np.random.randint(all_trans_coord.shape[0], size=batch_size)
        trans_coord = all_trans_coord[sampled_indices]

        # Convenient aliases
        trans_indices, env_indices = trans_coord[:, 0], trans_coord[:, 1]

        obs = {key: obs[trans_indices, env_indices] for key, obs in self.observations.items()}
        next_obs = {key: obs[trans_indices, env_indices] for key, obs in self.next_observations.items()}

        # HER
        her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        new_goals = self.sample_goals(trans_coord[her_indices])
        obs["desired_goal"][her_indices] = new_goals
        rewards = self.rewards[trans_indices, env_indices]
        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            rewards[her_indices] = self.env.env_method(
                "compute_reward",
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                next_obs["achieved_goal"][her_indices],
                # here we use the new desired goal
                obs["desired_goal"][her_indices],
                # infos[her_indices],
                [{}, {}],
                indices=0,  # only call method for one env
            )[0]
            # env is vectorized, so it returns a list; we have to take the first (and unique) element of this list

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs(obs)
        next_obs_ = self._normalize_obs(next_obs)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        actions = self.to_torch(self.actions[trans_indices, env_indices])
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}
        # Only use dones that are not due to timeouts
        # deactivated by default (timeouts is initialized as an array of False)
        dones = self.to_torch(
            self.dones[trans_indices, env_indices] * (1 - self.timeouts[trans_indices, env_indices])
        ).reshape(-1, 1)
        rewards = self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def sample_goals(self, trans_coord: np.ndarray) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.

        :param trans_coord: Coordinates of the transistions within the buffer
        :return: Return sampled goals.
        """
        goals_coord = np.zeros_like(trans_coord)
        episode_uids = self.episode_uid[trans_coord[:, 0], trans_coord[:, 1]]
        episodes = [self.get_episode_from_uid(episode_uid) for episode_uid in episode_uids]
        for i, (trans_idx, episode) in enumerate(zip(trans_coord, episodes)):
            if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
                # replay with final state of current episode
                goals_coord[i] = episode[-1]

            elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
                # replay with random state which comes from the same episode and was observed after current transition
                trans_idx_within_ep = np.where((episode == trans_idx).all(1))[0][0]
                sampled_idx_within_ep = np.random.randint(trans_idx_within_ep, episode.shape[0])
                goals_coord[i] = episode[sampled_idx_within_ep]

            elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
                # replay with random state which comes from the same episode as current transition
                goals_coord[i] = np.random.choice(episode)

            else:
                raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return self.observations["achieved_goal"][goals_coord[:, 0], goals_coord[:, 1]]

    def get_episode_from_uid(self, uid):
        upper_bound = self.buffer_size if self.full else self.pos
        all_trans_indices = np.mgrid[0:upper_bound, 0 : self.n_envs].transpose(1, 2, 0)
        episode = all_trans_indices[self.episode_uid[:upper_bound] == uid]
        return episode
