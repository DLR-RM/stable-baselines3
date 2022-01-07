from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy


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
        self.episode_uids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_episode_uid = np.arange(self.n_envs)
        # When the buffer is full, we rewrite on old episodes. When we start to rewrite on an old
        # episodes, we want the whole old episode to be deleted (and not only the transition on
        # which we rewrite). To do this we use a validity mask. When we start to rewrite on an
        # episode, all the transitions of the episode become invalid.
        self.is_episode_valid = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if self.full:
            old_episode_uid = self.episode_uids[self.pos]
            self.is_episode_valid[self.episode_uids == old_episode_uid] = False
        self.episode_uids[self.pos] = self._current_episode_uid.copy()
        self.is_episode_valid[self.pos] = True
        super().add(obs, next_obs, action, reward, done, infos)

        # When done, start a new episode by assigning a new episode uid.
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                if not self.online_sampling:
                    self.sample_offline(env_idx, self._current_episode_uid[env_idx])
                self._current_episode_uid[env_idx] = np.max(self._current_episode_uid) + 1

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        all_trans_coord = np.mgrid[0 : self.buffer_size, 0 : self.n_envs].transpose(1, 2, 0)

        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            episode_uid = self.episode_uids
            is_last = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
            # if a transition is the last one of the episode, then, the next transition has a different episode uid
            is_last[:-1] = episode_uid[:-1] != episode_uid[1:]
            # the last transition stored is always the last within its episode (truncate last trajectory)
            is_last[self.pos - 1] = True
            # remove all last transitions and invalid
            all_trans_coord = all_trans_coord[np.logical_and(np.logical_not(is_last), self.is_episode_valid)]
        else:
            all_trans_coord = all_trans_coord[self.is_episode_valid]

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
        episode_uids = self.episode_uids[trans_coord[:, 0], trans_coord[:, 1]]
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
        # upper_bound = self.buffer_size if self.full else self.pos
        all_trans_indices = np.mgrid[0 : self.buffer_size, 0 : self.n_envs].transpose(1, 2, 0)
        all_trans_indices = all_trans_indices[self.is_episode_valid]
        episode_uids = self.episode_uids[self.is_episode_valid]
        episode = all_trans_indices[episode_uids == uid]
        # When the buffer is full, an episode is stored in such a way that
        # the beginning of the episode is at the end of the buffer and the
        # end of the episode is at the beginning of the buffer. When this
        # episode is sampled, the transitions must be put back in order.
        gap = episode[1:, 0] - episode[:-1, 0]
        split = np.where(gap != 1)[0]
        if split:
            episode = np.roll(episode, shift=-(split[0] + 1), axis=0)
        return episode

    def sample_offline(self, env_idx, episode_uid, env: Optional[VecNormalize] = None) -> None:
        """
        Sample offline from the replay buffer.

        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        raise NotImplementedError("Offline sampling not implemented yet.")

        # Convenient aliases
        ep_lenght = np.sum(self.episode_uids[self.is_episode_valid] == episode_uid)
        trans_indices = np.tile(np.arange(ep_lenght), self.n_sampled_goal)
        env_indices = np.tile(env_idx, self.n_sampled_goal * ep_lenght)
        trans_coord = np.stack((trans_indices, env_indices), axis=1)

        obs = {key: obs[trans_indices, env_indices] for key, obs in self.observations.items()}
        next_obs = {key: obs[trans_indices, env_indices] for key, obs in self.next_observations.items()}

        # HER
        new_goals = self.sample_goals(trans_coord)
        obs["desired_goal"] = new_goals
        # Vectorized computation of the new reward
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next_achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use "next_achieved_goal" and not "achieved_goal"
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
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
