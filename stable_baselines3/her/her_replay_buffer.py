import copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy


class HerReplayBuffer(DictReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    .. warning::
      For backward compatibility, we implement offline sampling. The offline
      sampling mode only works for `n_envs == 1`.

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.

    :param buffer_size: The size of the buffer measured in transitions.
    :param observation_space: Observation space
    :param action_space: Action space
    :param compute_reward: Function used to compute new rewards
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
    :param online_sampling: If False, virtual transitions are saved in the replay buffer.
        Only works for `n_envs == 1`.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        compute_reward: Callable,
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
        self.compute_reward = compute_reward
        self.n_sampled_goal = n_sampled_goal
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])

        if isinstance(goal_selection_strategy, str):
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        # check if goal_selection_strategy is valid
        assert isinstance(
            goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"
        self.goal_selection_strategy = goal_selection_strategy
        self.online_sampling = online_sampling
        if not self.online_sampling:
            assert n_envs == 1, "Offline sampling is not compatible with multiprocessing."

        # Assigns a unique identifier to each episode.
        self.episode_uids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64) - 1
        self._current_episode_uid = np.arange(self.n_envs)
        self.is_episode_valid = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        is_virtual: bool = False,
    ) -> None:
        if self.full:
            # When the buffer is full, we rewrite on old episodes. When we start to rewrite on an old
            # episodes, we want the whole old episode to be deleted (and not only the transition on
            # which we rewrite). To do this we use a validity mask. When we start to rewrite on an
            # episode, all the transitions of the episode become invalid.
            old_episode_uid = self.episode_uids[self.pos]
            self.is_episode_valid[self.episode_uids == old_episode_uid] = False
        self.episode_uids[self.pos] = self._current_episode_uid.copy()

        # Remove termination signals due to timeout
        if self.handle_timeout_termination:
            timeout = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            done_ = done * (1 - timeout)
        else:
            done_ = done
        # Store the transition
        self.infos[self.pos] = infos
        super().add(obs, next_obs, action, reward, done_, infos)

        # When done, validate the episode and start a new episode by assigning a new episode uid.
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                # Validate the episode
                self.is_episode_valid[self.episode_uids == self._current_episode_uid] = True
                # Increment epsiode uid
                last_episode_uid = self._current_episode_uid[env_idx]
                self._current_episode_uid[env_idx] = np.max(self._current_episode_uid) + 1
                # When offline sampling, samples virtual transitions
                if not self.online_sampling and not is_virtual:
                    self._sample_offline(last_episode_uid)

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

        valid = self.is_episode_valid[:upper_bound]
        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            episode_uids = self.episode_uids[:upper_bound]
            # if a transition is the last one of the episode, then, the next transition has a different episode uid
            is_last = episode_uids != np.roll(episode_uids, shift=-1, axis=0)
            # remove all last transitions
            valid = np.logical_and(valid, np.logical_not(is_last))

        all_trans_coord = all_trans_coord[valid]

        # Uniform sampling on all transitions.
        sampled_indices = np.random.randint(all_trans_coord.shape[0], size=batch_size)
        trans_coord = all_trans_coord[sampled_indices]

        if self.online_sampling:
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            her_indices = np.array([], dtype=int)
        obs, next_obs, actions, rewards, dones, infos = self._create_virual_trans(trans_coord, her_indices)

        # Normalize if needed and remove extra dimension
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)
        rewards = self._normalize_reward(rewards.reshape(-1, 1), env)
        dones = dones.reshape(-1, 1)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs_) for key, obs_ in obs.items()}
        next_observations = {key: self.to_torch(obs_) for key, obs_ in next_obs.items()}
        actions = self.to_torch(actions)
        dones = self.to_torch(dones)
        rewards = self.to_torch(rewards)

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
        episodes = [self._get_episode_from_uid(episode_uid) for episode_uid in episode_uids]
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
                sampled_idx = np.random.randint(0, episode.shape[0])
                goals_coord[i] = episode[sampled_idx]

            else:
                raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return self.observations["achieved_goal"][goals_coord[:, 0], goals_coord[:, 1]]

    def _get_episode_from_uid(self, uid: int) -> np.ndarray:
        """
        Returns the transitions coordinates of and episode designated by its unique identifier.

        :param uid: The unique identifier of the episode
        :return: Coordinates of the transition of the episode
        """
        upper_bound = self.buffer_size if self.full else self.pos
        all_trans_indices = np.mgrid[0:upper_bound, 0 : self.n_envs].transpose(1, 2, 0)
        is_episode_valid = self.is_episode_valid[:upper_bound]
        episode_uids = self.episode_uids[:upper_bound]
        all_trans_indices = all_trans_indices[is_episode_valid]
        episode_uids = episode_uids[is_episode_valid]
        episode = all_trans_indices[episode_uids == uid]
        # When the buffer is full, an episode is stored in such a way that
        # the beginning of the episode is at the end of the buffer and the
        # end of the episode is at the beginning of the buffer. When this
        # episode is sampled, the transitions must be put back in order.
        gap = episode[1:, 0] - episode[:-1, 0]
        split = np.asarray(gap != 1).nonzero()[0]
        if split.shape[0] > 0:  # if there is a split
            episode = np.roll(episode, shift=-(split[0] + 1), axis=0)
        return episode

    def _sample_offline(self, episode_uid: int) -> None:
        episode = self._get_episode_from_uid(episode_uid)
        trans_coord = np.tile(episode, (self.n_sampled_goal, 1))

        # Here, all transions are virtual
        her_indices = np.arange(trans_coord.shape[0])
        obs, next_obs, actions, rewards, dones, infos = self._create_virual_trans(trans_coord, her_indices)

        for i in range(actions.shape[0]):
            self.add(
                {key: value[i] for key, value in obs.items()},
                {key: value[i] for key, value in next_obs.items()},
                actions[i],
                rewards[i],
                [dones[i]],
                [infos[i]],
                is_virtual=True,
            )

    def _create_virual_trans(
        self, trans_coord: np.ndarray, her_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Convenient aliases
        trans_indices, env_indices = trans_coord[:, 0], trans_coord[:, 1]

        obs = {key: obs[trans_indices, env_indices] for key, obs in self.observations.items()}
        next_obs = {key: obs[trans_indices, env_indices] for key, obs in self.next_observations.items()}
        actions = self.actions[trans_indices, env_indices]
        rewards = self.rewards[trans_indices, env_indices]
        dones = self.dones[trans_indices, env_indices]
        if self.handle_timeout_termination:
            timeout = self.timeouts[trans_indices, env_indices]
            dones = dones + timeout  # logical OR
        infos = copy.deepcopy(self.infos[trans_indices, env_indices])

        if her_indices.shape[0] > 0:
            # Sample and set new goals
            new_goals = self.sample_goals(trans_coord[her_indices])
            obs["desired_goal"][her_indices] = new_goals
            # The desired goal for the next observation must be the same as the previous one
            next_obs["desired_goal"][her_indices] = new_goals
            # The goal has changed, there is no longer a guarantee that the transition is
            # successful. Since it is not possible to easily get this information, we prefer
            # to remove it. The success information is not used in the learning algorithm anyway.
            for info in infos[her_indices]:
                info.pop("is_success", None)

            rewards[her_indices] = self.compute_reward(
                # here we use the new desired goal
                obs["desired_goal"][her_indices],
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use next_obs["achived_goal"] and not obs["achived_goal"]
                next_obs["achieved_goal"][her_indices],
                infos[her_indices],
            )
        return obs, next_obs, actions, rewards, dones, infos

    def truncate_last_trajectory(self) -> None:
        """
        Only for online sampling, called when loading the replay buffer.
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        last_episode_uids = np.max(self.episode_uids, axis=0)
        current_episode_uids = self._current_episode_uid

        # truncate interrupted episode
        for env_idx in range(self.n_envs):
            if last_episode_uids[env_idx] == current_episode_uids[env_idx]:
                warnings.warn(
                    "The last trajectory in the replay buffer will be truncated.\n"
                    "If you are in the same episode as when the replay buffer was saved,\n"
                    "you should use `truncate_last_trajectory=False` to avoid that issue."
                )
                self._current_episode_uid[env_idx] = np.max(self._current_episode_uid) + 1
                self.dones[self.pos - 1][env_idx] = True
