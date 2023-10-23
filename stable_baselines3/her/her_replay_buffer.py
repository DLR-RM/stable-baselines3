import copy
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy


class HerReplayBuffer(DictReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.

    .. note::

      Compared to other implementations, the ``future`` goal sampling strategy is inclusive:
      the current transition can be used when re-sampling.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment
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
    :param copy_info_dict: Whether to copy the info dictionary and pass it to
        ``compute_reward()`` method.
        Please note that the copy may cause a slowdown.
        False by default.
    """

    env: Optional[VecEnv]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        env: VecEnv,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        copy_info_dict: bool = False,
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
        self.copy_info_dict = copy_info_dict

        # convert goal_selection_strategy into GoalSelectionStrategy if string
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_selection_strategy = goal_selection_strategy

        # check if goal_selection_strategy is valid
        assert isinstance(
            self.goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"

        self.n_sampled_goal = n_sampled_goal

        # Compute ratio between HER replays and regular replays in percent
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # In some environments, the info dict is used to compute the reward. Then, we need to store it.
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])
        # To create virtual transitions, we need to know for each transition
        # when an episode starts and ends.
        # We use the following arrays to store the indices,
        # and update them when an episode ends.
        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.

        Excludes self.env, as in general Env's may not be pickleable.
        """
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["env"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.

        User must call ``set_env()`` after unpickling before using.

        :param state:
        """
        self.__dict__.update(state)
        assert "env" not in state
        self.env = None

    def set_env(self, env: VecEnv) -> None:
        """
        Sets the environment.

        :param env:
        """
        if self.env is not None:
            raise ValueError("Trying to set env of already initialized environment.")

        self.env = env

    def add(  # type: ignore[override]
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # When the buffer is full, we rewrite on old episodes. When we start to
        # rewrite on an old episodes, we want the whole old episode to be deleted
        # (and not only the transition on which we rewrite). To do this, we set
        # the length of the old episode to 0, so it can't be sampled anymore.
        for env_idx in range(self.n_envs):
            episode_start = self.ep_start[self.pos, env_idx]
            episode_length = self.ep_length[self.pos, env_idx]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = np.arange(self.pos, episode_end) % self.buffer_size
                self.ep_length[episode_indices, env_idx] = 0

        # Update episode start
        self.ep_start[self.pos] = self._current_ep_start.copy()

        if self.copy_info_dict:
            self.infos[self.pos] = infos
        # Store the transition
        super().add(obs, next_obs, action, reward, done, infos)

        # When episode ends, compute and store the episode length
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._compute_episode_length(env_idx)

    def _compute_episode_length(self, env_idx: int) -> None:
        """
        Compute and store the episode length for environment with index env_idx

        :param env_idx: index of the environment for which the episode length should be computed
        """
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.pos
        if episode_end < episode_start:
            # Occurs when the buffer becomes full, the storage resumes at the
            # beginning of the buffer. This can happen in the middle of an episode.
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        self.ep_length[episode_indices, env_idx] = episode_end - episode_start
        # Update the current episode start
        self._current_ep_start[env_idx] = self.pos

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:  # type: ignore[override]
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.
        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError(
                "Unable to sample before the end of the first episode. We recommend choosing a value "
                "for learning_starts that is greater than the maximum number of timesteps in the environment."
            )
        # Get the indices of valid transitions
        # Example:
        # if is_valid = [[True, False, False], [True, False, True]],
        # is_valid has shape (buffer_size=2, n_envs=3)
        # then valid_indices = [0, 3, 5]
        # they correspond to is_valid[0, 0], is_valid[1, 0] and is_valid[1, 2]
        # or in numpy format ([rows], [columns]): (array([0, 1, 1]), array([0, 0, 2]))
        # Those indices are obtained back using np.unravel_index(valid_indices, is_valid.shape)
        valid_indices = np.flatnonzero(is_valid)
        # Sample valid transitions that will constitute the minibatch of size batch_size
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        # Unravel the indexes, i.e. recover the batch and env indices.
        # Example: if sampled_indices = [0, 3, 5], then batch_indices = [0, 1, 1] and env_indices = [0, 0, 2]
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

        # Split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_indices, real_batch_indices = np.split(batch_indices, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # Get real and virtual data
        real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)
        # Create virtual transitions by sampling new desired goals and computing new rewards
        virtual_data = self._get_virtual_samples(virtual_batch_indices, virtual_env_indices, env)

        # Concatenate real and virtual data
        observations = {
            key: th.cat((real_data.observations[key], virtual_data.observations[key]))
            for key in virtual_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions))
        next_observations = {
            key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
            for key in virtual_data.next_observations.keys()
        }
        dones = th.cat((real_data.dones, virtual_data.dones))
        rewards = th.cat((real_data.rewards, virtual_data.rewards))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def _get_real_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, 1), env)),
        )

    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),  # type: ignore[attr-defined]
        )

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :return: Sampled goals
        """
        batch_ep_start = self.ep_start[batch_indices, env_indices]
        batch_ep_length = self.ep_length[batch_indices, env_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Replay with final state of current episode
            transition_indices_in_episode = batch_ep_length - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Replay with random state which comes from the same episode and was observed after current transition
            # Note: our implementation is inclusive: current transition can be sampled
            current_indices_in_episode = (batch_indices - batch_ep_start) % self.buffer_size
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length)

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Replay with random state which comes from the same episode as current transition
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.buffer_size
        return self.next_observations["achieved_goal"][transition_indices, env_indices]

    def truncate_last_trajectory(self) -> None:
        """
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        if (self._current_ep_start != self.pos).any():
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            # only consider epsiodes that are not finished
            for env_idx in np.where(self._current_ep_start != self.pos)[0]:
                # set done = True for last episodes
                self.dones[self.pos - 1, env_idx] = True
                # make sure that last episodes can be sampled and
                # update next episode start (self._current_ep_start)
                self._compute_episode_length(env_idx)
                # handle infinite horizon tasks
                if self.handle_timeout_termination:
                    self.timeouts[self.pos - 1, env_idx] = True  # not an actual timeout, but it allows bootstrapping
