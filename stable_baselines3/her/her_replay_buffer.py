import copy
import warnings
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

    .. warning::
      For backward compatibility, we implement offline sampling. The offline
      sampling mode only works for `n_envs == 1`.

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.

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
    :param online_sampling: If False, virtual transitions are saved in the replay buffer.
        Only works for `n_envs == 1`.
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

        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.child_buffer: ReplayBuffer = None  # when specifed, also store obs in this buffer

    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.
        Excludes self.env, as in general Env's may not be pickleable.
        Note: when using offline sampling, this will also save the offline replay buffer.
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
        # When the buffer is full, we rewrite on old episodes. When we start to
        # rewrite on an old episodes, we want the whole old episode to be deleted
        # (and not only the transition on which we rewrite). To do this, we set
        # the length of the old episode to 0, so it can't be sampled anymore.
        for env_idx in range(self.n_envs):
            episode_start = self.ep_start[self.pos][env_idx]
            episode_length = self.ep_length[self.pos][env_idx]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = np.arange(self.pos, episode_end) % self.buffer_size
                self.ep_length[episode_indices, env_idx] = 0

        # Update episode start
        self.ep_start[self.pos] = self._current_ep_start.copy()

        # Store the transition
        self.infos[self.pos] = infos
        super().add(obs, next_obs, action, reward, done, infos)

        # When episode ends, compute and store the episode length
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                episode_start = self._current_ep_start[env_idx]
                episode_end = self.pos
                if episode_end < episode_start:
                    # Occurs when the buffer becomes full, the storage resumes at the
                    # beginning of the buffer. This can happen in the middle of an episode.
                    episode_end += self.buffer_size
                episode = np.arange(episode_start, episode_end) % self.buffer_size
                self.ep_length[episode, env_idx] = episode_end - episode_start
                # Update the current episode start
                self._current_ep_start[env_idx] = self.pos
                if not self.online_sampling and not is_virtual:
                    self._sample_offline(env_idx)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        if self.online_sampling:
            return self._sample_online(batch_size, env)
        else:
            # virtual trnasition has already been saved
            return super().sample(batch_size, env)

    def _sample_online(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples
        """
        env_indices = np.random.randint(self.n_envs, size=batch_size)
        batch_inds = np.zeros_like(env_indices)
        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.
        all_inds = np.tile(np.arange(self.buffer_size), (self.n_envs, 1)).T
        is_valid = self.ep_length > 0
        # Special case when using the "future" goal sampling strategy, we cannot
        # sample all transitions, we restrict the sampling domain to non-final transitions
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            is_last = all_inds == (self.ep_start + self.ep_length - 1) % self.buffer_size
            is_valid = np.logical_and(np.logical_not(is_last), is_valid)

        valid_inds = [np.arange(self.buffer_size)[is_valid[:, env_idx]] for env_idx in range(self.n_envs)]
        for i, env_idx in enumerate(env_indices):
            batch_inds[i] = np.random.choice(valid_inds[env_idx])

        # Split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_inds, real_batch_inds = np.split(batch_inds, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # get real and virtual data
        real_data = self._get_real_samples(real_batch_inds, real_env_indices, env)
        virtual_data = self._get_virtual_samples(virtual_batch_inds, virtual_env_indices, env)

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

    def _sample_offline(self, env_idx: int) -> None:
        """
        Samples virtual transitions from the last episode, and stores them in the buffer.

        :param env_idx: Environment index
        """
        pos = self.pos - 1  # pos has already been incremented
        ep_length = self.ep_length[pos, env_idx]
        episode_start = self.ep_start[pos, env_idx]
        episode_end = episode_start + ep_length
        if episode_end < episode_start:
            # Occurs when the buffer becomes full, the storage resumes at the
            # beginning of the buffer. This can happen in the middle of an episode.
            episode_end += self.buffer_size
        batch_inds = np.tile(np.arange(episode_start, episode_end) % self.buffer_size, self.n_sampled_goal)
        env_indices = np.repeat(env_idx, self.n_sampled_goal * ep_length)

        # Special case when using the "future" goal sampling strategy, we cannot
        # sample all transitions, we restrict the sampling domain to non-final transitions
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            is_last = batch_inds == (episode_start + ep_length - 1) % self.buffer_size
            batch_inds = batch_inds[np.logical_not(is_last)]
            env_indices = env_indices[np.logical_not(is_last)]
            ep_length -= 1

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if batch_inds.shape[0] > 0:
            # All transitions are virtual
            data = self._get_virtual_samples(batch_inds, env_indices)
            # _get_virtual_samples returns done that are not due to timeout. Moreover,
            # the last transition may have been deleted if the goal selection strategy
            # is "future". Therefore, we impose here done=True for the last transition.
            is_last = batch_inds == (episode_start + ep_length - 1) % self.buffer_size
            dones = is_last.astype(np.float32)
            infos = self.infos[batch_inds, env_indices]

            for i in range(batch_inds.shape[0]):
                self.add(
                    {key: value[i].cpu().numpy() for key, value in data.observations.items()},
                    {key: value[i].cpu().numpy() for key, value in data.next_observations.items()},
                    data.actions[i].cpu().numpy(),
                    data.rewards[i].cpu().numpy(),
                    [dones[i]],
                    [infos[i]],
                    is_virtual=True,
                )

    def _get_real_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_inds: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()})

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )

    def _get_virtual_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_inds: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}
        infos = copy.deepcopy(self.infos[batch_inds, env_indices])
        # Sample and set new goals
        new_goals = self._sample_goals(batch_inds, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals
        # The goal has changed, there is no longer a guarantee that the transition is
        # successful. Since it is not possible to easily get this information, we prefer
        # to remove it. The success information is not used in the learning algorithm anyway.
        for info in infos:
            info.pop("is_success", None)
        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            # here we use the new desired goal
            obs["desired_goal"],
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achived_goal"] and not obs["achived_goal"]
            next_obs["achieved_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        obs = self._normalize_obs(obs)
        next_obs = self._normalize_obs(next_obs)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )

    def _sample_goals(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.

        :param trans_coord: Coordinates of the transistions within the buffer
        :return: Return sampled goals
        """
        batch_ep_start = self.ep_start[batch_inds, env_indices]
        batch_ep_length = self.ep_length[batch_inds, env_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transition_indices_in_episode = batch_ep_length - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            current_indices_in_episode = batch_inds - batch_ep_start
            transition_indices_in_episode = np.random.randint(current_indices_in_episode + 1, batch_ep_length)

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.buffer_size
        return self.observations["achieved_goal"][transition_indices, env_indices]

    def truncate_last_trajectory(self) -> None:
        """
        Only for online sampling, called when loading the replay buffer.
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        if (self.ep_start[self.pos] != self.pos).any():
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            self.ep_start[-1] = self.pos
            # set done = True for current episodes
            self.dones[self.pos - 1] = True
