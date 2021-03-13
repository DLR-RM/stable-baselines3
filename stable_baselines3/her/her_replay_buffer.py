import io
import pathlib
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import DictReplayBuffer
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


"""
Hindsight Experience Replay (HER)
Paper: https://arxiv.org/abs/1707.01495

.. warning::

  For performance reasons, the maximum number of steps per episodes must be specified.
  In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
  or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
  Otherwise, you can directly pass ``max_episode_length`` to the model constructor


For additional offline algorithm specific arguments please have a look at the corresponding documentation.

:param policy: The policy model to use.
:param env: The environment to learn from (if registered in Gym, can be str)
:param model_class: Off policy model which will be used with hindsight experience replay. (SAC, TD3, DDPG, DQN)
:param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
:param goal_selection_strategy: Strategy for sampling goals for replay.
    One of ['episode', 'final', 'future', 'random']
:param online_sampling: Sample HER transitions online.
:param learning_rate: learning rate for the optimizer,
    it can be a function of the current progress remaining (from 1 to 0)
:param max_episode_length: The maximum length of an episode. If not specified,
    it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
"""


class HerReplayBuffer(DictReplayBuffer):
    """
    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.

    :param env: The training environment
    :param buffer_size: The size of the buffer measured in transitions.
    :param max_episode_length: The length of an episode. (time horizon)
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :her_ratio: The ratio between HER transitions and regular transitions in percent
        (between 0 and 1, for online sampling)
        The default value ``her_ratio=0.8`` corresponds to 4 virtual transitions
        for one real transition (4 / (4 + 1) = 0.8)
    """

    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        # observation_space: spaces.Space,
        # action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        replay_buffer: Optional[DictReplayBuffer] = None,
        max_episode_length: Optional[int] = None,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
    ):

        super(HerReplayBuffer, self).__init__(buffer_size, env.observation_space, env.action_space, device, env.num_envs)

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
        # if we sample her transitions online use custom replay buffer
        self.online_sampling = online_sampling
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # maximum steps in episode
        self.max_episode_length = get_time_limit(env, max_episode_length)
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = buffer_size if online_sampling else self.max_episode_length

        self.env = env
        self.buffer_size = her_buffer_size
        self.max_episode_length = max_episode_length

        if online_sampling:
            replay_buffer = None
        self.replay_buffer = replay_buffer
        self.online_sampling = online_sampling

        # TODO: this should not be needed anymore
        # if self.get_vec_normalize_env() is not None:
        #     assert online_sampling, "You must pass `online_sampling=True` if you want to use `VecNormalize` with `HER`"

        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = self.buffer_size // self.max_episode_length
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0

        # get dimensions of observation and goal
        if isinstance(self.env.observation_space.spaces["observation"], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            self.obs_dim = self.env.observation_space.spaces["observation"].shape[0]
            self.goal_dim = self.env.observation_space.spaces["achieved_goal"].shape[0]

        # input dimensions for buffer initialization
        input_shape = {
            "observation": (self.env.num_envs, self.obs_dim),
            "achieved_goal": (self.env.num_envs, self.goal_dim),
            "desired_goal": (self.env.num_envs, self.goal_dim),
            "action": (self.action_dim,),
            "reward": (1,),
            "next_obs": (self.env.num_envs, self.obs_dim),
            "next_achieved_goal": (self.env.num_envs, self.goal_dim),
            "next_desired_goal": (self.env.num_envs, self.goal_dim),
            "done": (1,),
        }
        self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        self.buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        # Store info dicts are it can be used to compute the reward (e.g. continuity cost)
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

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

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Abstract method from base class.
        """
        raise NotImplementedError()

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize],
    ) -> DictReplayBufferSamples:  # pytype: disable=bad-return-type
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.

        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        if self.replay_buffer is not None:
            return self.replay_buffer.sample(batch_size, env)
        return self._sample_transitions(batch_size, maybe_vec_env=env, online_sampling=True)

    def sample_offline(
        self,
        n_sampled_goal: Optional[int] = None,
    ) -> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], ...]]:
        """
        Sample function for offline sampling of HER transition,
        in that case, only one episode is used and transitions
        are added to the regular replay buffer.

        :param n_sampled_goal: Number of sampled goals for replay
        :return: at most(n_sampled_goal * episode_length) HER transitions.
        """
        # env=None as we should store unnormalized transitions, they will be normalized at sampling time
        return self._sample_transitions(
            batch_size=None,
            maybe_vec_env=None,
            online_sampling=False,
            n_sampled_goal=n_sampled_goal,
        )

    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.

        :param episode_indices: Episode indices to use.
        :param her_indices: HER indices.
        :param transitions_indices: Transition indices to use.
        :return: Return sampled goals.
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

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return self.buffer["achieved_goal"][her_episode_indices, transitions_indices]

    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
        online_sampling: bool,
        n_sampled_goal: Optional[int] = None,
    ) -> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], ...]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        if online_sampling:
            assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            # A subset of the transitions will be relabeled using HER algorithm
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
            assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
            # Offline sampling: there is only one episode stored
            episode_length = self.episode_lengths[0]
            # we sample n_sampled_goal per timestep in the episode (only one is stored).
            episode_indices = np.tile(0, (episode_length * n_sampled_goal))
            # we only sample virtual transitions
            # as real transitions are already stored in the replay buffer
            her_indices = np.arange(len(episode_indices))

        ep_lengths = self.episode_lengths[episode_indices]

        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # restrict the sampling domain when ep_lengths > 1
            # otherwise filter out the indices
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        if online_sampling:
            # Select which transitions to use
            transitions_indices = np.random.randint(ep_lengths)
        else:
            if her_indices.size == 0:
                # Episode of one timestep, not enough for using the "future" strategy
                # no virtual transitions are created in that case
                return {}, {}, np.zeros(0), np.zeros(0)
            else:
                # Repeat every transition index n_sampled_goals times
                # to sample n_sampled_goal per timestep in the episode (only one is stored).
                # Now with the corrected episode length when using "future" strategy
                transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
                episode_indices = episode_indices[transitions_indices]
                her_indices = np.arange(len(episode_indices))

        # get selected transitions
        transitions = {key: self.buffer[key][episode_indices, transitions_indices].copy() for key in self.buffer.keys()}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Vectorized computation of the new reward
        transitions["reward"][her_indices, 0] = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next_achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use "next_achieved_goal" and not "achieved_goal"
            transitions["next_achieved_goal"][her_indices, 0],
            # here we use the new desired goal
            transitions["desired_goal"][her_indices, 0],
            transitions["info"][her_indices, 0],
        )

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            "desired_goal": transitions["next_desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        if online_sampling:
            next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

            normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

            return DictReplayBufferSamples(
                observations=normalized_obs,
                actions=self.to_torch(transitions["action"]),
                next_observations=next_obs,
                dones=self.to_torch(transitions["done"]),
                rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
            )
        else:
            return observations, next_observations, transitions["action"], transitions["reward"]

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
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

        # When doing offline sampling
        # Add real transition to normal replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(
                obs,
                next_obs,
                action,
                reward,
                done,
                infos,
            )

        self.info_buffer[self.pos].append(infos)

        # update current pointer
        self.current_idx += 1

        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode()
            if not self.online_sampling:
                # sample virtual transitions and store them in replay buffer
                self._sample_her_transitions()
                # clear storage for current episode
                self.reset()

            self.episode_steps = 0

    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
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

    def _sample_her_transitions(self) -> None:
        """
        Sample additional goals and store new transitions in replay buffer
        when using offline sampling.
        """

        # Sample goals and get new observations
        # maybe_vec_env=None as we should store unnormalized transitions,
        # they will be normalized at sampling time
        observations, next_observations, actions, rewards = self.sample_offline(n_sampled_goal=self.n_sampled_goal)

        # Store virtual transitions in the replay buffer, if available
        if len(observations) > 0:
            for i in range(len(observations["observation"])):
                self.replay_buffer.add(
                    {key: obs[i] for key, obs in observations.items()},
                    {key: next_obs[i] for key, next_obs in next_observations.items()},
                    actions[i],
                    rewards[i],
                    done=[False],
                )

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    def load_replay_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase], truncate_last_trajectory: bool = True
    ) -> None:
        """
        Load a replay buffer from a pickle file and set environment for replay buffer (only online sampling).

        :param path: Path to the pickled replay buffer.
        :param truncate_last_trajectory: Only for online sampling.
            If set to ``True`` we assume that the last trajectory in the replay buffer was finished.
            If it is set to ``False`` we assume that we continue the same trajectory (same episode).
        """
        if self.online_sampling:
            # set environment
            self.set_env(self.env)
            # If we are at the start of an episode, no need to truncate
            current_idx = self.current_idx

            # truncate interrupted episode
            if truncate_last_trajectory and current_idx > 0:
                warnings.warn(
                    "The last trajectory in the replay buffer will be truncated.\n"
                    "If you are in the same episode as when the replay buffer was saved,\n"
                    "you should use `truncate_last_trajectory=False` to avoid that issue."
                )
                # get current episode and transition index
                pos = self.pos
                # set episode length for current episode
                self.episode_lengths[pos] = current_idx
                # set done = True for current episode
                # current_idx was already incremented
                self.buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
                # reset current transition index
                self.current_idx = 0
                # increment episode counter
                self.pos = (self.pos + 1) % self.max_episode_stored
                # update "full" indicator
                self.full = self.full or self.pos == 0
