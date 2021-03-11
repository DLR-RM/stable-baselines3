import io
import pathlib
import types
import warnings
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, TrainFreq
from stable_baselines3.common.utils import check_for_correct_spaces, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


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


def _store_transition(
    self,
    replay_buffer: ReplayBuffer,
    buffer_action: np.ndarray,
    new_obs: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    infos: List[Dict[str, Any]],
    her_model: "HER",
) -> None:
    """
    Store transition in the replay buffer(s) when using HER.
    This function is a hack that will replace the default ``_store_transition``
    of the off-policy algorithm.

    We store the normalized action and the unnormalized observation.
    It also handles terminal observations (because VecEnv resets automatically).

    :param replay_buffer: Replay buffer object where to store the transition.
    :param buffer_action: normalized action
    :param new_obs: next observation in the current episode
        or first observation of the episode (when done is True)
    :param reward: reward for the current transition
    :param done: Termination signal
    :param infos: List of additional information about the transition.
        It contains the terminal observations.
    :param her_model: The ``HER`` object.
    """
    # Store data in replay buffer (normalized action and unnormalized observation)
    if her_model.online_sampling:
        replay_buffers = [her_model._episode_storage]
    else:
        # Add normal transition to replay buffer
        # and to episode storage in order to sample virtual transitions
        # at the end of an episode
        replay_buffers = [her_model.model.replay_buffer, her_model._episode_storage]

    # Call parent method (from off-policy algorithm)
    super(self.__class__, self)._store_transition(
        replay_buffers,
        buffer_action,
        new_obs,
        reward,
        done,
        infos,
    )
    her_model.episode_steps += 1

    if done or her_model.episode_steps >= her_model.max_episode_length:
        if her_model.online_sampling:
            her_model.replay_buffer.store_episode()
        else:
            her_model._episode_storage.store_episode()
            # sample virtual transitions and store them in replay buffer
            her_model._sample_her_transitions()
            # clear storage for current episode
            her_model._episode_storage.reset()

        her_model.episode_steps = 0


# TODO: rewrite HER class as soon as dict obs are supported
class HER(BaseAlgorithm):
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

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        model_class: Type[OffPolicyAlgorithm],
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = False,
        max_episode_length: Optional[int] = None,
        *args,
        **kwargs,
    ):

        # we will use the policy and learning rate from the model
        super(HER, self).__init__(policy=BasePolicy, env=env, policy_base=BasePolicy, learning_rate=0.0)
        del self.policy, self.learning_rate

        if self.get_vec_normalize_env() is not None:
            assert online_sampling, "You must pass `online_sampling=True` if you want to use `VecNormalize` with `HER`"

        _init_setup_model = kwargs.get("_init_setup_model", True)
        if "_init_setup_model" in kwargs:
            del kwargs["_init_setup_model"]
        # model initialization
        self.model_class = model_class
        self.model = model_class(
            policy=policy,
            env=self.env,
            _init_setup_model=False,  # pytype: disable=wrong-keyword-args
            *args,
            **kwargs,  # pytype: disable=wrong-keyword-args
        )

        # Make HER use self.model.action_noise
        del self.action_noise
        self.verbose = self.model.verbose
        self.tensorboard_log = self.model.tensorboard_log

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
        self.max_episode_length = get_time_limit(self.env, max_episode_length)
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = self.buffer_size if online_sampling else self.max_episode_length

        # TODO: check if that can be done during `_setup_learn()`
        assert self.env is not None, "Because it needs access to `env.compute_reward()` HER you must provide the env."

        self._episode_storage = HerReplayBuffer(
            self.env,
            her_buffer_size,
            self.max_episode_length,
            self.goal_selection_strategy,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            self.n_envs,
            self.her_ratio,  # pytype: disable=wrong-arg-types
        )

        # counter for steps in episode
        self.episode_steps = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.model._setup_model()

        # Save reference to her model
        new_store_method = partial(_store_transition, her_model=self)
        # Overwrite default store transition
        self.model._store_transition = types.MethodType(new_store_method, self.model)

        # assign episode storage to replay buffer when using online HER sampling
        if self.online_sampling:
            self.model.replay_buffer = self._episode_storage

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        return self.model.predict(observation, state, mask, deterministic)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "HER",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> BaseAlgorithm:

        self.model.learn(
            total_timesteps,
            callback,
            log_interval,
            eval_env,
            eval_freq,
            n_eval_episodes,
            tb_log_name,
            eval_log_path,
            reset_num_timesteps,
        )

        return self

    def _sample_her_transitions(self) -> None:
        """
        Sample additional goals and store new transitions in replay buffer
        when using offline sampling.
        """

        # Sample goals and get new observations
        # maybe_vec_env=None as we should store unnormalized transitions,
        # they will be normalized at sampling time
        observations, next_observations, actions, rewards = self._episode_storage.sample_offline(
            n_sampled_goal=self.n_sampled_goal
        )

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

    def __getattr__(self, item: str) -> Any:
        """
        Find attribute from model class if this class does not have it.
        """
        if hasattr(self.model, item):
            return getattr(self.model, item)
        else:
            raise AttributeError(f"{self} has no attribute {item}")

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        return self.model._get_torch_save_params()

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default one
        :param include: name of parameters that might be excluded but should be included anyway
        """

        # add HER parameters to model
        self.model.n_sampled_goal = self.n_sampled_goal
        self.model.goal_selection_strategy = self.goal_selection_strategy
        self.model.online_sampling = self.online_sampling
        self.model.model_class = self.model_class
        self.model.max_episode_length = self.max_episode_length
        if exclude is None:
            exclude = ["_episode_storage", "_store_transition"]

        self.model.save(path, exclude, include)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(path, device=device, custom_objects=custom_objects)

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        # check if observation space and action space are part of the saved parameters
        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # check if given env is valid
        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        if "use_sde" in data and data["use_sde"]:
            kwargs["use_sde"] = True

        # Keys that cannot be changed
        for key in {"model_class", "online_sampling", "max_episode_length"}:
            if key in kwargs:
                del kwargs[key]

        # Keys that can be changed
        for key in {"n_sampled_goal", "goal_selection_strategy"}:
            if key in kwargs:
                data[key] = kwargs[key]  # pytype: disable=unsupported-operands
                del kwargs[key]

        # noinspection PyArgumentList
        her_model = cls(
            policy=data["policy_class"],
            env=env,
            model_class=data["model_class"],
            n_sampled_goal=data["n_sampled_goal"],
            goal_selection_strategy=data["goal_selection_strategy"],
            online_sampling=data["online_sampling"],
            max_episode_length=data["max_episode_length"],
            policy_kwargs=data["policy_kwargs"],
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
            **kwargs,
        )

        # load parameters
        her_model.model.__dict__.update(data)
        her_model.model.__dict__.update(kwargs)
        her_model._setup_model()

        her_model._total_timesteps = her_model.model._total_timesteps
        her_model.num_timesteps = her_model.model.num_timesteps
        her_model._episode_num = her_model.model._episode_num

        # put state_dicts back in place
        her_model.model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(her_model.model, name, pytorch_variables[name])

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if her_model.model.use_sde:
            her_model.model.policy.reset_noise()  # pytype: disable=attribute-error
        return her_model

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
        self.model.load_replay_buffer(path=path)

        if self.online_sampling:
            # set environment
            self.replay_buffer.set_env(self.env)
            # If we are at the start of an episode, no need to truncate
            current_idx = self.replay_buffer.current_idx

            # truncate interrupted episode
            if truncate_last_trajectory and current_idx > 0:
                warnings.warn(
                    "The last trajectory in the replay buffer will be truncated.\n"
                    "If you are in the same episode as when the replay buffer was saved,\n"
                    "you should use `truncate_last_trajectory=False` to avoid that issue."
                )
                # get current episode and transition index
                pos = self.replay_buffer.pos
                # set episode length for current episode
                self.replay_buffer.episode_lengths[pos] = current_idx
                # set done = True for current episode
                # current_idx was already incremented
                self.replay_buffer.buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
                # reset current transition index
                self.replay_buffer.current_idx = 0
                # increment episode counter
                self.replay_buffer.pos = (self.replay_buffer.pos + 1) % self.replay_buffer.max_episode_stored
                # update "full" indicator
                self.replay_buffer.full = self.replay_buffer.full or self.replay_buffer.pos == 0
