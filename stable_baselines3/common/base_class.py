import time
import os
import pickle
import warnings
from typing import Union, Type, Optional, Dict, Any, List, Tuple, Callable
from abc import ABC, abstractmethod
from collections import deque

import gym
import torch as th
import numpy as np

from stable_baselines3.common import logger, utils
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy, get_policy_from_name
from stable_baselines3.common.utils import (set_random_seed, get_schedule_fn, update_learning_rate, get_device,
                                            check_for_correct_spaces, safe_mean)
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, unwrap_vec_normalize, VecNormalize, VecTransposeImage
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.save_util import (recursive_getattr, recursive_setattr, save_to_zip_file,
                                                load_from_zip_file)
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, MaybeCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer


class BaseAlgorithm(ABC):
    """
    The base of RL algorithms

    :param policy: (Type[BasePolicy]) Policy object
    :param env: (Union[GymEnv, str]) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: (Type[BasePolicy]) The base policy used by this method
    :param learning_rate: (float or callable) learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param policy_kwargs: (Dict[str, Any]) Additional arguments to be passed to the policy on creation
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param verbose: (int) The verbosity level: 0 none, 1 training information, 2 debug
    :param device: (Union[th.device, str]) Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: (bool) Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: (bool) When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: (Optional[int]) Seed for the pseudo random generators
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    """

    def __init__(self,
                 policy: Type[BasePolicy],
                 env: Union[GymEnv, str],
                 policy_base: Type[BasePolicy],
                 learning_rate: Union[float, Callable],
                 policy_kwargs: Dict[str, Any] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 device: Union[th.device, str] = 'auto',
                 support_multi_env: bool = False,
                 create_eval_env: bool = False,
                 monitor_wrapper: bool = True,
                 seed: Optional[int] = None,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1):

        if isinstance(policy, str) and policy_base is not None:
            self.policy_class = get_policy_from_name(policy_base, policy)
        else:
            self.policy_class = policy

        self.device = get_device(device)
        if verbose > 0:
            print(f"Using {self.device} device")

        self.env = None  # type: Optional[GymEnv]
        # get VecNormalize object if needed
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None  # type: Optional[gym.spaces.Space]
        self.action_space = None  # type: Optional[gym.spaces.Space]
        self.n_envs = None
        self.num_timesteps = 0
        self.eval_env = None
        self.seed = seed
        self.action_noise = None  # type: Optional[ActionNoise]
        self.start_time = None
        self.policy = None
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.lr_schedule = None  # type: Optional[Callable]
        self._last_obs = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[np.ndarray]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1
        # Buffers for logging
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging
        self._n_updates = 0  # type: int

        # Create and wrap the env if needed
        if env is not None:
            if isinstance(env, str):
                if create_eval_env:
                    eval_env = gym.make(env)
                    if monitor_wrapper:
                        eval_env = Monitor(eval_env, filename=None)
                    self.eval_env = DummyVecEnv([lambda: eval_env])
                if self.verbose >= 1:
                    print("Creating environment from the given name, wrapped in a DummyVecEnv.")

                env = gym.make(env)
                if monitor_wrapper:
                    env = Monitor(env, filename=None)
                env = DummyVecEnv([lambda: env])

            env = self._wrap_env(env)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            if not support_multi_env and self.n_envs > 1:
                raise ValueError("Error: the model does not support multiple envs requires a single vectorized"
                                 " environment.")

    def _wrap_env(self, env: GymEnv) -> VecEnv:
        if not isinstance(env, VecEnv):
            if self.verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])

        if is_image_space(env.observation_space) and not isinstance(env, VecTransposeImage):
            if self.verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)
        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """
        Create networks, buffer and optimizers
        """
        raise NotImplementedError()

    def _get_eval_env(self, eval_env: Optional[GymEnv]) -> Optional[GymEnv]:
        """
        Return the environment that will be used for evaluation.

        :param eval_env: (Optional[GymEnv]))
        :return: (Optional[GymEnv])
        """
        if eval_env is None:
            eval_env = self.eval_env

        if eval_env is not None:
            eval_env = self._wrap_env(eval_env)
            assert eval_env.num_envs == 1
        return eval_env

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers: (Union[List[th.optim.Optimizer], th.optim.Optimizer])
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: (Optional[VecEnv]) The current environment
        """
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.
        :return: Optional[VecNormalize] The ``VecNormalize`` env.
        """
        return self._vec_normalize_env

    def set_env(self, env: GymEnv) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        """
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # it must be coherent now
        # if it is not a VecEnv, make it a VecEnv
        env = self._wrap_env(env)

        self.n_envs = env.num_envs
        self.env = env

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variable that will be saved.
        ``th.save`` and ``th.load`` will be used with the right device
        instead of the default pickling strategy.

        :return: (Tuple[List[str], List[str]])
            name of the variables with state dicts to save, name of additional torch tensors,
        """
        state_dicts = ["policy"]

        return state_dicts, []

    @abstractmethod
    def learn(self, total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 100,
              tb_log_name: str = "run",
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> 'BaseAlgorithm':
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples (env steps) to train on
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :param eval_env: (gym.Env) Environment that will be used to evaluate the agent
        :param eval_freq: (int) Evaluate the agent every ``eval_freq`` timesteps (this may vary a little)
        :param n_eval_episodes: (int) Number of episode to evaluate the agent
        :param eval_log_path: (Optional[str]) Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: (bool)
        :return: (BaseAlgorithm) the trained model
        """
        raise NotImplementedError()

    def predict(self, observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (Optional[np.ndarray]) The last states (can be None, used in recurrent policies)
        :param mask: (Optional[np.ndarray]) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, mask, deterministic)

    @classmethod
    def load(cls, load_path: str, env: Optional[GymEnv] = None, **kwargs):
        """
        Load the model from a zip-file

        :param load_path: the location of the saved data
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, tensors = load_from_zip_file(load_path)

        if 'policy_kwargs' in data:
            for arg_to_remove in ['device']:
                if arg_to_remove in data['policy_kwargs']:
                    del data['policy_kwargs'][arg_to_remove]

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError(f"The specified policy kwargs do not equal the stored policy kwargs."
                             f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}")

        # check if observation space and action space are part of the saved parameters
        if ("observation_space" not in data or "action_space" not in data) and "env" not in data:
            raise ValueError("The observation_space and action_space was not given, can't verify new environments")
        # check if given env is valid
        if env is not None:
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        # if no new env was given use stored env if possible
        if env is None and "env" in data:
            env = data["env"]

        # noinspection PyArgumentList
        model = cls(policy=data["policy_class"], env=env, device='auto', _init_setup_model=False)

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        if not hasattr(model, "_setup_model") and len(params) > 0:
            raise NotImplementedError(f"{cls} has no ``_setup_model()`` method")
        model._setup_model()

        # put state_dicts back in place
        for name in params:
            attr = recursive_getattr(model, name)
            attr.load_state_dict(params[name])

        # put tensors back in place
        if tensors is not None:
            for name in tensors:
                recursive_setattr(model, name, tensors[name])

        return model

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed: (int)
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device == th.device('cuda'))
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    def _init_callback(self,
                       callback: Union[None, Callable, List[BaseCallback], BaseCallback],
                       eval_env: Optional[VecEnv] = None,
                       eval_freq: int = 10000,
                       n_eval_episodes: int = 5,
                       log_path: Optional[str] = None) -> BaseCallback:
        """
        :param callback: (Union[callable, [BaseCallback], BaseCallback, None])
        :return: (BaseCallback)
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Create eval callback in charge of the evaluation
        if eval_env is not None:
            eval_callback = EvalCallback(eval_env,
                                         best_model_save_path=log_path,
                                         log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def _setup_learn(self,
                     total_timesteps: int,
                     eval_env: Optional[GymEnv],
                     callback: Union[None, Callable, List[BaseCallback], BaseCallback] = None,
                     eval_freq: int = 10000,
                     n_eval_episodes: int = 5,
                     log_path: Optional[str] = None,
                     reset_num_timesteps: bool = True,
                     tb_log_name: str = 'run',
                     ) -> Tuple[int, 'BaseCallback']:
        """
        Initialize different variables needed for training.

        :param total_timesteps: (int) The total number of samples (env steps) to train on
        :param eval_env: (Optional[GymEnv])
        :param callback: (Union[None, BaseCallback, List[BaseCallback, Callable]])
        :param eval_freq: (int)
        :param n_eval_episodes: (int)
        :param log_path (Optional[str]): Path to a log folder
        :param reset_num_timesteps: (bool) Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: (str) the name of the run for tensorboard log
        :return: (int, Tuple[BaseCallback])
        """
        self.start_time = time.time()
        self.ep_info_buffer = deque(maxlen=100)
        self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs
        utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward and episode length and update the buffer
        if using Monitor wrapper.

        :param infos: ([dict])
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get('episode')
            maybe_is_success = info.get('is_success')
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: ([str]) List of parameters that should be excluded from save
        """
        return ["policy", "device", "env", "eval_env", "replay_buffer", "rollout_buffer", "_vec_normalize_env"]

    def save(self, path: str, exclude: Optional[List[str]] = None, include: Optional[List[str]] = None) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default one
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()
        # use standard list of excluded parameters if none given
        if exclude is None:
            exclude = self.excluded_save_params()
        else:
            # append standard exclude params to the given params
            exclude.extend([param for param in self.excluded_save_params() if param not in exclude])

        # do not exclude params if they are specifically included
        if include is not None:
            exclude = [param_name for param_name in exclude if param_name not in include]

        state_dicts_names, tensors_names = self.get_torch_variables()
        # any params that are in the save vars must not be saved by data
        torch_variables = state_dicts_names + tensors_names
        for torch_var in torch_variables:
            # we need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split('.')[0]
            exclude.append(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            if param_name in data:
                data.pop(param_name, None)

        # Build dict of tensor variables
        tensors = None
        if tensors_names is not None:
            tensors = {}
            for name in tensors_names:
                attr = recursive_getattr(self, name)
                tensors[name] = attr

        # Build dict of state_dicts
        params_to_save = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params_to_save[name] = attr.state_dict()

        save_to_zip_file(path, data=data, params=params_to_save, tensors=tensors)


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: (float or callable) learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param batch_size: (int) Minibatch size for each gradient update
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: (bool) Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: (bool) Whether the model support gSDE or not
    """

    def __init__(self,
                 policy: Type[BasePolicy],
                 env: Union[GymEnv, str],
                 policy_base: Type[BasePolicy],
                 learning_rate: Union[float, Callable],
                 buffer_size: int = int(1e6),
                 learning_starts: int = 100,
                 batch_size: int = 256,
                 policy_kwargs: Dict[str, Any] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 device: Union[th.device, str] = 'auto',
                 support_multi_env: bool = False,
                 create_eval_env: bool = False,
                 monitor_wrapper: bool = True,
                 seed: Optional[int] = None,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False,
                 sde_support: bool = True):

        super(OffPolicyAlgorithm, self).__init__(policy, env, policy_base, learning_rate,
                                               policy_kwargs, tensorboard_log, verbose,
                                               device, support_multi_env, create_eval_env, monitor_wrapper,
                                               seed, use_sde, sde_sample_freq)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs['use_sde'] = self.use_sde
        self.policy_kwargs['device'] = self.device
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup

    def _setup_model(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_space,
                                          self.action_space, self.device)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.lr_schedule, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

    def save_replay_buffer(self, path: str):
        """
        Save the replay buffer as a pickle file.

        :param path: (str) Path to a log folder
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        with open(os.path.join(path, 'replay_buffer.pkl'), 'wb') as file_handler:
            pickle.dump(self.replay_buffer, file_handler)

    def load_replay_buffer(self, path: str):
        """

        :param path: (str) Path to the pickled replay buffer.
        """
        with open(path, 'rb') as file_handler:
            self.replay_buffer = pickle.load(file_handler)
        assert isinstance(self.replay_buffer, ReplayBuffer), 'The replay buffer must inherit from ReplayBuffer class'

    def collect_rollouts(self,  # noqa: C901
                         env: VecEnv,
                         # Type hint as string to avoid circular import
                         callback: 'BaseCallback',
                         n_episodes: int = 1,
                         n_steps: int = -1,
                         action_noise: Optional[ActionNoise] = None,
                         learning_starts: int = 0,
                         replay_buffer: Optional[ReplayBuffer] = None,
                         log_interval: Optional[int] = None) -> RolloutReturn:
        """
        Collect rollout using the current policy (and possibly fill the replay buffer)

        :param env: (VecEnv) The training environment
        :param n_episodes: (int) Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: (int) Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: (Optional[ActionNoise]) Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :param replay_buffer: (ReplayBuffer)
        :param log_interval: (int) Log data every ``log_interval`` episodes
        :return: (RolloutReturn)
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if n_episodes > 0 and n_steps > 0:
            # Note we are refering to the constructor arguments
            # that are named `train_freq` and `n_episodes_rollout`
            # but correspond to `n_steps` and `n_episodes` here
            warnings.warn("You passed a positive value for `train_freq` and `n_episodes_rollout`."
                          "Please make sure this is intended. "
                          "The agent will collect data by stepping in the environment "
                          "until both conditions are true: "
                          "`number of steps in the env` >= `train_freq` and "
                          "`number of episodes` > `n_episodes_rollout`")

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
                    # Warmup phase
                    unscaled_action = np.array([self.action_space.sample()])
                else:
                    # Note: we assume that the policy uses tanh to scale the action
                    # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                    unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

                # Rescale the action from [low, high] to [-1, 1]
                if isinstance(self.action_space, gym.spaces.Box):
                    scaled_action = self.policy.scale_action(unscaled_action)

                    # Add noise to the action (improve exploration)
                    if action_noise is not None:
                        # NOTE: in the original implementation of TD3, the noise was applied to the unscaled action
                        # Update(October 2019): Not anymore
                        scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

                    # We store the scaled action in the buffer
                    buffer_action = scaled_action
                    action = self.policy.unscale_action(scaled_action)
                else:
                    # Discrete case, no need to normalize or clip
                    buffer_action = unscaled_action
                    action = buffer_action

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                    replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_, done)

                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                if 0 < n_steps <= total_steps:
                    break

            if done:
                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    fps = int(self.num_timesteps / (time.time() - self.start_time))
                    logger.record("time/episodes", self._episode_num, exclude="tensorboard")
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        logger.record('rollout/ep_rew_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                        logger.record('rollout/ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                    logger.record("time/fps", fps)
                    logger.record('time/time_elapsed', int(time.time() - self.start_time), exclude="tensorboard")
                    logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
                    if self.use_sde:
                        logger.record("train/std", (self.actor.get_std()).mean().item())

                    if len(self.ep_success_buffer) > 0:
                        logger.record('rollout/success rate', safe_mean(self.ep_success_buffer))
                    # Pass the number of timesteps for tensorboard
                    logger.dump(step=self.num_timesteps)

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable],
                 n_steps: int,
                 gamma: float,
                 gae_lambda: float,
                 ent_coef: float,
                 vf_coef: float,
                 max_grad_norm: float,
                 use_sde: bool,
                 sde_sample_freq: int,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True):

        super(OnPolicyAlgorithm, self).__init__(policy, env, ActorCriticPolicy, learning_rate, policy_kwargs=policy_kwargs,
                                              verbose=verbose, device=device, use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                                              create_eval_env=create_eval_env, support_multi_env=True, seed=seed)

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, self.observation_space,
                                            self.action_space, self.device,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda,
                                            n_envs=self.n_envs)
        self.policy = self.policy_class(self.observation_space, self.action_space,
                                        self.lr_schedule, use_sde=self.use_sde, device=self.device,
                                        **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

    def collect_rollouts(self,
                         env: VecEnv,
                         callback: BaseCallback,
                         rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1
            self.num_timesteps += env.num_envs

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, dones, values, log_probs)
            self._last_obs = new_obs

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()

        return True

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "OnPolicyAlgorithm",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> 'OnPolicyAlgorithm':
        iteration = 0

        total_timesteps, callback = self._setup_learn(total_timesteps, eval_env, callback, eval_freq,
                                                      n_eval_episodes, eval_log_path, reset_num_timesteps,
                                                      tb_log_name)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback,
                                                      self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                  safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean",
                                  safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
