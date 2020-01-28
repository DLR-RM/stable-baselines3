import time
import os
import io
import zipfile
from typing import Union, Type, Optional, Dict, Any, List, Tuple, Callable
from abc import ABC, abstractmethod
from collections import deque

import gym
import torch as th
import numpy as np

from torchy_baselines.common import logger
from torchy_baselines.common.policies import BasePolicy, get_policy_from_name
from torchy_baselines.common.utils import set_random_seed, get_schedule_fn, update_learning_rate
from torchy_baselines.common.vec_env import DummyVecEnv, VecEnv, unwrap_vec_normalize
from torchy_baselines.common.monitor import Monitor
from torchy_baselines.common.save_util import data_to_json, json_to_data
from torchy_baselines.common.type_aliases import GymEnv, TensorDict, OptimizerStateDict
from torchy_baselines.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from torchy_baselines.common.noise import ActionNoise


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
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
    :param sde_sample_freq: Sample a new noise matrix every n steps when using SDE
        Default: -1 (only sample at the beginning of the rollout)
    """

    def __init__(self,
                 policy: Type[BasePolicy],
                 env: Union[GymEnv, str],
                 policy_base: Type[BasePolicy],
                 policy_kwargs: Dict[str, Any] = None,
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

        if device == 'auto':
            device = 'cuda' if th.cuda.is_available() else 'cpu'

        self.device = th.device(device)
        if verbose > 0:
            print(f"Using {self.device} device")

        self.env = None  # type: GymEnv
        # get VecNormalize object if needed
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self.num_timesteps = 0
        self.eval_env = None
        self.replay_buffer = None
        self.seed = seed
        self.action_noise = None  # type: ActionNoise
        self.start_time = None
        self.policy, self.actor = None, None
        self.learning_rate = None
        # Used for SDE only
        self.rollout_data = None
        self.on_policy_exploration = False
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress = 1

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

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            if not isinstance(env, VecEnv):
                if self.verbose >= 1:
                    print("Wrapping the env in a DummyVecEnv.")
                env = DummyVecEnv([lambda: env])
            self.n_envs = env.num_envs
            self.env = env

            if not support_multi_env and self.n_envs > 1:
                raise ValueError("Error: the model does not support multiple envs requires a single vectorized"
                                 " environment.")

    def _get_eval_env(self, eval_env: Optional[GymEnv]) -> Optional[GymEnv]:
        """
        Return the environment that will be used for evaluation.
        """
        if eval_env is None:
            eval_env = self.eval_env

        if eval_env is not None:
            if not isinstance(eval_env, VecEnv):
                eval_env = DummyVecEnv([lambda: eval_env])
            assert eval_env.num_envs == 1
        return eval_env

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _setup_learning_rate(self) -> None:
        """Transform to callable if needed."""
        self.learning_rate = get_schedule_fn(self.learning_rate)

    def _update_current_progress(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress (from 1 to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress (from 1 to 0).

        :param optimizers: An optimizer or a list of optimizer.
        """
        # Log the current learning rate
        logger.logkv("learning_rate", self.learning_rate(self._current_progress))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.learning_rate(self._current_progress))

    @staticmethod
    def safe_mean(arr: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compute the mean of an array if there is at least one element.
        For empty array, return NaN. It is used for logging only.

        :param arr:
        :return:
        """
        return np.nan if len(arr) == 0 else np.mean(arr)

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    @staticmethod
    def check_env(env, observation_space: gym.spaces.Space, action_space: gym.spaces.Space) -> bool:
        """
        Checks the validity of the environment and returns if it is consistent.
        Checked parameters:
        - observation_space
        - action_space

        :param observation_space: (gym.spaces.Space)
        :param action_space: (gym.spaces.Space)
        :return: (bool) True if environment seems to be coherent
        """
        if observation_space != env.observation_space:
            return False
        if action_space != env.action_space:
            return False
        # return true if no check failed
        return True

    def set_env(self, env: GymEnv) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        """
        if self.check_env(env, self.observation_space, self.action_space) is False:
            raise ValueError("The given environment is not compatible with model: "
                             "observation and action spaces do not match")
        # it must be coherent now
        # if it is not a VecEnv, make it a VecEnv
        if not isinstance(env, VecEnv):
            if self.verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])
        self.n_envs = env.num_envs
        self.env = env

    def get_parameters(self) -> Tuple[TensorDict, OptimizerStateDict]:
        """
        Returns policy and optimizer parameters as a tuple

        :return: policy_parameters, opt_parameters
        """
        return self.get_policy_parameters(), self.get_opt_parameters()

    def get_policy_parameters(self) -> TensorDict:
        """
        Get current model policy parameters as dictionary of variable name -> tensors.

        :return: Dictionary of variable name -> tensor of model's policy parameters.
        """
        return self.policy.state_dict()

    @abstractmethod
    def get_opt_parameters(self) -> OptimizerStateDict:
        """
        Get current model optimizer parameters as dictionary of variable names -> tensors
        :return: (dict) Dictionary of variable name -> tensor of model's optimizer parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def learn(self, total_timesteps: int,
              callback: Union[None, Callable, List[BaseCallback], BaseCallback] = None,
              log_interval: int = 100,
              tb_log_name: str = "run",
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :param eval_env: (gym.Env) Environment that will be used to evaluate the agent
        :param eval_freq: (int) Evaluate the agent every `eval_freq` timesteps (this may vary a little)
        :param n_eval_episodes: (int) Number of episode to evaluate the agent
        :param eval_log_path: (Optional[str]) Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: (bool)
        :return: (BaseRLModel) the trained model
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> np.ndarray:
        """
        Get the model's action from an observation

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state (used in recurrent policies)
        """
        raise NotImplementedError()

    def load_parameters(self, load_dict: TensorDict, opt_params: OptimizerStateDict) -> None:
        """
        Load model parameters from a dictionary
        load_dict should contain all keys from torch.model.state_dict()
        If opt_params are given this does also load agent's optimizer-parameters,
        but can only be handled in child classes.


        :param load_dict: dict of parameters from model.state_dict()
        :param opt_params: dict of optimizer state_dicts should be handled in child class
        """
        if opt_params is not None:
            raise ValueError("Optimizer Parameters where given but no overloaded load function exists for this class")
        self.policy.load_state_dict(load_dict)

    @classmethod
    def load(cls, load_path: str, env: Optional[GymEnv] = None, **kwargs):
        """
        Load the model from a zip-file

        :param load_path: the location of the saved data
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, opt_params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError(f"The specified policy kwargs do not equal the stored policy kwargs."
                             "Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}")

        # check if observation space and action space are part of the saved parameters
        if ("observation_space" not in data or "action_space" not in data) and "env" not in data:
            raise ValueError("The observation_space and action_space was not given, can't verify new environments")
        # check if given env is valid
        if env is not None and cls.check_env(env, data["observation_space"], data["action_space"]) is False:
            raise ValueError("The given environment does not comply to the model")
        # if no new env was given use stored env if possible
        if env is None and "env" in data:
            env = data["env"]

        # first create model, but only setup if a env was given
        # noinspection PyArgumentList
        model = cls(policy=data["policy_class"], env=env, _init_setup_model=env is not None)

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.load_parameters(params, opt_params)
        return model

    @staticmethod
    def _load_from_file(load_path: str, load_data: bool = True) -> (Tuple[Optional[Dict[str, Any]],
                                                                          Optional[TensorDict],
                                                                          Optional[OptimizerStateDict]]):
        """ Load model data from a .zip archive

        :param load_path: Where to load the model from
        :param load_data: Whether we should load and return data
            (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
        :return: (dict),(dict),(dict) Class parameters, model parameters (state_dict)
            and dict of optimizer parameters (dict of state_dict)
        """
        # Check if file exists if load_path is a string
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".zip"):
                    load_path += ".zip"
                else:
                    raise ValueError(f"Error: the file {load_path} could not be found")

        # Open the zip archive and load data
        try:
            with zipfile.ZipFile(load_path, "r") as archive:
                namelist = archive.namelist()
                # If data or parameters is not in the
                # zip archive, assume they were stored
                # as None (_save_to_file_zip allows this).
                data = None
                params = None
                opt_params = None
                if "data" in namelist and load_data:
                    # Load class parameters and convert to string
                    json_data = archive.read("data").decode()
                    data = json_to_data(json_data)

                if "params.pth" in namelist:
                    # Load parameters with build in torch function
                    with archive.open("params.pth", mode="r") as param_file:
                        # File has to be seekable, but param_file is not, so load in BytesIO first
                        # fixed in python >= 3.7
                        file_content = io.BytesIO()
                        file_content.write(param_file.read())
                        # go to start of file
                        file_content.seek(0)
                        params = th.load(file_content)

                # check for all other .pth files
                other_files = [file_name for file_name in namelist if
                               os.path.splitext(file_name)[1] == ".pth" and file_name != "params.pth"]
                # if there are any other files which end with .pth and aren't "params.pth"
                # assume that they each are optimizer parameters
                if len(other_files) > 0:
                    opt_params = dict()
                    for file_path in other_files:
                        with archive.open(file_path, mode="r") as opt_param_file:
                            # File has to be seekable, but opt_param_file is not, so load in BytesIO first
                            # fixed in python >= 3.7
                            file_content = io.BytesIO()
                            file_content.write(opt_param_file.read())
                            # go to start of file
                            file_content.seek(0)
                            # save the parameters in dict with file name but trim file ending
                            opt_params[os.path.splitext(file_path)[0]] = th.load(file_content)
        except zipfile.BadZipFile:
            # load_path wasn't a zip file
            raise ValueError(f"Error: the file {load_path} wasn't a zip-file")

        return data, params, opt_params

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
            # Same folder as the rest
            best_model_save_path = os.path.dirname(log_path) if log_path is not None else None

            eval_callback = EvalCallback(eval_env,
                                         best_model_save_path=best_model_save_path,
                                         log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def _setup_learn(self,
                     eval_env: Optional[GymEnv],
                     callback: Union[None, Callable, List[BaseCallback], BaseCallback] = None,
                     eval_freq: int = 10000,
                     n_eval_episodes: int = 5,
                     log_path: Optional[str] = None
                     ) -> Tuple[int, np.ndarray, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param eval_env: (Optional[GymEnv])
        :param callback: (Union[None, BaseCallback, List[BaseCallback, Callable]])
        :param eval_freq: (int)
        :param n_eval_episodes: (int)
        :param log_path (Optional[str]):
        :return: (Tuple[int, np.ndarray, BaseCallback])
        """
        self.start_time = time.time()
        self.ep_info_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        timesteps_since_eval, episode_num = 0, 0

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)
        obs = self.env.reset()

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return episode_num, obs, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]]) -> None:
        """
        Retrieve reward and episode length and update the buffer
        if using Monitor wrapper.

        :param infos: ([dict])
        """
        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])

    def collect_rollouts(self,
                         env: VecEnv,
                         callback: 'BaseCallback',  # Type hint as string to avoid circular import
                         n_episodes: int = 1,
                         n_steps: int = -1,
                         action_noise: Optional[ActionNoise] = None,
                         deterministic: bool = False,
                         learning_starts: int = 0,
                         replay_buffer=None,
                         obs: Optional[np.ndarray] = None,
                         episode_num: int = 0,
                         log_interval: Optional[int] = None) -> Tuple[float, int, int, Optional[np.ndarray], bool]:
        """
        Collect rollout using the current policy (and possibly fill the replay buffer)
        TODO: move this method to off-policy base class.

        :param env: (VecEnv)
        :param n_episodes: (int)
        :param n_steps: (int)
        :param action_noise: (ActionNoise)
        :param deterministic: (bool)
        :param callback: (BaseCallback)
        :param learning_starts: (int)
        :param replay_buffer: (ReplayBuffer)
        :param obs: (np.ndarray)
        :param episode_num: (int)
        :param log_interval: (int)
        """
        episode_rewards = []
        total_timesteps = []
        total_steps, total_episodes = 0, 0
        assert isinstance(env, VecEnv)
        assert env.num_envs == 1

        # Retrieve unnormalized observation for saving into the buffer
        if self._vec_normalize_env is not None:
            obs_ = self._vec_normalize_env.get_original_obs()

        self.rollout_data = None
        if self.use_sde:
            self.actor.reset_noise()
            # Reset rollout data
            if self.on_policy_exploration:
                self.rollout_data = {key: [] for key in ['observations', 'actions', 'rewards', 'dones', 'values']}

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            # Reset environment: not needed for VecEnv
            # obs = env.reset()
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                # Only stop training if return value is False, not when it is None.
                if callback() is False:
                    continue_training = False
                    return 0.0, total_steps, total_episodes, None, continue_training

                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                # TODO: use action from policy when using SDE during the warmup phase?
                # if num_timesteps < learning_starts and not self.use_sde:
                if self.num_timesteps < learning_starts:
                    # Warmup phase
                    unscaled_action = np.array([self.action_space.sample()])
                else:
                    unscaled_action = self.predict(obs, deterministic=not self.use_sde)

                # Rescale the action from [low, high] to [-1, 1]
                scaled_action = self.scale_action(unscaled_action)

                if self.use_sde:
                    # When using SDE, the action can be out of bounds
                    # TODO: fix with squashing and account for that in the proba distribution
                    clipped_action = np.clip(scaled_action, -1, 1)
                else:
                    clipped_action = scaled_action

                # Add noise to the action (improve exploration)
                if action_noise is not None:
                    # NOTE: in the original implementation of TD3, the noise was applied to the unscaled action
                    # Update(October 2019): Not anymore
                    clipped_action = np.clip(clipped_action + action_noise(), -1, 1)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(self.unscale_action(clipped_action))

                done_bool = [float(done[0])]
                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        obs_, new_obs_, reward_ = obs, new_obs, reward

                    replay_buffer.add(obs_, new_obs_, clipped_action, reward_, done_bool)

                if self.rollout_data is not None:
                    # Assume only one env
                    self.rollout_data['observations'].append(obs[0].copy())
                    self.rollout_data['actions'].append(scaled_action[0].copy())
                    self.rollout_data['rewards'].append(reward[0].copy())
                    self.rollout_data['dones'].append(np.array(done_bool[0]).copy())
                    obs_tensor = th.FloatTensor(obs).to(self.device)
                    self.rollout_data['values'].append(self.vf_net(obs_tensor)[0].cpu().detach().numpy())

                obs = new_obs
                # Save the true unnormalized observation
                # otherwise obs_ = self._vec_normalize_env.unnormalize_obs(obs)
                # is a good approximation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                if 0 < n_steps <= total_steps:
                    break

            if done:
                total_episodes += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)
                # TODO: reset SDE matrix at the end of the episode?
                if action_noise is not None:
                    action_noise.reset()

                # Display training infos
                if self.verbose >= 1 and log_interval is not None and (
                        episode_num + total_episodes) % log_interval == 0:
                    fps = int(self.num_timesteps / (time.time() - self.start_time))
                    logger.logkv("episodes", episode_num + total_episodes)
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        logger.logkv('ep_rew_mean', self.safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                        logger.logkv('ep_len_mean', self.safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                    # logger.logkv("n_updates", n_updates)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - self.start_time))
                    logger.logkv("total timesteps", self.num_timesteps)
                    if self.use_sde:
                        logger.logkv("std", (self.actor.get_std()).mean().item())
                    logger.dumpkvs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        # Post processing
        if self.rollout_data is not None:
            for key in ['observations', 'actions', 'rewards', 'dones', 'values']:
                self.rollout_data[key] = th.FloatTensor(np.array(self.rollout_data[key])).to(self.device)

            self.rollout_data['returns'] = self.rollout_data['rewards'].clone()
            self.rollout_data['advantage'] = self.rollout_data['rewards'].clone()

            # Compute return and advantage
            last_return = 0.0
            for step in reversed(range(len(self.rollout_data['rewards']))):
                if step == len(self.rollout_data['rewards']) - 1:
                    next_non_terminal = 1.0 - done[0]
                    next_value = self.vf_net(th.FloatTensor(obs).to(self.device))[0].detach()
                    last_return = self.rollout_data['rewards'][step] + next_non_terminal * next_value
                else:
                    next_non_terminal = 1.0 - self.rollout_data['dones'][step + 1]
                    last_return = self.rollout_data['rewards'][step] + self.gamma * last_return * next_non_terminal
                self.rollout_data['returns'][step] = last_return
            self.rollout_data['advantage'] = self.rollout_data['returns'] - self.rollout_data['values']

        callback.on_rollout_end()

        return mean_reward, total_steps, total_episodes, obs, continue_training

    @staticmethod
    def _save_to_file_zip(save_path: str, data: Dict[str, Any] = None,
                          params: TensorDict = None, opt_params: OptimizerStateDict = None) -> None:
        """
        Save model to a zip archive.

        :param save_path: Where to store the model
        :param data: Class parameters being stored
        :param params: Model parameters being stored expected to be state_dict
        :param opt_params: Optimizer parameters being stored expected to contain an entry for every
                                         optimizer with its name and the state_dict
        """

        # data/params can be None, so do not
        # try to serialize them blindly
        if data is not None:
            serialized_data = data_to_json(data)

        # Check postfix if save_path is a string
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".zip"

        # Create a zip-archive and write our objects
        # there. This works when save_path is either
        # str or a file-like
        with zipfile.ZipFile(save_path, "w") as archive:
            # Do not try to save "None" elements
            if data is not None:
                archive.writestr("data", serialized_data)
            if params is not None:
                with archive.open('params.pth', mode="w") as param_file:
                    th.save(params, param_file)
            if opt_params is not None:
                for file_name, dict_ in opt_params.items():
                    with archive.open(file_name + '.pth', mode="w") as opt_param_file:
                        th.save(dict_, opt_param_file)

    @staticmethod
    def excluded_save_params() -> List[str]:
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: ([str]) List of parameters that should be excluded from save
        """
        return ["env", "eval_env", "replay_buffer", "rollout_buffer", "_vec_normalize_env"]

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

        # remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            if param_name in data:
                data.pop(param_name, None)

        params_to_save = self.get_policy_parameters()
        opt_params_to_save = self.get_opt_parameters()
        self._save_to_file_zip(path, data=data, params=params_to_save, opt_params=opt_params_to_save)
