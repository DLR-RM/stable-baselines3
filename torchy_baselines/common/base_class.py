import time
from abc import ABCMeta, abstractmethod
from collections import deque
import os
import io
import zipfile

import gym
import torch as th
import numpy as np

from torchy_baselines.common.policies import get_policy_from_name
from torchy_baselines.common.utils import set_random_seed, get_schedule_fn, update_learning_rate
from torchy_baselines.common.vec_env import DummyVecEnv, VecEnv
from torchy_baselines.common.monitor import Monitor
from torchy_baselines.common import logger
from torchy_baselines.common.save_util import data_to_json, json_to_data


class BaseRLModel(object):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 debug
    :param policy_base: (BasePolicy) the base policy used by this method
    :param device: (str or th.device) Device on which the code should.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param monitor_wrapper: (bool) When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    """
    __metaclass__ = ABCMeta

    def __init__(self, policy, env, policy_base, policy_kwargs=None,
                 verbose=0, device='auto', support_multi_env=False,
                 create_eval_env=False, monitor_wrapper=True, seed=None):
        if isinstance(policy, str) and policy_base is not None:
            self.policy_class = get_policy_from_name(policy_base, policy)
        else:
            self.policy_class = policy

        if device == 'auto':
            device = 'cuda' if th.cuda.is_available() else 'cpu'

        self.device = th.device(device)
        if verbose > 0:
            print("Using {} device".format(self.device))

        self.env = env
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self.num_timesteps = 0
        self.eval_env = None
        self.replay_buffer = None
        self.seed = seed
        self.action_noise = None
        # Track the training progress (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress = 1

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

    def _get_eval_env(self, eval_env):
        if eval_env is None:
            eval_env = self.eval_env

        if eval_env is not None:
            if not isinstance(eval_env, VecEnv):
                eval_env = DummyVecEnv([lambda: eval_env])
            assert eval_env.num_envs == 1
        return eval_env

    def scale_action(self, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _setup_learning_rate(self):
        """Transform to callable if needed."""
        self.learning_rate = get_schedule_fn(self.learning_rate)

    def _update_current_progress(self, num_timesteps, total_timesteps):
        """
        Compute current progress (from 1 to 0)

        :param num_timesteps: (int)
        :param total_timesteps: (int)
        """
        self._current_progress = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers):
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress (from 1 to 0).

        :param optimizers: ([th.optim.Optimizer] or Optimizer) An optimizer
            or a list of optimizer.
        """
        # Log the current learning rate
        logger.logkv("learning_rate", self.learning_rate(self._current_progress))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.learning_rate(self._current_progress))

    @staticmethod
    def safe_mean(arr):
        """
        Compute the mean of an array if there is at least one element.
        For empty array, return nan. It is used for logging only.

        :param arr: (np.ndarray)
        :return: (float)
        """
        return np.nan if len(arr) == 0 else np.mean(arr)

    def get_env(self):
        """
        returns the current environment (can be None if not defined)

        :return: (Gym Environment) The current environment
        """
        return self.env

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env: (Gym Environment) The environment for learning a policy
        """
        pass

    def get_policy_parameters(self):
        """
        Get current model policy parameters as dictionary of variable name -> tensors.

        :return: (dict) Dictionary of variable name -> tensor of model's policy parameters.
        """
        return self.policy.state_dict()

    @abstractmethod
    def get_opt_parameters(self):
        """
        Get current model optimizer parameters as dictionary of variable names -> tensors
        :return: (dict) Dictionary of variable name -> tensor of model's optimizer parameters
        """
        raise NotImplementedError()

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """
        raise NotImplementedError()

    @abstractmethod
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run",
              eval_env=None, eval_freq=-1, n_eval_episodes=5, reset_num_timesteps=True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    def load_parameters(self, load_dict, opt_params=None):
        """
        Load model parameters from a dictionary
        load_dict should contain all keys from torch.model.state_dict()
        If opt_params are given this does also load agent's optimizer-parameters, but can only be handled in child classes.


        :param load_dict: (dict) dict of parameters from model.state_dict()
        :param opt_params: (dict of dicts) dict of optimizer state_dicts should be handled in child_class
        """
        if opt_params is not None:
            raise ValueError("Optimizer Parameters where given but no overloaded load function exists for this class")
        self.policy.load_state_dict(load_dict)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        """
        Load the model from a zip-file

        :param load_path: (str) the location of the saved data
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, opt_params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs."
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        if env is None and "env" in data:
            env = data["env"]
        model = cls(policy=data["policy_class"], env=env, _init_setup_model=True)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.load_parameters(params, opt_params)
        return model

    @staticmethod
    def _load_from_file(load_path, load_data=True):
        """ Load model data from a .zip archive

        :param load_path: (str) Where to load the model from
        :param load_data: (bool) Whether we should load and return data
            (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
        :return: (dict),(dict),(dict) Class parameters, model parameters (state_dict) and dict of optimizer parameters (dict of state_dict)
        """
        # Check if file exists if load_path is a string
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".zip"):
                    load_path += ".zip"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

        # Open the zip archive and load data
        try:
            with zipfile.ZipFile(load_path, "r") as file_:
                namelist = file_.namelist()
                # If data or parameters is not in the
                # zip archive, assume they were stored
                # as None (_save_to_file_zip allows this).
                data = None
                params = None
                opt_params = None
                if "data" in namelist and load_data:
                    # Load class parameters and convert to string
                    json_data = file_.read("data").decode()
                    data = json_to_data(json_data)

                if "params.pth" in namelist:
                    # Load parameters with build in torch function
                    with file_.open("params.pth", mode="r") as param_file:
                        # File has to be seekable so load in BytesIO first
                        file_content = io.BytesIO()
                        file_content.write(param_file.read())
                        # go to start of file
                        file_content.seek(0)
                        params = th.load(file_content)
                # check for all other .pth files
                other_files = [file_name for file_name in namelist if
                               os.path.splitext(file_name)[1] == ".pth" and file_name != "params.pth"]
                if len(other_files) > 0:
                    opt_params = dict()
                    for file in other_files:
                        with file_.open(file, mode="r") as opt_param_file:
                            # File has to be seekable so load in BytesIO first
                            file_content = io.BytesIO()
                            file_content.write(opt_param_file.read())
                            # go to start of file
                            file_content.seek(0)
                            opt_params[os.path.splitext(file)[0]] = th.load(file_content)
        except zipfile.BadZipFile:
            # load_path wasn't a zip file
            raise ValueError("Error: the file {} wasn't a zip-file".format(load_path))

        return data, params, opt_params

    def set_random_seed(self, seed=None):
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

    def _setup_learn(self, eval_env):
        self.start_time = time.time()
        self.ep_info_buffer = deque(maxlen=100)
        if self.action_noise is not None:
            self.action_noise.reset()
        timesteps_since_eval, episode_num = 0, 0
        evaluations = []
        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)
        eval_env = self._get_eval_env(eval_env)
        obs = self.env.reset()
        return timesteps_since_eval, episode_num, evaluations, obs, eval_env

    def _update_info_buffer(self, infos):
        """
        Retrieve reward and episode length if using Monitor wrapper.
        :param infos: ([dict])
        """
        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])

    def collect_rollouts(self, env, n_episodes=1, n_steps=-1, action_noise=None,
                         deterministic=False, callback=None,
                         learning_starts=0, num_timesteps=0,
                         replay_buffer=None, obs=None,
                         episode_num=0, log_interval=None):

        episode_rewards = []
        total_timesteps = []
        total_steps, total_episodes = 0, 0
        assert isinstance(env, VecEnv)
        assert env.num_envs == 1

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            # Reset environment: not needed for VecEnv
            # obs = env.reset()
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                # Select action randomly or according to policy
                if num_timesteps < learning_starts:
                    action = np.array([self.action_space.sample()])
                else:
                    action = self.predict(obs, deterministic=deterministic)

                # Rescale the action from [low, high] to [-1, 1]
                action = self.scale_action(action)

                # Add noise to the action (improve exploration)
                if action_noise is not None:
                    # NOTE: in the original implementation of TD3, the noise was applied to the unscaled action
                    action = np.clip(action + action_noise(), -1, 1)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(self.unscale_action(action))

                done_bool = [float(done[0])]
                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos)

                # Store data in replay buffer
                if replay_buffer is not None:
                    replay_buffer.add(obs, new_obs, action, reward, done_bool)

                obs = new_obs

                num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                if n_steps > 0 and total_steps >= n_steps:
                    break

            if done:
                total_episodes += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)
                if action_noise is not None:
                    action_noise.reset()

                # Display training infos
                if self.verbose >= 1 and log_interval is not None and (
                            episode_num + total_episodes) % log_interval == 0:
                    fps = int(num_timesteps / (time.time() - self.start_time))
                    logger.logkv("episodes", episode_num + total_episodes)
                    # logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        logger.logkv('ep_rew_mean', self.safe_mean([ep_info['r'] for ep_info in self.ep_info_buffer]))
                        logger.logkv('ep_len_mean', self.safe_mean([ep_info['l'] for ep_info in self.ep_info_buffer]))
                    # logger.logkv("n_updates", n_updates)
                    # logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - self.start_time))
                    logger.logkv("total timesteps", num_timesteps)
                    logger.dumpkvs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0
        return mean_reward, total_steps, total_episodes, obs

    @staticmethod
    def _save_to_file_zip(save_path, data=None, params=None, opt_params=None):
        """Save model to a zip archive
    
        :param save_path: (str) Where to store the model
        :param data: (dict) Class parameters being stored
        :param params: (dict) Model parameters being stored expected to be state_dict
        :param opt_params: (dict) Optimizer parameters being stored expected to contain an entry for every
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
        with zipfile.ZipFile(save_path, "w") as file_:
            # Do not try to save "None" elements
            if data is not None:
                file_.writestr("data", serialized_data)
            if params is not None:
                with file_.open('params.pth', mode="w") as param_file:
                    th.save(params, param_file)
            if opt_params is not None:
                for file_name, dict in opt_params.items():
                    with file_.open(file_name + '.pth', mode="w") as opt_param_file:
                        th.save(dict, opt_param_file)

    def excluded_save_params(self):
        """
        returns the names of the parameters that should be excluded from save
        :return: (list) List of parameters that should be excluded from save
        """
        return ["replay_buffer"]

    def save(self, path, exclude=None, include=None):
        """
        saves all the params from init and pytorch params in a zip-file for continuous learning

        :param path: (str) path to the file where the data should be saved
        :param exclude: (list) name of parameters that should be excluded, use standard exclude params if None
        :param include: (list) name of parameters that might be excluded but should be included anyway
        :return:
        """
        data = self.__dict__
        # use standard list of excluded parameters if none given
        if exclude is None:
            exclude = self.excluded_save_params()
        # do not exclude params if they are specifically included
        if include is not None:
            exclude = [param_name for param_name in exclude if param_name not in include]

        # remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        params_to_save = self.get_policy_parameters()
        opt_params_to_save = self.get_opt_parameters()
        self._save_to_file_zip(path, data=data, params=params_to_save, opt_params=opt_params_to_save)
