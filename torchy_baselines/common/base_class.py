import time
from abc import ABCMeta, abstractmethod
from collections import deque

import gym
import torch as th
import numpy as np

from torchy_baselines.common.policies import get_policy_from_name
from torchy_baselines.common.utils import set_random_seed
from torchy_baselines.common.vec_env import DummyVecEnv, VecEnv
from torchy_baselines.common.monitor import Monitor
from torchy_baselines.common import logger


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
            self.policy = get_policy_from_name(policy_base, policy)
        else:
            self.policy = policy

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
        self.params = None
        self.eval_env = None
        self.replay_buffer = None
        self.seed = seed
        self.action_noise = None

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
        return low + (0.5 * (scaled_action + 1.0) * (high -  low))

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

    def get_parameter_list(self):
        """
        Get pytorch Variables of model's parameters

        This includes all variables necessary for continuing training (saving / loading).

        :return: (list) List of pytorch Variables
        """
        pass

    def get_parameters(self):
        """
        Get current model parameters as dictionary of variable name -> ndarray.

        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
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

    def load_parameters(self, load_path_or_dict, exact_match=True):
        """
        Load model parameters from a file or a dictionary

        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.

        This does not load agent's hyper-parameters.

        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.

        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: (str or file-like object) the save location
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param kwargs: extra arguments to change the model when loading
        """
        raise NotImplementedError()

    def set_random_seed(self, seed=0):
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
                if self.verbose >= 1 and log_interval is not None and (episode_num + total_episodes) % log_interval == 0:
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
