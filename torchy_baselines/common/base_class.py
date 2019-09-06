from abc import ABC, abstractmethod


import numpy as np
import gym

from torchy_baselines.common.policies import get_policy_from_name


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 debug
    :param policy_base: (BasePolicy) the base policy used by this method
    """

    def __init__(self, policy, env, policy_base, policy_kwargs=None, verbose=0):
        if isinstance(policy, str) and policy_base is not None:
            self.policy = get_policy_from_name(policy_base, policy)
        else:
            self.policy = policy
        self.env = env
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self.num_timesteps = 0
        self.params = None

        if env is not None:
            self.env = env
            self.n_envs = 1
            self.observation_space = env.observation_space
            self.action_space = env.action_space

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
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="run",
              reset_num_timesteps=True):
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
