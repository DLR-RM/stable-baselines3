from collections import deque
from typing import Callable, Union, Optional
import random
import os
import glob


import gym
import numpy as np
import torch as th
# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import VecTransposeImage


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators
    :param seed: (int)
    :param using_cuda: (bool)
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: (th.optim.Optimizer)
    :param learning_rate: (float)
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def get_schedule_fn(value_schedule: Union[Callable, float]) -> Callable:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def constant_fn(val: float) -> Callable:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (Callable)
    """

    def func(_):
        return val

    return func


def get_device(device: Union[th.device, str] = 'auto') -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: (Union[str, th.device]) One for 'auto', 'cuda', 'cpu'
    :return: (th.device)
    """
    # Cuda by default
    if device == 'auto':
        device = 'cuda'
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device == th.device('cuda') and not th.cuda.is_available():
        return th.device('cpu')

    return device


def get_latest_run_id(log_path: Optional[str] = None, log_name: str = '') -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def configure_logger(verbose: int = 0, tensorboard_log: Optional[str] = None,
                     tb_log_name: str = '', reset_num_timesteps: bool = True) -> None:
    """
    Configure the logger's outputs.

    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param tb_log_name: (str) tensorboard log
    """
    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        if verbose >= 1:
            logger.configure(save_path, ["stdout", "tensorboard"])
        else:
            logger.configure(save_path, ["tensorboard"])
    elif verbose == 0:
        logger.configure(format_strings=[""])


def check_for_correct_spaces(env: GymEnv, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
    """
    Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
    spaces match after loading the model with given env.
    Checked parameters:
    - observation_space
    - action_space

    :param env: (GymEnv) Environment to check for valid spaces
    :param observation_space: (gym.spaces.Space) Observation space to check against
    :param action_space: (gym.spaces.Space) Action space to check against
    """
    if (observation_space != env.observation_space
        # Special cases for images that need to be transposed
        and not (is_image_space(env.observation_space)
                 and observation_space == VecTransposeImage.transpose_space(env.observation_space))):
        raise ValueError(f'Observation spaces do not match: {observation_space} != {env.observation_space}')
    if action_space != env.action_space:
        raise ValueError(f'Action spaces do not match: {action_space} != {env.action_space}')


def is_vectorized_observation(observation: np.ndarray, observation_space: gym.spaces.Space) -> bool:
    """
    For every observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: (np.ndarray) the input observation to validate
    :param observation_space: (gym.spaces) the observation space
    :return: (bool) whether the given observation is vectorized or not
    """
    if isinstance(observation_space, gym.spaces.Box):
        if observation.shape == observation_space.shape:
            return False
        elif observation.shape[1:] == observation_space.shape:
            return True
        else:
            raise ValueError(f"Error: Unexpected observation shape {observation.shape} for "
                             + f"Box environment, please use {observation_space.shape} "
                             + "or (n_env, {}) for the observation shape."
                             .format(", ".join(map(str, observation_space.shape))))
    elif isinstance(observation_space, gym.spaces.Discrete):
        if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
            return False
        elif len(observation.shape) == 1:
            return True
        else:
            raise ValueError(f"Error: Unexpected observation shape {observation.shape} for "
                             + "Discrete environment, please use (1,) or (n_env, 1) for the observation shape.")

    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        if observation.shape == (len(observation_space.nvec),):
            return False
        elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
            return True
        else:
            raise ValueError(f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
                             + f"environment, please use ({len(observation_space.nvec)},) or "
                             + f"(n_env, {len(observation_space.nvec)}) for the observation shape.")
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        if observation.shape == (observation_space.n,):
            return False
        elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
            return True
        else:
            raise ValueError(f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
                             + f"environment, please use ({observation_space.n},) or "
                             + f"(n_env, {observation_space.n}) for the observation shape.")
    else:
        raise ValueError("Error: Cannot determine if the observation is vectorized "
                         + f" with the space type {observation_space}.")


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
