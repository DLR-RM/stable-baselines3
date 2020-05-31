from typing import Callable, Union, Optional
import random

import os
import glob
import numpy as np
import torch as th
# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from stable_baselines3.common import logger


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
