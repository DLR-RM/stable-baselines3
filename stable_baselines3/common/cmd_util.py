import os
import warnings
from typing import Dict, Any, Optional, Callable, Type, Union

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_vec_env(env_id: Union[str, Type[gym.Env]],
                 n_envs: int = 1,
                 seed: Optional[int] = None,
                 start_index: int = 0,
                 monitor_dir: Optional[str] = None,
                 wrapper_class: Optional[Callable] = None,
                 env_kwargs: Optional[Dict[str, Any]] = None,
                 vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
                 vec_env_kwargs: Optional[Dict[str, Any]] = None):
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the ``VecEnv`` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id)
                if len(env_kwargs) > 0:
                    warnings.warn("No environment class was passed (only an env ID) so ``env_kwargs`` will be ignored")
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def make_atari_env(env_id: Union[str, Type[gym.Env]],
                   n_envs: int = 1,
                   seed: Optional[int] = None,
                   start_index: int = 0,
                   monitor_dir: Optional[str] = None,
                   wrapper_kwargs: Optional[Dict[str, Any]] = None,
                   env_kwargs: Optional[Dict[str, Any]] = None,
                   vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
                   vec_env_kwargs: Optional[Dict[str, Any]] = None):
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (Dict[str, Any]) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def atari_wrapper(env: gym.Env) -> gym.Env:
        env = AtariWrapper(env, **wrapper_kwargs)
        return env

    return make_vec_env(env_id, n_envs=n_envs, seed=seed, start_index=start_index,
                        monitor_dir=monitor_dir, wrapper_class=atari_wrapper,
                        env_kwargs=env_kwargs, vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)
