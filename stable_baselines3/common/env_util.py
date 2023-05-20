import os
from typing import Any, Callable, Dict, Optional, Type, Union

import gymnasium as gym

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def unwrap_wrapper(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env


def make_atari_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=AtariWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )
