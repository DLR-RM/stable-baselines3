from copy import deepcopy
from typing import Optional, Type, TypeVar

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.common.vec_env.vec_extract_dict_obs import VecExtractDictObs
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

VecEnvWrapperT = TypeVar("VecEnvWrapperT", bound=VecEnvWrapper)


def unwrap_vec_wrapper(env: VecEnv, vec_wrapper_class: Type[VecEnvWrapperT]) -> Optional[VecEnvWrapperT]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: The ``VecEnv`` that is going to be unwrapped
    :param vec_wrapper_class: The desired ``VecEnvWrapper`` class.
    :return: The ``VecEnvWrapper`` object if the ``VecEnv`` is wrapped with the desired wrapper, None otherwise
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, vec_wrapper_class):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


def unwrap_vec_normalize(env: VecEnv) -> Optional[VecNormalize]:
    """
    Retrieve a ``VecNormalize`` object by recursively searching.

    :param env: The VecEnv that is going to be unwrapped
    :return: The ``VecNormalize`` object if the ``VecEnv`` is wrapped with ``VecNormalize``, None otherwise
    """
    return unwrap_vec_wrapper(env, VecNormalize)


def is_vecenv_wrapped(env: VecEnv, vec_wrapper_class: Type[VecEnvWrapper]) -> bool:
    """
    Check if an environment is already wrapped in a given ``VecEnvWrapper``.

    :param env: The VecEnv that is going to be checked
    :param vec_wrapper_class: The desired ``VecEnvWrapper`` class.
    :return: True if the ``VecEnv`` is wrapped with the desired wrapper, False otherwise
    """
    return unwrap_vec_wrapper(env, vec_wrapper_class) is not None


def sync_envs_normalization(env: VecEnv, eval_env: VecEnv) -> None:
    """
    Synchronize the normalization statistics of an eval environment and train environment
    when they are both wrapped in a ``VecNormalize`` wrapper.

    :param env: Training env
    :param eval_env: Environment used for evaluation.
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        assert isinstance(eval_env_tmp, VecEnvWrapper), (
            "Error while synchronizing normalization stats: expected the eval env to be "
            f"a VecEnvWrapper but got {eval_env_tmp} instead. "
            "This is probably due to the training env not being wrapped the same way as the evaluation env. "
            f"Training env type: {env_tmp}."
        )
        if isinstance(env_tmp, VecNormalize):
            assert isinstance(eval_env_tmp, VecNormalize), (
                "Error while synchronizing normalization stats: expected the eval env to be "
                f"a VecNormalize but got {eval_env_tmp} instead. "
                "This is probably due to the training env not being wrapped the same way as the evaluation env. "
                f"Training env type: {env_tmp}."
            )
            # Only synchronize if observation normalization exists
            if hasattr(env_tmp, "obs_rms"):
                eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
        env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv


__all__ = [
    "CloudpickleWrapper",
    "VecEnv",
    "VecEnvWrapper",
    "DummyVecEnv",
    "StackedObservations",
    "SubprocVecEnv",
    "VecCheckNan",
    "VecExtractDictObs",
    "VecFrameStack",
    "VecMonitor",
    "VecNormalize",
    "VecTransposeImage",
    "VecVideoRecorder",
    "unwrap_vec_wrapper",
    "unwrap_vec_normalize",
    "is_vecenv_wrapped",
    "sync_envs_normalization",
]
