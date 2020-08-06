# flake8: noqa F401
import typing
from copy import deepcopy
from typing import Optional, Type, Union

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

# Avoid circular import
if typing.TYPE_CHECKING:
    from stable_baselines3.common.type_aliases import GymEnv


def unwrap_vec_wrapper(env: Union["GymEnv", VecEnv], vec_wrapper_class: Type[VecEnvWrapper]) -> Optional[VecEnvWrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: (gym.Env)
    :param vec_wrapper_class: (VecEnvWrapper)
    :return: (VecEnvWrapper)
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, vec_wrapper_class):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


def unwrap_vec_normalize(env: Union["GymEnv", VecEnv]) -> Optional[VecNormalize]:
    """
    :param env: (gym.Env)
    :return: (VecNormalize)
    """
    return unwrap_vec_wrapper(env, VecNormalize)  # pytype:disable=bad-return-type


# Define here to avoid circular import
def sync_envs_normalization(env: "GymEnv", eval_env: "GymEnv") -> None:
    """
    Sync eval env and train env when using VecNormalize

    :param env: (GymEnv)
    :param eval_env: (GymEnv)
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
        env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv
