# flake8: noqa F401
import typing
from typing import Optional
from copy import deepcopy

from torchy_baselines.common.vec_env.base_vec_env import AlreadySteppingError, NotSteppingError,\
    VecEnv, VecEnvWrapper, CloudpickleWrapper
from torchy_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from torchy_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from torchy_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from torchy_baselines.common.vec_env.vec_normalize import VecNormalize

# Avoid circular import
if typing.TYPE_CHECKING:
    from torchy_baselines.common.type_aliases import GymEnv


def unwrap_vec_normalize(env: 'GymEnv') -> Optional[VecNormalize]:
    """
    :param env: (gym.Env)
    :return: (VecNormalize)
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


# Define here to avoid circular import
def sync_envs_normalization(env: 'GymEnv', eval_env: 'GymEnv') -> None:
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
