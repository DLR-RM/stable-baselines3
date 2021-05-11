from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedDictObservations, StackedObservations


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment. Designed for image observations.

    Uses the StackedObservations class, or StackedDictObservations depending on the observations space

    :param venv: the vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
        Alternatively channels_order can be a dictionary which can be used with environments with Dict observation spaces
    """

    def __init__(self, venv: VecEnv, n_stack: int, channels_order: Optional[Union[str, Dict[str, str]]] = None):
        self.venv = venv
        self.n_stack = n_stack

        wrapped_obs_space = venv.observation_space

        if isinstance(wrapped_obs_space, spaces.Box):
            assert not isinstance(
                channels_order, dict
            ), f"Expected None or string for channels_order but received {channels_order}"
            self.stackedobs = StackedObservations(venv.num_envs, n_stack, wrapped_obs_space, channels_order)

        elif isinstance(wrapped_obs_space, spaces.Dict):
            self.stackedobs = StackedDictObservations(venv.num_envs, n_stack, wrapped_obs_space, channels_order)

        else:
            raise Exception("VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces")

        observation_space = self.stackedobs.stack_observation_space(wrapped_obs_space)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(
        self,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, Any]],]:

        observations, rewards, dones, infos = self.venv.step_wait()

        observations, infos = self.stackedobs.update(observations, dones, infos)

        return observations, rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        """
        observation = self.venv.reset()  # pytype:disable=annotation-type-mismatch

        observation = self.stackedobs.reset(observation)
        return observation

    def close(self) -> None:
        self.venv.close()
