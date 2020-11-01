import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment.
    Frames are concatenated on desired axis (e.g. 4 RGB images -> 12 channels).

    :param venv: the vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param channels_first: If True, stack on first image dimension, otherwise stack on the last (default)
    """

    def __init__(self, venv: VecEnv, n_stack: int, channels_first: bool = False):
        self.venv = venv
        self.n_stack = n_stack
        self.channels_first = channels_first
        # This includes the vec-env dimension (first)
        self.stack_dimension = 1 if self.channels_first else -1
        wrapped_obs_space = venv.observation_space
        assert isinstance(wrapped_obs_space, spaces.Box), "VecFrameStack only work with gym.spaces.Box observation space"
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=0 if channels_first else -1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=0 if channels_first else -1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        observations, rewards, dones, infos = self.venv.step_wait()
        # Let pytype know that observation is not a dict
        assert isinstance(observations, np.ndarray)
        stack_ax_size = observations.shape[self.stack_dimension]
        self.stackedobs = np.roll(self.stackedobs, shift=-stack_ax_size, axis=self.stack_dimension)
        for i, done in enumerate(dones):
            if done:
                if "terminal_observation" in infos[i]:
                    old_terminal = infos[i]["terminal_observation"]
                    if self.channels_first:
                        new_terminal = np.concatenate(
                            (self.stackedobs[i, :-stack_ax_size, ...], old_terminal),
                            axis=self.stack_dimension
                        )
                    else:
                        new_terminal = np.concatenate(
                            (self.stackedobs[i, ..., :-stack_ax_size], old_terminal),
                            axis=self.stack_dimension
                        )
                    infos[i]["terminal_observation"] = new_terminal
                else:
                    warnings.warn("VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stackedobs[i] = 0
        if self.channels_first:
            self.stackedobs[:, -observations.shape[-1]:, ...] = observations
        else:
            self.stackedobs[..., -observations.shape[-1]:] = observations

        return self.stackedobs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Reset all environments
        """
        obs: np.ndarray = self.venv.reset()  # pytype:disable=annotation-type-mismatch
        self.stackedobs[...] = 0
        if self.channels_first:
            self.stackedobs[:, -obs.shape[self.stack_dimension]:, ...] = obs
        else:
            self.stackedobs[..., -obs.shape[self.stack_dimension]:] = obs
        return self.stackedobs

    def close(self) -> None:
        self.venv.close()
