import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment

    :param venv: the vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param dim: Dimension along which to stack frames (not including batch dimension).
    """

    def __init__(self, venv: VecEnv, n_stack: int, dim: int = -1):
        self.venv = venv
        self.n_stack = n_stack
        wrapped_obs_space = venv.observation_space
        assert isinstance(wrapped_obs_space, spaces.Box), "VecFrameStack only work with gym.spaces.Box observation space"
        # convert dim to non-negative value, then add 1 to account for leading batch dimension
        self.dim_no_batch = dim if dim >= 0 else len(wrapped_obs_space.shape) + dim
        assert 0 <= self.dim_no_batch < len(wrapped_obs_space.shape), \
            f"dim={dim} out of range obs space with shape {wrapped_obs_space.shape}"
        self.dim_batch = self.dim_no_batch + 1
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=self.dim_no_batch)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=self.dim_no_batch)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        observations, rewards, dones, infos = self.venv.step_wait()
        # Let pytype know that observation is not a dict
        assert isinstance(observations, np.ndarray)
        stack_ax_size = observations.shape[self.dim_batch]
        self.stackedobs = np.roll(self.stackedobs, shift=-stack_ax_size, axis=self.dim_batch)
        for i, done in enumerate(dones):
            if done:
                if "terminal_observation" in infos[i]:
                    old_terminal = infos[i]["terminal_observation"]
                    # this slice expression pulls out all but the oldest observation (which we moved to the back with our
                    # earlier .roll() call)
                    obs_slice = (i, ) + (np.s_[:], ) * (self.dim_batch - 1) + (np.s_[:-stack_ax_size], )
                    new_terminal = np.concatenate((self.stackedobs[obs_slice], old_terminal), axis=self.dim_no_batch)
                    infos[i]["terminal_observation"] = new_terminal
                else:
                    warnings.warn("VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stackedobs[i] = 0
        # replace the most recent obs
        insert_index = (np.s_[:], ) * self.dim_batch + (np.s_[-stack_ax_size:], )
        self.stackedobs[insert_index] = observations
        return self.stackedobs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Reset all environments
        """
        obs: np.ndarray = self.venv.reset()  # pytype:disable=annotation-type-mismatch
        self.stackedobs[...] = 0
        stack_ax_size = obs.shape[-1]
        # this slice expression selects only the final observation
        insert_slice = (np.s_[:], ) * self.dim_batch + (np.s_[-stack_ax_size:], )
        self.stackedobs[insert_slice] = obs
        return self.stackedobs

    def close(self) -> None:
        self.venv.close()
