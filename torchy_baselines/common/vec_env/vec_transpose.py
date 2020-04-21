import warnings

import numpy as np
from gym import spaces

from torchy_baselines.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from torchy_baselines.common.preprocessing import is_image_space


class VecTransposeImage(VecEnvWrapper):
    """
    Re-order channels, from WxHxC to CxWxH.
    :param venv: (VecEnv)
    """
    def __init__(self, venv: VecEnv):
        assert is_image_space(venv.observation_space), 'The observation space must be an image'

        observation_space = self.transpose_space(venv.observation_space)
        super(VecTransposeImage, self).__init__(venv, observation_space=observation_space)

    @staticmethod
    def transpose_space(observation_space: spaces.Box) -> spaces.Box:
        assert is_image_space(observation_space), 'The observation space must be an image'
        width, height, channels = observation_space.shape
        new_shape = (channels, width, height)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

    @staticmethod
    def transpose_image(image: np.ndarray) -> np.ndarray:
        return np.transpose(image, (0, 3, 1, 2))

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        return self.transpose_image(observations), rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Reset all environments
        """
        return self.transpose_image(self.venv.reset())

    def close(self) -> None:
        self.venv.close()
