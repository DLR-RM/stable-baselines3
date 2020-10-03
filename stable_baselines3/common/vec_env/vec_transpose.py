import numpy as np
from gym import spaces

from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class VecTransposeImage(VecEnvWrapper):
    """
    Re-order channels, from HxWxC to CxHxW.
    It is required for PyTorch convolution layers.

    :param venv:
    """

    def __init__(self, venv: VecEnv):
        assert is_image_space(venv.observation_space), "The observation space must be an image"

        observation_space = self.transpose_space(venv.observation_space)
        super(VecTransposeImage, self).__init__(venv, observation_space=observation_space)

    @staticmethod
    def transpose_space(observation_space: spaces.Box) -> spaces.Box:
        """
        Transpose an observation space (re-order channels).

        :param observation_space:
        :return:
        """
        assert is_image_space(observation_space), "The observation space must be an image"
        width, height, channels = observation_space.shape
        new_shape = (channels, width, height)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

    @staticmethod
    def transpose_image(image: np.ndarray) -> np.ndarray:
        """
        Transpose an image or batch of images (re-order channels).

        :param image:
        :return:
        """
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return np.transpose(image, (0, 3, 1, 2))

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        return self.transpose_image(observations), rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Reset all environments
        """
        return self.transpose_image(self.venv.reset())

    def close(self) -> None:
        self.venv.close()
