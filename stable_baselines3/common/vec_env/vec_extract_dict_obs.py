import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for extracting dictionary observations.

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        assert isinstance(
            venv.observation_space, spaces.Dict
        ), f"VecExtractDictObs can only be used with Dict obs space, not {venv.observation_space}"
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        assert isinstance(obs, dict)
        return obs[self.key]

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, infos = self.venv.step_wait()
        assert isinstance(obs, dict)
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"] = info["terminal_observation"][self.key]
        return obs[self.key], reward, done, infos
