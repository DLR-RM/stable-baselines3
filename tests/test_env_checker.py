import gym
import numpy as np
import pytest
from gym.spaces import Box, Dict, Discrete

from stable_baselines3.common.env_checker import check_env


class ActionDictTestEnv(gym.Env):
    action_space = Dict({"position": Discrete(1), "velocity": Discrete(1)})
    observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5], dtype=self.observation_space.dtype)
        reward = 1
        done = True
        truncated = False
        info = {}
        return observation, reward, done, truncated, info

    def reset(self):
        return np.array([1.0, 1.5, 0.5], dtype=self.observation_space.dtype), {}

    def render(self):
        pass


def test_check_env_dict_action():
    test_env = ActionDictTestEnv()

    with pytest.warns(Warning):
        check_env(env=test_env, warn=True)
