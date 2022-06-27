import gym
import numpy as np
import pytest
from gym.spaces import Box, Dict, Discrete

from stable_baselines3.common.env_checker import check_env


class ActionDictTestEnv(gym.Env):
    action_space = Dict({"position": Discrete(1), "velocity": Discrete(1)})
    observation_space = Dict(
        {
            "x": Box(
                low=np.array([-1, -1]),
                high=np.array([1, 1]),
                shape=(2,),
                dtype=np.float32,
            ),
            "y": Box(
                low=np.array([-1, -1]),
                high=np.array([1, 1]),
                shape=(2,),
                dtype=np.float32,
            ),
        }
    )

    def step(self, action):
        observation = {
            "x": np.array([0.5, -1.0], dtype=np.float32),
            "y": np.array([-0.5, 0.8], dtype=np.float32),
        }
        reward = 1
        done = True
        info = {}
        return observation, reward, done, info

    def reset(self):
        return {
            "x": np.array([0.5, -1.0], dtype=np.float32),
            "y": np.array([-0.5, 0.8], dtype=np.float32),
        }

    def render(self, mode="human"):
        pass


def test_check_env_dict_action():
    test_env = ActionDictTestEnv()

    with pytest.warns(Warning):
        check_env(env=test_env, warn=True)
