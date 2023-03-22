import gym
import numpy as np
import pytest
from gym import spaces

from stable_baselines3.common.env_checker import check_env


class ActionDictTestEnv(gym.Env):
    action_space = spaces.Dict({"position": spaces.Discrete(1), "velocity": spaces.Discrete(1)})
    observation_space = spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5], dtype=self.observation_space.dtype)
        reward = 1
        done = True
        info = {}
        return observation, reward, done, info

    def reset(self):
        return np.array([1.0, 1.5, 0.5], dtype=self.observation_space.dtype)

    def render(self, mode="human"):
        pass


def test_check_env_dict_action():
    test_env = ActionDictTestEnv()

    with pytest.warns(Warning):
        check_env(env=test_env, warn=True)


def test_check_env_observation_space_shape():
    class ObservationSpaceShapeTestEnv(gym.Env):
        action_space = spaces.Dict({"position": spaces.Discrete(1), "velocity": spaces.Discrete(1)})
        observation_space = spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

        def reset(self):
            return np.array([[1.0, 1.5, 0.5], [1.0, 1.5, 0.5]], dtype=self.observation_space.dtype)

    test_env = ObservationSpaceShapeTestEnv()
    with pytest.raises(AssertionError, match="Expected: \(3,\), actual: \(2, 3\)"):
        check_env(env=test_env)


def test_check_env_observation_space_dtype():
    class ObservationSpaceDTypeTestEnv(gym.Env):
        action_space = spaces.Dict({"position": spaces.Discrete(1), "velocity": spaces.Discrete(1)})
        observation_space = spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

        def reset(self):
            return np.array([1.0, 1.5, 0.5], dtype=np.float64)

    test_env = ObservationSpaceDTypeTestEnv()
    with pytest.raises(AssertionError, match="Expected: float32, actual: float64"):
        check_env(env=test_env)


def test_check_env_observation_space_lower_bound():
    class ObservationSpaceDTypeTestEnv(gym.Env):
        action_space = spaces.Dict({"position": spaces.Discrete(1), "velocity": spaces.Discrete(1)})
        observation_space = spaces.Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32)

        def reset(self):
            return np.array([-1.0, 1.5, 0.5], dtype=np.float32)

    test_env = ObservationSpaceDTypeTestEnv()
    with pytest.raises(AssertionError, match="Expected: obs >= 0\.0, actual: -1\.0"):
        check_env(env=test_env)


def test_check_env_observation_space_upper_bound():
    class ObservationSpaceDTypeTestEnv(gym.Env):
        action_space = spaces.Dict({"position": spaces.Discrete(1), "velocity": spaces.Discrete(1)})
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        def reset(self):
            return np.array([1.0, 1.5, 0.5], dtype=np.float32)

    test_env = ObservationSpaceDTypeTestEnv()
    with pytest.raises(AssertionError, match="Expected: obs <= 1\.0, actual: 1\.5"):
        check_env(env=test_env)
