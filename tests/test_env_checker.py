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


@pytest.mark.parametrize(
    "obs_tuple",
    [
        # Above upper bound
        (
            spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            np.array([1.0, 1.5, 0.5], dtype=np.float32),
            r"Expected: obs <= 1\.0, actual max value: 1\.5 at index 1",
        ),
        # Below lower bound
        (
            spaces.Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32),
            np.array([-1.0, 1.5, 0.5], dtype=np.float32),
            r"Expected: obs >= 0\.0, actual min value: -1\.0 at index 0",
        ),
        # Wrong dtype
        (
            spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32),
            np.array([1.0, 1.5, 0.5], dtype=np.float64),
            r"Expected: float32, actual dtype: float64",
        ),
        # Wrong shape
        (
            spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32),
            np.array([[1.0, 1.5, 0.5], [1.0, 1.5, 0.5]], dtype=np.float32),
            r"Expected: \(3,\), actual shape: \(2, 3\)",
        ),
        # Wrong shape (dict obs)
        (
            spaces.Dict({"obs": spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)}),
            {"obs": np.array([[1.0, 1.5, 0.5], [1.0, 1.5, 0.5]], dtype=np.float32)},
            r"Error while checking key=obs.*Expected: \(3,\), actual shape: \(2, 3\)",
        ),
        # Wrong shape (multi discrete)
        (
            spaces.MultiDiscrete([3, 3]),
            np.array([[2, 0]]),
            r"Expected: \(2,\), actual shape: \(1, 2\)",
        ),
        # Wrong shape (multi binary)
        (
            spaces.MultiBinary(3),
            np.array([[1, 0, 0]]),
            r"Expected: \(3,\), actual shape: \(1, 3\)",
        ),
    ],
)
@pytest.mark.parametrize(
    # Check when it happens at reset or during step
    "method",
    ["reset", "step"],
)
def test_check_env_detailed_error(obs_tuple, method):
    """
    Check that the env checker returns more detail error
    when the observation is not in the obs space.
    """
    observation_space, wrong_obs, error_message = obs_tuple
    good_obs = observation_space.sample()

    class TestEnv(gym.Env):
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        def reset(self):
            return wrong_obs if method == "reset" else good_obs

        def step(self, action):
            obs = wrong_obs if method == "step" else good_obs
            return obs, 0.0, True, {}

    TestEnv.observation_space = observation_space

    test_env = TestEnv()
    with pytest.raises(AssertionError, match=error_message):
        check_env(env=test_env)
