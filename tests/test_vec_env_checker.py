import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env_checker import check_vecenv


class BrokenVecEnv:
    """A broken VecEnv that doesn't inherit from VecEnv."""

    def __init__(self):
        self.num_envs = 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.action_space = spaces.Discrete(2)


class MissingAttributeVecEnv(VecEnv):
    """A VecEnv missing required attributes."""

    def __init__(self):
        # Intentionally not calling super().__init__
        pass

    def reset(self):
        pass

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * getattr(self, "num_envs", 1)


class WrongShapeVecEnv(VecEnv):
    """A VecEnv that returns wrong-shaped observations."""

    def __init__(self):
        super().__init__(
            num_envs=2, observation_space=spaces.Box(low=-1.0, high=1.0, shape=(3,)), action_space=spaces.Discrete(2)
        )

    def reset(self):
        # Return wrong shape (should be (2, 3) but return (3,))
        return np.zeros(3)

    def step_async(self, actions):
        pass

    def step_wait(self):
        # Return wrong shapes
        obs = np.zeros(3)  # Should be (2, 3)
        rewards = np.zeros(3)  # Should be (2,)
        dones = np.zeros(3)  # Should be (2,)
        infos = [{}]  # Should be [{}, {}] - list or tuple with 2 elements
        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


def test_check_vecenv_basic():
    """Test basic VecEnv checker functionality with a working VecEnv."""

    def make_env():
        return gym.make("CartPole-v1")

    vec_env = DummyVecEnv([make_env for _ in range(2)])

    try:
        # Should pass without issues
        check_vecenv(vec_env, warn=True)
    finally:
        vec_env.close()


def test_check_vecenv_not_vecenv():
    """Test that check_vecenv raises error for non-VecEnv objects."""

    broken_env = BrokenVecEnv()

    with pytest.raises(AssertionError, match=r"must inherit from.*VecEnv"):
        check_vecenv(broken_env)


def test_check_vecenv_missing_attributes():
    """Test that check_vecenv raises error for VecEnv with missing attributes."""

    broken_env = MissingAttributeVecEnv()

    with pytest.raises(AssertionError, match=r"must have.*attribute"):
        check_vecenv(broken_env)


def test_check_vecenv_wrong_shapes():
    """Test that check_vecenv catches wrong-shaped observations and returns."""

    broken_env = WrongShapeVecEnv()

    try:
        with pytest.raises(AssertionError, match="Expected observation shape"):
            check_vecenv(broken_env)
    finally:
        broken_env.close()


def test_check_vecenv_dict_space():
    """Test VecEnv checker with Dict observation space."""

    class DictEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=-1.0, high=1.0, shape=(4,)),
                    "achieved_goal": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
                }
            )
            self.action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return {
                "observation": np.zeros(4),
                "achieved_goal": np.zeros(2),
            }, {}

        def step(self, action):
            obs = {
                "observation": np.zeros(4),
                "achieved_goal": np.zeros(2),
            }
            return obs, 0.0, False, False, {}

    def make_dict_env():
        return DictEnv()

    vec_env = DummyVecEnv([make_dict_env for _ in range(2)])

    try:
        check_vecenv(vec_env, warn=True)
    finally:
        vec_env.close()


def test_check_vecenv_warnings():
    """Test that check_vecenv emits appropriate warnings."""

    class BoxActionEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
            # Asymmetric action space should trigger warning
            self.action_space = spaces.Box(low=-2.0, high=3.0, shape=(2,))

        def reset(self, *, seed=None, options=None):
            return np.zeros(4), {}

        def step(self, action):
            return np.zeros(4), 0.0, False, False, {}

    def make_box_env():
        return BoxActionEnv()

    vec_env = DummyVecEnv([make_box_env for _ in range(2)])

    try:
        with pytest.warns(UserWarning, match="symmetric and normalized Box action space"):
            check_vecenv(vec_env, warn=True)
    finally:
        vec_env.close()
