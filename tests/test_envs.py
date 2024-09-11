import types
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.envs import (
    BitFlippingEnv,
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
    SimpleMultiObsEnv,
)

ENV_CLASSES = [
    BitFlippingEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
    FakeImageEnv,
    SimpleMultiObsEnv,
]


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_env(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    env = gym.make(env_id)
    with warnings.catch_warnings(record=True) as record:
        check_env(env)

    # Pendulum-v1 will produce a warning because the action space is
    # in [-2, 2] and not [-1, 1]
    if env_id == "Pendulum-v1":
        assert len(record) == 1
    else:
        # The other environments must pass without warning
        assert len(record) == 0


@pytest.mark.parametrize("env_class", ENV_CLASSES)
def test_custom_envs(env_class):
    env = env_class()
    with warnings.catch_warnings(record=True) as record:
        check_env(env)
    # No warnings for custom envs
    assert len(record) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(continuous=True),
        dict(discrete_obs_space=True),
        dict(image_obs_space=True, channel_first=True),
        dict(image_obs_space=True, channel_first=False),
    ],
)
def test_bit_flipping(kwargs):
    # Additional tests for BitFlippingEnv
    env = BitFlippingEnv(**kwargs)
    with warnings.catch_warnings(record=True) as record:
        check_env(env)

    # No warnings for custom envs
    assert len(record) == 0

    # Remove a key, must throw an error
    obs_space = env.observation_space.spaces["observation"]
    del env.observation_space.spaces["observation"]
    with pytest.raises(AssertionError):
        check_env(env)

    # Rename a key, must throw an error
    env.observation_space.spaces["obs"] = obs_space
    with pytest.raises(AssertionError):
        check_env(env)


def test_high_dimension_action_space():
    """
    Test for continuous action space
    with more than one action.
    """
    env = FakeImageEnv()
    # Patch the action space
    env.action_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)

    # Patch to avoid error
    def patched_step(_action):
        return env.observation_space.sample(), 0.0, False, False, {}

    env.step = patched_step
    check_env(env)


@pytest.mark.parametrize(
    "new_obs_space",
    [
        # Small image
        spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
        # Range not in [0, 255]
        spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.uint8),
        # Wrong dtype
        spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.float32),
        # Not an image, it should be a 1D vector
        spaces.Box(low=-1, high=1, shape=(64, 3), dtype=np.float32),
        # Tuple space is not supported by SB
        spaces.Tuple([spaces.Discrete(5), spaces.Discrete(10)]),
        # Nested dict space is not supported by SB3
        spaces.Dict({"position": spaces.Dict({"abs": spaces.Discrete(5), "rel": spaces.Discrete(2)})}),
        # Small image inside a dict
        spaces.Dict({"img": spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)}),
        # Non zero start index
        spaces.Discrete(3, start=-1),
        # 2D MultiDiscrete
        spaces.MultiDiscrete(np.array([[4, 4], [2, 3]])),
        # Non zero start index (MultiDiscrete)
        spaces.MultiDiscrete([4, 4], start=[1, 0]),
        # Non zero start index inside a Dict
        spaces.Dict({"obs": spaces.Discrete(3, start=1)}),
    ],
)
def test_non_default_spaces(new_obs_space):
    env = FakeImageEnv()
    env.observation_space = new_obs_space

    # Patch methods to avoid errors
    def patched_reset(seed=None):
        return new_obs_space.sample(), {}

    env.reset = patched_reset

    def patched_step(_action):
        return new_obs_space.sample(), 0.0, False, False, {}

    env.step = patched_step
    with pytest.warns(UserWarning):
        check_env(env)


@pytest.mark.parametrize(
    "new_action_space",
    [
        # Not symmetric
        spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        # Wrong dtype
        spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64),
        # Too big range
        spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32),
        # Too small range
        spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
        # Same boundaries
        spaces.Box(low=1, high=1, shape=(2,), dtype=np.float32),
        # Unbounded action space
        spaces.Box(low=-np.inf, high=1, shape=(2,), dtype=np.float32),
        # Almost good, except for one dim
        spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 0.99]), dtype=np.float32),
        # Non zero start index
        spaces.Discrete(3, start=-1),
        # Non zero start index (MultiDiscrete)
        spaces.MultiDiscrete([4, 4], start=[1, 0]),
    ],
)
def test_non_default_action_spaces(new_action_space):
    env = FakeImageEnv(discrete=False)
    # Default, should pass the test
    with warnings.catch_warnings(record=True) as record:
        check_env(env)

    # No warnings for custom envs
    assert len(record) == 0

    # Change the action space
    env.action_space = new_action_space

    # Discrete action space
    if isinstance(new_action_space, (spaces.Discrete, spaces.MultiDiscrete)):
        with pytest.warns(UserWarning):
            check_env(env)
        return

    low, high = new_action_space.low[0], new_action_space.high[0]
    # Unbounded action space throws an error,
    # the rest only warning
    if not np.all(np.isfinite(env.action_space.low)):
        with pytest.raises(AssertionError), pytest.warns(UserWarning):
            check_env(env)
    # numpy >= 1.21 raises a ValueError
    elif int(np.__version__.split(".")[1]) >= 21 and (low > high):
        with pytest.raises(ValueError), pytest.warns(UserWarning):
            check_env(env)
    else:
        with pytest.warns(UserWarning):
            check_env(env)


def check_reset_assert_error(env, new_reset_return):
    """
    Helper to check that the error is caught.
    :param env: (gym.Env)
    :param new_reset_return: (Any)
    """

    def wrong_reset(seed=None):
        return new_reset_return, {}

    # Patch the reset method with a wrong one
    env.reset = wrong_reset
    with pytest.raises(AssertionError):
        check_env(env)


def test_common_failures_reset():
    """
    Test that common failure cases of the `reset_method` are caught
    """
    env = IdentityEnvBox()
    # Return an observation that does not match the observation_space
    check_reset_assert_error(env, np.ones((3,)))
    # The observation is not a numpy array
    check_reset_assert_error(env, 1)

    # Return only obs (gym < 0.26)
    def wrong_reset(self, seed=None):
        return env.observation_space.sample()

    env.reset = types.MethodType(wrong_reset, env)
    with pytest.raises(AssertionError):
        check_env(env)

    # No seed parameter (gym < 0.26)
    def wrong_reset(self):
        return env.observation_space.sample(), {}

    env.reset = types.MethodType(wrong_reset, env)
    with pytest.raises(TypeError):
        check_env(env)

    # Return not only the observation
    check_reset_assert_error(env, (env.observation_space.sample(), False))

    env = SimpleMultiObsEnv()

    # Observation keys and observation space keys must match
    wrong_obs = env.observation_space.sample()
    wrong_obs.pop("img")
    check_reset_assert_error(env, wrong_obs)
    wrong_obs = {**env.observation_space.sample(), "extra_key": None}
    check_reset_assert_error(env, wrong_obs)

    obs, _ = env.reset()

    def wrong_reset(self, seed=None):
        return {"img": obs["img"], "vec": obs["img"]}, {}

    env.reset = types.MethodType(wrong_reset, env)
    with pytest.raises(AssertionError) as excinfo:
        check_env(env)

    # Check that the key is explicitly mentioned
    assert "vec" in str(excinfo.value)


def check_step_assert_error(env, new_step_return=()):
    """
    Helper to check that the error is caught.
    :param env: (gym.Env)
    :param new_step_return: (tuple)
    """

    def wrong_step(_action):
        return new_step_return

    # Patch the step method with a wrong one
    env.step = wrong_step
    with pytest.raises(AssertionError):
        check_env(env)


def test_common_failures_step():
    """
    Test that common failure cases of the `step` method are caught
    """
    env = IdentityEnvBox()

    # Wrong shape for the observation
    check_step_assert_error(env, (np.ones((4,)), 1.0, False, False, {}))
    # Obs is not a numpy array
    check_step_assert_error(env, (1, 1.0, False, False, {}))

    # Return a wrong reward
    check_step_assert_error(env, (env.observation_space.sample(), np.ones(1), False, False, {}))

    # Info dict is not returned
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, False, False))

    # Truncated is not returned (gym < 0.26)
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, False, {}))

    # Done is not a boolean
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, 3.0, False, {}))
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, 1, False, {}))
    # Truncated is not a boolean
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, False, 1.0, {}))

    env = SimpleMultiObsEnv()

    # Observation keys and observation space keys must match
    wrong_obs = env.observation_space.sample()
    wrong_obs.pop("img")
    check_step_assert_error(env, (wrong_obs, 0.0, False, False, {}))
    wrong_obs = {**env.observation_space.sample(), "extra_key": None}
    check_step_assert_error(env, (wrong_obs, 0.0, False, False, {}))

    obs, _ = env.reset()

    def wrong_step(self, action):
        return {"img": obs["vec"], "vec": obs["vec"]}, 0.0, False, False, {}

    env.step = types.MethodType(wrong_step, env)
    with pytest.raises(AssertionError) as excinfo:
        check_env(env)

    # Check that the key is explicitly mentioned
    assert "img" in str(excinfo.value)
