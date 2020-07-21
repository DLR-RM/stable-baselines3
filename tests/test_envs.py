import gym
import numpy as np
import pytest
from gym import spaces

from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.identity_env import (
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)

ENV_CLASSES = [BitFlippingEnv, IdentityEnv, IdentityEnvBox, IdentityEnvMultiBinary, IdentityEnvMultiDiscrete, FakeImageEnv]


@pytest.mark.parametrize("env_id", ["CartPole-v0", "Pendulum-v0"])
def test_env(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    env = gym.make(env_id)
    with pytest.warns(None) as record:
        check_env(env)

    # Pendulum-v0 will produce a warning because the action space is
    # in [-2, 2] and not [-1, 1]
    if env_id == "Pendulum-v0":
        assert len(record) == 1
    else:
        # The other environments must pass without warning
        assert len(record) == 0


@pytest.mark.parametrize("env_class", ENV_CLASSES)
def test_custom_envs(env_class):
    env = env_class()
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
        return env.observation_space.sample(), 0.0, False, {}

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
        # Dict space is not supported by SB when env is not a GoalEnv
        spaces.Dict({"position": spaces.Discrete(5)}),
    ],
)
def test_non_default_spaces(new_obs_space):
    env = FakeImageEnv()
    env.observation_space = new_obs_space
    # Patch methods to avoid errors
    env.reset = new_obs_space.sample

    def patched_step(_action):
        return new_obs_space.sample(), 0.0, False, {}

    env.step = patched_step
    with pytest.warns(UserWarning):
        check_env(env)


def check_reset_assert_error(env, new_reset_return):
    """
    Helper to check that the error is caught.
    :param env: (gym.Env)
    :param new_reset_return: (Any)
    """

    def wrong_reset():
        return new_reset_return

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

    # Return not only the observation
    check_reset_assert_error(env, (env.observation_space.sample(), False))


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
    check_step_assert_error(env, (np.ones((4,)), 1.0, False, {}))
    # Obs is not a numpy array
    check_step_assert_error(env, (1, 1.0, False, {}))

    # Return a wrong reward
    check_step_assert_error(env, (env.observation_space.sample(), np.ones(1), False, {}))

    # Info dict is not returned
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, False))

    # Done is not a boolean
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, 3.0, {}))
    check_step_assert_error(env, (env.observation_space.sample(), 0.0, 1, {}))
