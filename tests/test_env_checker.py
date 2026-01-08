from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env


class ActionDictTestEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    render_mode = None

    action_space = spaces.Dict({"position": spaces.Discrete(1), "velocity": spaces.Discrete(1)})
    observation_space = spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5], dtype=self.observation_space.dtype)
        reward = 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        return np.array([1.0, 1.5, 0.5], dtype=self.observation_space.dtype), {}

    def render(self):
        pass


def test_check_env_dict_action():
    test_env = ActionDictTestEnv()

    with pytest.warns(Warning):
        check_env(env=test_env, warn=True)


class CustomEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 2}

    def __init__(self, render_mode=None):
        # Test Sequence obs
        self.observation_space = spaces.Sequence(spaces.Discrete(8))
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 1.0, False, False, {}


@pytest.mark.parametrize(
    "obs_tuple",
    [
        # Above upper bound
        (
            spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([2.0, 1.0, 1.0]), shape=(3,), dtype=np.float32),
            np.array([1.0, 1.5, 0.5], dtype=np.float32),
            r"Expected: 0\.0 <= obs\[1] <= 1\.0, actual value: 1\.5",
        ),
        # Above upper bound (multi-dim)
        (
            spaces.Box(low=-1.0, high=2.0, shape=(2, 3, 3, 1), dtype=np.float32),
            3.0 * np.ones((2, 3, 3, 1), dtype=np.float32),
            # Note: this is one of the 18 invalid indices
            r"Expected: -1\.0 <= obs\[1,2,1,0\] <= 2\.0, actual value: 3\.0",
        ),
        # Below lower bound
        (
            spaces.Box(low=np.array([0.0, -10.0, 0.0]), high=np.array([2.0, 1.0, 1.0]), shape=(3,), dtype=np.float32),
            np.array([-1.0, 1.5, 0.5], dtype=np.float32),
            r"Expected: 0\.0 <= obs\[0] <= 2\.0, actual value: -1\.0",
        ),
        # Below lower bound (multi-dim)
        (
            spaces.Box(low=-1.0, high=2.0, shape=(2, 3, 3, 1), dtype=np.float32),
            -2 * np.ones((2, 3, 3, 1), dtype=np.float32),
            r"18 invalid indices:",
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

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            return wrong_obs if method == "reset" else good_obs, {}

        def step(self, action):
            obs = wrong_obs if method == "step" else good_obs
            return obs, 0.0, True, False, {}

    TestEnv.observation_space = observation_space

    test_env = TestEnv()
    with pytest.raises(AssertionError, match=error_message):
        check_env(env=test_env, warn=False)


class LimitedStepsTestEnv(gym.Env):
    action_space = spaces.Discrete(n=2)
    observation_space = spaces.Discrete(n=2)

    def __init__(self, steps_before_termination: int = 1):
        super().__init__()

        assert steps_before_termination >= 1
        self._steps_before_termination = steps_before_termination

        self._steps_called = 0
        self._terminated = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        super().reset(seed=seed)

        self._steps_called = 0
        self._terminated = False

        return 0, {}

    def step(self, action: np.ndarray) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self._steps_called += 1

        assert not self._terminated

        observation = 0
        reward = 0.0
        self._terminated = self._steps_called >= self._steps_before_termination
        truncated = False

        return observation, reward, self._terminated, truncated, {}

    def render(self) -> None:
        pass


def test_check_env_single_step_env():
    test_env = LimitedStepsTestEnv(steps_before_termination=1)

    # This should not throw
    check_env(env=test_env, warn=True)


class SimpleGraphEnv(CustomEnv):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(low=0, high=1, shape=(2,)),
            edge_space=spaces.Box(low=0, high=1, shape=(3,)),
        )


class SimpleDictGraphEnv(CustomEnv):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "test": spaces.Graph(
                    node_space=spaces.Box(low=0, high=1, shape=(2,)),
                    edge_space=spaces.Box(low=0, high=1, shape=(3,)),
                )
            }
        )


def test_check_env_graph_space():
    # Should emit a warning about Graph space, but not fail
    with pytest.warns(UserWarning, match=r"Graph.*not supported"):
        check_env(SimpleGraphEnv(), warn=True)

    with pytest.warns(UserWarning, match=r"Graph.*not supported"):
        check_env(SimpleDictGraphEnv(), warn=True)


class SequenceInDictEnv(CustomEnv):
    """Test env with Sequence space inside Dict space."""

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {"seq": spaces.Sequence(spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32))}
        )


class SequenceInTupleEnv(CustomEnv):
    """Test env with Sequence space inside Tuple space."""

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((spaces.Sequence(spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)),))


class SequenceInOneOfEnv(CustomEnv):
    """Test env with Sequence space inside OneOf space."""

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.OneOf(
            (
                spaces.Sequence(spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)),
                spaces.Discrete(3),
            )
        )


@pytest.mark.parametrize("env_class", [CustomEnv, SequenceInDictEnv])
def test_check_env_sequence_obs(env_class):
    with pytest.warns(Warning, match=r"Sequence.*not supported"):
        check_env(env_class(), warn=True)


def test_check_env_sequence_tuple():
    with (
        pytest.warns(Warning, match=r"Sequence.*not supported"),
        pytest.warns(Warning, match=r"Tuple.*not supported"),
    ):
        check_env(SequenceInTupleEnv(), warn=True)


def test_check_env_oneof():
    try:
        env = SequenceInOneOfEnv()
    except AttributeError:
        pytest.skip("OneOf not supported by current Gymnasium version")

    with pytest.warns(Warning, match=r"OneOf.*not supported"):
        check_env(env, warn=True)
