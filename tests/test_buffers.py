from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from stable_baselines3 import A2C
from stable_baselines3.common.buffers import DictReplayBuffer, DictRolloutBuffer, ReplayBuffer, RolloutBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


class DummyEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        self.action_space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Box(1, 5, (1,))
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self, *, seed=None, options=None):
        self._t = 0
        obs = self._observations[0]
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = self._observations[index]
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


class DummyDictEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        # Test for multi-dim action space
        self.action_space = spaces.Box(1, 5, shape=(10, 7))
        space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Dict({"observation": space, "achieved_goal": space, "desired_goal": space})
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self, seed=None, options=None):
        self._t = 0
        obs = {key: self._observations[0] for key in self.observation_space.spaces.keys()}
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = {key: self._observations[index] for key in self.observation_space.spaces.keys()}
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


@pytest.mark.parametrize("env_cls", [DummyEnv, DummyDictEnv])
def test_env(env_cls):
    # Check the env used for testing
    # Do not warn for asymmetric space
    check_env(env_cls(), warn=False, skip_render_check=True)


@pytest.mark.parametrize("replay_buffer_cls", [ReplayBuffer, DictReplayBuffer])
def test_replay_buffer_normalization(replay_buffer_cls):
    env = {ReplayBuffer: DummyEnv, DictReplayBuffer: DummyDictEnv}[replay_buffer_cls]
    env = make_vec_env(env)
    env = VecNormalize(env)

    buffer = replay_buffer_cls(100, env.observation_space, env.action_space, device="cpu")

    # Interact and store transitions
    env.reset()
    obs = env.get_original_obs()
    for _ in range(100):
        action = env.action_space.sample()
        _, _, done, info = env.step(action)
        next_obs = env.get_original_obs()
        reward = env.get_original_reward()
        buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    sample = buffer.sample(50, env)
    # Test observation normalization
    for observations in [sample.observations, sample.next_observations]:
        if isinstance(sample, DictReplayBufferSamples):
            for key in observations.keys():
                assert th.allclose(observations[key].mean(0), th.zeros(1), atol=1)
        elif isinstance(sample, ReplayBufferSamples):
            assert th.allclose(observations.mean(0), th.zeros(1), atol=1)
    # Test reward normalization
    assert np.allclose(sample.rewards.mean(0), np.zeros(1), atol=1)


@pytest.mark.parametrize("replay_buffer_cls", [DictReplayBuffer, DictRolloutBuffer, ReplayBuffer, RolloutBuffer])
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_device_buffer(replay_buffer_cls, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    env = {
        RolloutBuffer: DummyEnv,
        DictRolloutBuffer: DummyDictEnv,
        ReplayBuffer: DummyEnv,
        DictReplayBuffer: DummyDictEnv,
    }[replay_buffer_cls]
    env = make_vec_env(env)

    buffer = replay_buffer_cls(100, env.observation_space, env.action_space, device=device)

    # Interact and store transitions
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
            episode_start, values, log_prob = np.zeros(1), th.zeros(1), th.ones(1)
            buffer.add(obs, action, reward, episode_start, values, log_prob)
        else:
            buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    # Get data from the buffer
    if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
        # get returns an iterator over minibatches
        data = buffer.get(50)
    elif replay_buffer_cls in [ReplayBuffer, DictReplayBuffer]:
        data = [buffer.sample(50)]

    # Check that all data are on the desired device
    desired_device = get_device(device).type
    for minibatch in list(data):
        for value in minibatch:
            if isinstance(value, dict):
                for key in value.keys():
                    assert value[key].device.type == desired_device
            elif isinstance(value, th.Tensor):
                assert value.device.type == desired_device
            elif isinstance(value, np.ndarray):
                # For prioritized replay weights/indices
                pass
            elif value is None:
                # discounts factors are only set for n-step replay buffer
                pass
            else:
                raise TypeError(f"Unknown value type: {type(value)}")


@pytest.mark.parametrize(
    "obs_dtype",
    [
        np.dtype(np.uint8),
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.int16),
        np.dtype(np.uint32),
        np.dtype(np.int32),
        np.dtype(np.uint64),
        np.dtype(np.int64),
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ],
)
@pytest.mark.parametrize("use_dict", [False, True])
@pytest.mark.parametrize(
    "action_space",
    [
        spaces.Discrete(10),
        spaces.Box(low=-1.0, high=1.0, dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, dtype=np.float64),
    ],
)
def test_buffer_dtypes(obs_dtype, use_dict, action_space):
    obs_space = spaces.Box(0, 100, dtype=obs_dtype)
    buffer_params = dict(buffer_size=1, action_space=action_space)
    # For off-policy algorithms, we cast float64 actions to float32, see GH#1145
    actual_replay_action_dtype = ReplayBuffer._maybe_cast_dtype(action_space.dtype)
    # For on-policy, we cast at sample time to float32 for backward compat
    # and to avoid issue computing log prob with multibinary
    actual_rollout_action_dtype = np.float32

    if use_dict:
        dict_obs_space = spaces.Dict({"obs": obs_space, "obs_2": spaces.Box(0, 100, dtype=np.uint8)})
        buffer_params["observation_space"] = dict_obs_space
        rollout_buffer = DictRolloutBuffer(**buffer_params)
        replay_buffer = DictReplayBuffer(**buffer_params)
        assert rollout_buffer.observations["obs"].dtype == obs_dtype
        assert replay_buffer.observations["obs"].dtype == obs_dtype
        assert rollout_buffer.observations["obs_2"].dtype == np.uint8
        assert replay_buffer.observations["obs_2"].dtype == np.uint8
    else:
        buffer_params["observation_space"] = obs_space
        rollout_buffer = RolloutBuffer(**buffer_params)
        replay_buffer = ReplayBuffer(**buffer_params)
        assert rollout_buffer.observations.dtype == obs_dtype
        assert replay_buffer.observations.dtype == obs_dtype

    assert rollout_buffer.actions.dtype == action_space.dtype
    assert replay_buffer.actions.dtype == actual_replay_action_dtype
    # Check that sampled types are corrects
    rollout_buffer.full = True
    replay_buffer.full = True
    rollout_data = next(rollout_buffer.get(batch_size=64))
    buffer_data = replay_buffer.sample(batch_size=64)
    assert rollout_data.actions.numpy().dtype == actual_rollout_action_dtype
    assert buffer_data.actions.numpy().dtype == actual_replay_action_dtype
    if use_dict:
        assert buffer_data.observations["obs"].numpy().dtype == obs_dtype
        assert buffer_data.observations["obs_2"].numpy().dtype == np.uint8
        assert rollout_data.observations["obs"].numpy().dtype == obs_dtype
        assert rollout_data.observations["obs_2"].numpy().dtype == np.uint8
    else:
        assert buffer_data.observations.numpy().dtype == obs_dtype
        assert rollout_data.observations.numpy().dtype == obs_dtype


def test_custom_rollout_buffer():
    A2C("MlpPolicy", "Pendulum-v1", rollout_buffer_class=RolloutBuffer, rollout_buffer_kwargs=dict())

    with pytest.raises(TypeError, match=r"unexpected keyword argument 'wrong_keyword'"):
        A2C("MlpPolicy", "Pendulum-v1", rollout_buffer_class=RolloutBuffer, rollout_buffer_kwargs=dict(wrong_keyword=1))

    with pytest.raises(TypeError, match=r"got multiple values for keyword argument 'gamma'"):
        A2C("MlpPolicy", "Pendulum-v1", rollout_buffer_class=RolloutBuffer, rollout_buffer_kwargs=dict(gamma=1))

    with pytest.raises(AssertionError, match=r"DictRolloutBuffer must be used with Dict obs space only"):
        A2C("MlpPolicy", "Pendulum-v1", rollout_buffer_class=DictRolloutBuffer)


class TestDataclassSamples:
    """Tests for dataclass-based buffer samples (issue #2202)."""

    def test_subclassing_replay_buffer_samples(self):
        """Subclassing ReplayBufferSamples should work (main motivation for #2202)."""

        @dataclass
        class CustomReplayBufferSamples(ReplayBufferSamples):
            extra_field: th.Tensor = None

        sample = CustomReplayBufferSamples(
            observations=th.zeros(2, 3),
            actions=th.zeros(2, 1),
            next_observations=th.zeros(2, 3),
            dones=th.zeros(2, 1),
            rewards=th.zeros(2, 1),
            extra_field=th.ones(2, 1),
        )
        assert sample.extra_field is not None
        assert th.equal(sample.extra_field, th.ones(2, 1))

    def test_subclassing_rollout_buffer_samples(self):
        """Subclassing RolloutBufferSamples should work."""

        @dataclass
        class CustomRolloutBufferSamples(RolloutBufferSamples):
            auxiliary_loss: th.Tensor = None

        sample = CustomRolloutBufferSamples(
            observations=th.zeros(2, 3),
            actions=th.zeros(2, 1),
            old_values=th.zeros(2),
            old_log_prob=th.zeros(2),
            advantages=th.zeros(2),
            returns=th.zeros(2),
            auxiliary_loss=th.ones(2),
        )
        assert th.equal(sample.auxiliary_loss, th.ones(2))

    def test_positional_construction(self):
        """Positional arg construction should still work (backward compat)."""
        sample = ReplayBufferSamples(
            th.zeros(2, 3),
            th.zeros(2, 1),
            th.zeros(2, 3),
            th.zeros(2, 1),
            th.zeros(2, 1),
        )
        assert sample.discounts is None
        assert sample.observations.shape == (2, 3)

    def test_iteration_backward_compat(self):
        """Iteration over fields should work like NamedTuple."""
        sample = RolloutBufferSamples(
            observations=th.zeros(2, 3),
            actions=th.zeros(2, 1),
            old_values=th.zeros(2),
            old_log_prob=th.zeros(2),
            advantages=th.zeros(2),
            returns=th.zeros(2),
        )
        values = list(sample)
        assert len(values) == 6
        assert th.equal(values[0], th.zeros(2, 3))

    def test_dict_replay_buffer_samples_subclass(self):
        """DictReplayBufferSamples should be subclassable."""

        @dataclass
        class CustomDictReplayBufferSamples(DictReplayBufferSamples):
            priorities: th.Tensor = None

        sample = CustomDictReplayBufferSamples(
            observations={"obs": th.zeros(2, 3)},
            actions=th.zeros(2, 1),
            next_observations={"obs": th.zeros(2, 3)},
            dones=th.zeros(2, 1),
            rewards=th.zeros(2, 1),
            priorities=th.ones(2),
        )
        assert th.equal(sample.priorities, th.ones(2))

    def test_default_discounts(self):
        """ReplayBufferSamples.discounts should default to None."""
        sample = ReplayBufferSamples(
            observations=th.zeros(2, 3),
            actions=th.zeros(2, 1),
            next_observations=th.zeros(2, 3),
            dones=th.zeros(2, 1),
            rewards=th.zeros(2, 1),
        )
        assert sample.discounts is None

        sample_with_discounts = ReplayBufferSamples(
            observations=th.zeros(2, 3),
            actions=th.zeros(2, 1),
            next_observations=th.zeros(2, 3),
            dones=th.zeros(2, 1),
            rewards=th.zeros(2, 1),
            discounts=th.ones(2, 1),
        )
        assert th.equal(sample_with_discounts.discounts, th.ones(2, 1))
