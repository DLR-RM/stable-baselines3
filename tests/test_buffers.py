import gym
import numpy as np
import pytest
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class DummyEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        self.action_space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Box(1, 5, (1,))
        self._observations = [1, 2, 3, 4, 5]
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self):
        self._t = 0
        obs = self._observations[0]
        return obs

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = self._observations[index]
        done = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, done, {}


class DummyDictEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        self.action_space = spaces.Box(1, 5, (1,))
        space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Dict({"observation": space, "achieved_goal": space, "desired_goal": space})
        self._observations = [1, 2, 3, 4, 5]
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self):
        self._t = 0
        obs = {key: self._observations[0] for key in self.observation_space.spaces.keys()}
        return obs

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = {key: self._observations[index] for key in self.observation_space.spaces.keys()}
        done = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, done, {}


@pytest.mark.parametrize("replay_buffer_cls", [ReplayBuffer, DictReplayBuffer])
def test_replay_buffer_normalization(replay_buffer_cls):
    env = {ReplayBuffer: DummyEnv, DictReplayBuffer: DummyDictEnv}[replay_buffer_cls]
    env = make_vec_env(env)
    env = VecNormalize(env)

    buffer = replay_buffer_cls(100, env.observation_space, env.action_space)

    # Interract and store transitions
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
