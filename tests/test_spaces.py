import numpy as np
import pytest
import gym

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy


class DummyMultiDiscreteSpace(gym.Env):
    def __init__(self, nvec):
        super(DummyMultiDiscreteSpace, self).__init__()
        self.observation_space = gym.spaces.MultiDiscrete(nvec)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


class DummyMultiBinary(gym.Env):
    def __init__(self, n):
        super(DummyMultiBinary, self).__init__()
        self.observation_space = gym.spaces.MultiBinary(n)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


@pytest.mark.parametrize("model_class", [SAC, TD3])
@pytest.mark.parametrize("env", [DummyMultiDiscreteSpace([4, 3]), DummyMultiBinary(8)])
def test_identity_spaces(model_class, env):
    """
    Additional tests for SAC/TD3 to check observation space support
    for MultiDiscrete and MultiBinary.
    """
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    model = model_class("MlpPolicy", env, gamma=0.5, seed=1, policy_kwargs=dict(net_arch=[64]))
    model.learn(total_timesteps=500)

    evaluate_policy(model, env, n_eval_episodes=5)
