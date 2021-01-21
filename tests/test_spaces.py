import gym
import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
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


@pytest.mark.parametrize("model_class", [SAC, TD3, DQN])
@pytest.mark.parametrize("env", [DummyMultiDiscreteSpace([4, 3]), DummyMultiBinary(8)])
def test_identity_spaces(model_class, env):
    """
    Additional tests for DQ/SAC/TD3 to check observation space support
    for MultiDiscrete and MultiBinary.
    """
    # DQN only support discrete actions
    if model_class == DQN:
        env.action_space = gym.spaces.Discrete(4)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    model = model_class("MlpPolicy", env, gamma=0.5, seed=1, policy_kwargs=dict(net_arch=[64]))
    model.learn(total_timesteps=500)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [A2C, DDPG, DQN, PPO, SAC, TD3])
@pytest.mark.parametrize("env", ["Pendulum-v0", "CartPole-v1"])
def test_action_spaces(model_class, env):
    if model_class in [SAC, DDPG, TD3]:
        supported_action_space = env == "Pendulum-v0"
    elif model_class == DQN:
        supported_action_space = env == "CartPole-v1"
    elif model_class in [A2C, PPO]:
        supported_action_space = True

    if supported_action_space:
        model_class("MlpPolicy", env)
    else:
        with pytest.raises(AssertionError):
            model_class("MlpPolicy", env)


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env", ["Taxi-v3"])
def test_discrete_obs_space(model_class, env):
    env = make_vec_env(env, n_envs=2, seed=0)
    model_class("MlpPolicy", env, n_steps=256).learn(500)
