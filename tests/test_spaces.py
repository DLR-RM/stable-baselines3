from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.spaces.space import Space

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

BOX_SPACE_FLOAT64 = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
BOX_SPACE_FLOAT32 = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


@dataclass
class DummyEnv(gym.Env):
    observation_space: Space
    action_space: Space

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            super().reset(seed=seed)
        return self.observation_space.sample(), {}


class DummyMultidimensionalAction(DummyEnv):
    def __init__(self):
        super().__init__(
            BOX_SPACE_FLOAT32,
            spaces.Box(low=-1, high=1, shape=(2, 2), dtype=np.float32),
        )


class DummyMultiBinary(DummyEnv):
    def __init__(self, n):
        super().__init__(
            spaces.MultiBinary(n),
            BOX_SPACE_FLOAT32,
        )


class DummyMultiDiscreteSpace(DummyEnv):
    def __init__(self, nvec):
        super().__init__(
            spaces.MultiDiscrete(nvec),
            BOX_SPACE_FLOAT32,
        )


@pytest.mark.parametrize(
    "env",
    [
        DummyMultiDiscreteSpace([4, 3]),
        DummyMultiBinary(8),
        DummyMultiBinary((3, 2)),
        DummyMultidimensionalAction(),
    ],
)
def test_env(env):
    # Check the env used for testing
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("model_class", [SAC, TD3, DQN])
@pytest.mark.parametrize("env", [DummyMultiDiscreteSpace([4, 3]), DummyMultiBinary(8), DummyMultiBinary((3, 2))])
def test_identity_spaces(model_class, env):
    """
    Additional tests for DQ/SAC/TD3 to check observation space support
    for MultiDiscrete and MultiBinary.
    """
    # DQN only support discrete actions
    if model_class == DQN:
        env.action_space = spaces.Discrete(4)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    model = model_class("MlpPolicy", env, gamma=0.5, seed=1, policy_kwargs=dict(net_arch=[64]))
    model.learn(total_timesteps=500)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [A2C, DDPG, DQN, PPO, SAC, TD3])
@pytest.mark.parametrize("env", ["Pendulum-v1", "CartPole-v1", DummyMultidimensionalAction()])
def test_action_spaces(model_class, env):
    kwargs = {}
    if model_class in [SAC, DDPG, TD3]:
        supported_action_space = env == "Pendulum-v1" or isinstance(env, DummyMultidimensionalAction)
        kwargs["learning_starts"] = 2
        kwargs["train_freq"] = 32
    elif model_class == DQN:
        supported_action_space = env == "CartPole-v1"
    elif model_class in [A2C, PPO]:
        supported_action_space = True
        kwargs["n_steps"] = 64

    if supported_action_space:
        model = model_class("MlpPolicy", env, **kwargs)
        if isinstance(env, DummyMultidimensionalAction):
            model.learn(64)
    else:
        with pytest.raises(AssertionError):
            model_class("MlpPolicy", env)


def test_sde_multi_dim():
    SAC(
        "MlpPolicy",
        DummyMultidimensionalAction(),
        learning_starts=10,
        use_sde=True,
        sde_sample_freq=2,
        use_sde_at_warmup=True,
    ).learn(20)


@pytest.mark.parametrize("model_class", [A2C, PPO, DQN])
@pytest.mark.parametrize("env", ["Taxi-v3"])
def test_discrete_obs_space(model_class, env):
    env = make_vec_env(env, n_envs=2, seed=0)
    kwargs = {}
    if model_class == DQN:
        kwargs = dict(buffer_size=1000, learning_starts=100)
    else:
        kwargs = dict(n_steps=256)
    model_class("MlpPolicy", env, **kwargs).learn(256)


@pytest.mark.parametrize("model_class", [SAC, TD3, PPO, DDPG, A2C])
@pytest.mark.parametrize(
    "obs_space",
    [
        BOX_SPACE_FLOAT32,
        BOX_SPACE_FLOAT64,
        spaces.Dict({"a": BOX_SPACE_FLOAT32, "b": BOX_SPACE_FLOAT32}),
        spaces.Dict({"a": BOX_SPACE_FLOAT32, "b": BOX_SPACE_FLOAT64}),
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        BOX_SPACE_FLOAT32,
        BOX_SPACE_FLOAT64,
    ],
)
def test_float64_action_space(model_class, obs_space, action_space):
    env = DummyEnv(obs_space, action_space)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    if isinstance(env.observation_space, spaces.Dict):
        policy = "MultiInputPolicy"
    else:
        policy = "MlpPolicy"

    if model_class in [PPO, A2C]:
        kwargs = dict(n_steps=64, policy_kwargs=dict(net_arch=[12]))
    else:
        kwargs = dict(learning_starts=60, policy_kwargs=dict(net_arch=[12]))

    model = model_class(policy, env, **kwargs)
    model.learn(64)
    initial_obs, _ = env.reset()
    action, _ = model.predict(initial_obs, deterministic=False)
    assert action.dtype == env.action_space.dtype


def test_multidim_binary_not_supported():
    env = DummyEnv(BOX_SPACE_FLOAT32, spaces.MultiBinary([2, 3]))
    with pytest.raises(AssertionError, match=r"Multi-dimensional MultiBinary\(.*\) action space is not supported"):
        A2C("MlpPolicy", env)
