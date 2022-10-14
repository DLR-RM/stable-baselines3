import gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.envs import IdentityEnv
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv

MODEL_LIST = [
    PPO,
    A2C,
    TD3,
    SAC,
    DQN,
]


class SubClassedBox(gym.spaces.Box):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CustomSubClassedSpaceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = SubClassedBox(-1, 1, shape=(2,), dtype=np.float32)
        self.action_space = SubClassedBox(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, np.random.rand() > 0.5, {}


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_auto_wrap(model_class):
    """Test auto wrapping of env into a VecEnv."""
    # Use different environment for DQN
    if model_class is DQN:
        env_name = "CartPole-v0"
    else:
        env_name = "Pendulum-v1"
    env = gym.make(env_name)
    model = model_class("MlpPolicy", env)
    model.learn(100)


@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_predict(model_class, env_id, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    if env_id == "CartPole-v1":
        if model_class in [SAC, TD3]:
            return
    elif model_class in [DQN]:
        return

    # Test detection of different shapes by the predict method
    model = model_class("MlpPolicy", env_id, device=device)
    # Check that the policy is on the right device
    assert get_device(device).type == model.policy.device.type

    env = gym.make(env_id)
    vec_env = DummyVecEnv([lambda: gym.make(env_id), lambda: gym.make(env_id)])

    obs = env.reset()
    action, _ = model.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == env.action_space.shape
    assert env.action_space.contains(action)

    vec_env_obs = vec_env.reset()
    action, _ = model.predict(vec_env_obs)
    assert isinstance(action, np.ndarray)
    assert action.shape[0] == vec_env_obs.shape[0]

    # Special case for DQN to check the epsilon greedy exploration
    if model_class == DQN:
        model.exploration_rate = 1.0
        action, _ = model.predict(obs, deterministic=False)
        assert action.shape == env.action_space.shape
        assert env.action_space.contains(action)

        action, _ = model.predict(vec_env_obs, deterministic=False)
        assert action.shape[0] == vec_env_obs.shape[0]


def test_dqn_epsilon_greedy():
    env = IdentityEnv(2)
    model = DQN("MlpPolicy", env)
    model.exploration_rate = 1.0
    obs = env.reset()
    # is vectorized should not crash with discrete obs
    action, _ = model.predict(obs, deterministic=False)
    assert env.action_space.contains(action)


@pytest.mark.parametrize("model_class", [A2C, SAC, PPO, TD3])
def test_subclassed_space_env(model_class):
    env = CustomSubClassedSpaceEnv()
    model = model_class("MlpPolicy", env, policy_kwargs=dict(net_arch=[32]))
    model.learn(300)
    obs = env.reset()
    env.step(model.predict(obs))
