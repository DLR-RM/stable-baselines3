import gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3, DDPG
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv
from timeit import default_timer

MODEL_LIST = [
    PPO,
    TD3,
    SAC,
    DQN,
    DDPG
]

@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_compile_speedup(model_class, env_id, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    if env_id == "CartPole-v1":
        if model_class in [SAC, TD3, DDPG]:
            return
    elif model_class in [DQN]:
        return

    # Test detection of different shapes by the predict method
    model_no_compile = model_class("MlpPolicy", env_id, device=device, torch_compile=False)
    model_compiled = model_class("MlpPolicy", env_id, device=device, torch_compile=True)

    start = default_timer()
    model_no_compile.learn(total_timesteps=1_000)
    duration_no_compile = default_timer() - start

    start = default_timer()
    model_compiled.learn(total_timesteps=1_000)
    duration_compile = default_timer() - start

    assert duration_compile < duration_no_compile
