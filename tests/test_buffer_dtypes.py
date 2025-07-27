import numpy as np
import torch as th
import pytest

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.env_util import make_vec_env


@pytest.mark.parametrize("model_class", [PPO, A2C])
def test_run_rollout(model_class):
    env_id = "CartPole-v1"
    env = make_vec_env(env_id, n_envs=2)
    model_kwargs = dict(batch_size=100) if model_class == PPO else dict()

    model = model_class(
        "MlpPolicy",
        env,
        n_steps=50,
        rollout_buffer_kwargs={"dtypes": dict(observations=np.float16)},
        device="cpu",
        **model_kwargs
    )

    assert isinstance(model.rollout_buffer, RolloutBuffer)
    assert model.rollout_buffer.observations.dtype == np.float16

    model.learn(total_timesteps=150)
    model.rollout_buffer.full = True
    assert next(model.rollout_buffer.get(batch_size=64)).observations.dtype == th.float32
