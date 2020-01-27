import os

import pytest
import numpy as np

from torchy_baselines import A2C, CEMRL, PPO, SAC, TD3
from torchy_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


@pytest.mark.parametrize('action_noise', [action_noise, OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1 * np.ones(1))])
def test_td3(action_noise):
    model = TD3('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True, action_noise=action_noise)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.zip")


def test_cemrl():
    model = CEMRL('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[16]), pop_size=2, n_grad=1,
                  learning_starts=100, verbose=1, create_eval_env=True, action_noise=action_noise)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.zip")


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env_id", ['CartPole-v1', 'Pendulum-v0'])
def test_onpolicy(model_class, env_id):
    model = model_class('MlpPolicy', env_id, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.zip")


@pytest.mark.parametrize("ent_coef", ['auto', 0.01])
def test_sac(ent_coef):
    model = SAC('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True, ent_coef=ent_coef,
                action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)))
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.zip")
