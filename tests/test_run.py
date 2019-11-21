import os

import pytest
import numpy as np

from torchy_baselines import A2C, CEMRL, PPO, SAC, TD3
from torchy_baselines.common.noise import NormalActionNoise

action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


def test_td3():
    model = TD3('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True, action_noise=action_noise)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.pth")


def test_cemrl():
    model = CEMRL('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[16]), pop_size=2, n_grad=1,
                  learning_starts=100, verbose=1, create_eval_env=True, action_noise=action_noise)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.pth")


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env_id", ['CartPole-v1', 'Pendulum-v0'])
def test_onpolicy(model_class, env_id):
    model = model_class('MlpPolicy', env_id, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    #os.remove("test_save.pth")


def test_sac():
    model = SAC('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True, ent_coef='auto',
                action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)))
    model.learn(total_timesteps=1000, eval_freq=500)
