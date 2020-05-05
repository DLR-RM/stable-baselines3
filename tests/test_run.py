import numpy as np
import pytest

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


@pytest.mark.parametrize('action_noise', [normal_action_noise, OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1 * np.ones(1))])
def test_td3(action_noise):
    model = TD3('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True, action_noise=action_noise)
    model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env_id", ['CartPole-v1', 'Pendulum-v0'])
def test_onpolicy(model_class, env_id):
    model = model_class('MlpPolicy', env_id, seed=0, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("ent_coef", ['auto', 0.01])
def test_sac(ent_coef):
    model = SAC('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True, ent_coef=ent_coef,
                action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)))
    model.learn(total_timesteps=1000, eval_freq=500)
