import os

import gym

from torchy_baselines import TD3, CEMRL, PPO

def test_td3():
    model = TD3('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                start_timesteps=100, verbose=1, create_eval_env=True)
    model.learn(total_timesteps=20000, eval_freq=1000)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.pth")

def test_cemrl():
    model = CEMRL('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[16]), pop_size=2, n_grad=1,
                 start_timesteps=100, verbose=1, create_eval_env=True)
    model.learn(total_timesteps=20000, eval_freq=1000)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.pth")

def test_ppo():
    model = PPO('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)
    # model.save("test_save")
    # model.load("test_save")
    # os.remove("test_save.pth")
