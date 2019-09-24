import os

from torchy_baselines import TD3, CEMRL, PPO, SAC


def test_td3():
    model = TD3('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.pth")


def test_cemrl():
    model = CEMRL('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[16]), pop_size=2, n_grad=1,
                  learning_starts=100, verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save")
    model.load("test_save")
    os.remove("test_save.pth")


def test_ppo():
    model = PPO('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)
    # model.save("test_save")
    # model.load("test_save")
    # os.remove("test_save.pth")

def test_sac():
    model = SAC('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True, ent_coef='auto')
    model.learn(total_timesteps=1000, eval_freq=500)
