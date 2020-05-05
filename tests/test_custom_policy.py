import pytest
import torch as th

from stable_baselines3 import A2C, PPO, SAC, TD3


@pytest.mark.parametrize('net_arch', [
    [12, dict(vf=[16], pi=[8])],
    [4],
    [],
    [4, 4],
    [12, dict(vf=[8, 4], pi=[8])],
    [12, dict(vf=[8], pi=[8, 4])],
    [12, dict(pi=[8])],
])
@pytest.mark.parametrize('model_class', [A2C, PPO])
def test_flexible_mlp(model_class, net_arch):
    _ = model_class('MlpPolicy', 'CartPole-v1', policy_kwargs=dict(net_arch=net_arch), n_steps=100).learn(1000)


@pytest.mark.parametrize('net_arch', [
    [4],
    [4, 4],
])
@pytest.mark.parametrize('model_class', [SAC, TD3])
def test_custom_offpolicy(model_class, net_arch):
    _ = model_class('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=net_arch)).learn(1000)


@pytest.mark.parametrize('model_class', [A2C, PPO, SAC, TD3])
@pytest.mark.parametrize('optimizer_kwargs', [None, dict(weight_decay=0.0)])
def test_custom_optimizer(model_class, optimizer_kwargs):
    policy_kwargs = dict(optimizer_class=th.optim.AdamW, optimizer_kwargs=optimizer_kwargs, net_arch=[32])
    _ = model_class('MlpPolicy', 'Pendulum-v0', policy_kwargs=policy_kwargs).learn(1000)
