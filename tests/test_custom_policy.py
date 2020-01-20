import pytest

from torchy_baselines import PPO


@pytest.mark.parametrize('net_arch', [
    [12, dict(vf=[16], pi=[8])],
    [4],
    [4, 4],
    [12, dict(vf=[8, 4], pi=[8])],
    [12, dict(vf=[8], pi=[8, 4])],
    [12, dict(pi=[8])],
])
def test_flexible_mlp(net_arch):
    _ = PPO('MlpPolicy', 'CartPole-v1', policy_kwargs=dict(net_arch=net_arch), n_steps=100).learn(1000)
