import pytest
import torch as th

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


@pytest.mark.parametrize(
    "net_arch",
    [
        [12, dict(vf=[16], pi=[8])],
        [4],
        [],
        [4, 4],
        [12, dict(vf=[8, 4], pi=[8])],
        [12, dict(vf=[8], pi=[8, 4])],
        [12, dict(pi=[8])],
    ],
)
@pytest.mark.parametrize("model_class", [A2C, PPO])
def test_flexible_mlp(model_class, net_arch):
    _ = model_class("MlpPolicy", "CartPole-v1", policy_kwargs=dict(net_arch=net_arch), n_steps=64).learn(300)


@pytest.mark.parametrize("net_arch", [[], [4], [4, 4], dict(qf=[8], pi=[8, 4])])
@pytest.mark.parametrize("model_class", [SAC, TD3])
def test_custom_offpolicy(model_class, net_arch):
    _ = model_class("MlpPolicy", "Pendulum-v1", policy_kwargs=dict(net_arch=net_arch), learning_starts=100).learn(300)


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3])
@pytest.mark.parametrize("optimizer_kwargs", [None, dict(weight_decay=0.0)])
def test_custom_optimizer(model_class, optimizer_kwargs):
    kwargs = {}
    if model_class in {DQN, SAC, TD3}:
        kwargs = dict(learning_starts=100)
    elif model_class in {A2C, PPO}:
        kwargs = dict(n_steps=64)

    policy_kwargs = dict(optimizer_class=th.optim.AdamW, optimizer_kwargs=optimizer_kwargs, net_arch=[32])
    _ = model_class("MlpPolicy", "Pendulum-v1", policy_kwargs=policy_kwargs, **kwargs).learn(300)


def test_tf_like_rmsprop_optimizer():
    policy_kwargs = dict(optimizer_class=RMSpropTFLike, net_arch=[32])
    _ = A2C("MlpPolicy", "Pendulum-v1", policy_kwargs=policy_kwargs).learn(500)


def test_dqn_custom_policy():
    policy_kwargs = dict(optimizer_class=RMSpropTFLike, net_arch=[32])
    _ = DQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, learning_starts=100).learn(300)
