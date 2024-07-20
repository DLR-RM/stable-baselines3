import pytest
import torch as th
import torch.nn as nn

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.torch_layers import create_mlp


@pytest.mark.parametrize(
    "net_arch",
    [
        [],
        [4],
        [4, 4],
        dict(vf=[16], pi=[8]),
        dict(vf=[8, 4], pi=[8]),
        dict(vf=[8], pi=[8, 4]),
        dict(pi=[8]),
        # Old format, emits a warning
        [dict(vf=[8])],
        [dict(vf=[8], pi=[4])],
    ],
)
@pytest.mark.parametrize("model_class", [A2C, PPO])
def test_flexible_mlp(model_class, net_arch):
    if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
        with pytest.warns(UserWarning):
            _ = model_class("MlpPolicy", "CartPole-v1", policy_kwargs=dict(net_arch=net_arch), n_steps=64).learn(300)
    else:
        _ = model_class("MlpPolicy", "CartPole-v1", policy_kwargs=dict(net_arch=net_arch), n_steps=64).learn(300)


@pytest.mark.parametrize("net_arch", [[], [4], [4, 4], dict(qf=[8], pi=[8, 4])])
@pytest.mark.parametrize("model_class", [SAC, TD3])
def test_custom_offpolicy(model_class, net_arch):
    _ = model_class("MlpPolicy", "Pendulum-v1", policy_kwargs=dict(net_arch=net_arch), learning_starts=100).learn(300)


@pytest.mark.parametrize("model_class", [A2C, DQN, PPO, SAC, TD3])
@pytest.mark.parametrize("optimizer_kwargs", [None, dict(weight_decay=0.0)])
def test_custom_optimizer(model_class, optimizer_kwargs):
    # Use different environment for DQN
    if model_class is DQN:
        env_id = "CartPole-v1"
    else:
        env_id = "Pendulum-v1"

    kwargs = {}
    if model_class in {DQN, SAC, TD3}:
        kwargs = dict(learning_starts=100)
    elif model_class in {A2C, PPO}:
        kwargs = dict(n_steps=64)

    policy_kwargs = dict(optimizer_class=th.optim.AdamW, optimizer_kwargs=optimizer_kwargs, net_arch=[32])
    _ = model_class("MlpPolicy", env_id, policy_kwargs=policy_kwargs, **kwargs).learn(300)


def test_tf_like_rmsprop_optimizer():
    policy_kwargs = dict(optimizer_class=RMSpropTFLike, net_arch=[32])
    _ = A2C("MlpPolicy", "Pendulum-v1", policy_kwargs=policy_kwargs).learn(500)


def test_dqn_custom_policy():
    policy_kwargs = dict(optimizer_class=RMSpropTFLike, net_arch=[32])
    _ = DQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, learning_starts=100).learn(300)


def test_create_mlp():
    net = create_mlp(4, 2, net_arch=[16, 8], squash_output=True)
    # We cannot compare the network directly because the modules have different ids
    # assert net == [nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2),
    # nn.Tanh()]
    assert len(net) == 6
    assert isinstance(net[0], nn.Linear)
    assert net[0].in_features == 4
    assert net[0].out_features == 16
    assert isinstance(net[1], nn.ReLU)
    assert isinstance(net[2], nn.Linear)
    assert isinstance(net[4], nn.Linear)
    assert net[4].in_features == 8
    assert net[4].out_features == 2
    assert isinstance(net[5], nn.Tanh)

    # Linear network
    net = create_mlp(4, -1, net_arch=[])
    assert net == []

    # No output layer, with custom activation function
    net = create_mlp(6, -1, net_arch=[8], activation_fn=nn.Tanh)
    # assert net == [nn.Linear(6, 8), nn.Tanh()]
    assert len(net) == 2
    assert isinstance(net[0], nn.Linear)
    assert net[0].in_features == 6
    assert net[0].out_features == 8
    assert isinstance(net[1], nn.Tanh)

    # Using pre-linear and post-linear modules
    pre_linear = [nn.BatchNorm1d]
    post_linear = [nn.LayerNorm]
    net = create_mlp(6, 2, net_arch=[8, 12], pre_linear_modules=pre_linear, post_linear_modules=post_linear)
    # assert net == [nn.BatchNorm1d(6), nn.Linear(6, 8), nn.LayerNorm(8), nn.ReLU()
    #  nn.BatchNorm1d(6), nn.Linear(8, 12), nn.LayerNorm(12), nn.ReLU(),
    # nn.BatchNorm1d(12),  nn.Linear(12, 2)] # Last layer does not have post_linear
    assert len(net) == 10
    assert isinstance(net[0], nn.BatchNorm1d)
    assert net[0].num_features == 6
    assert isinstance(net[1], nn.Linear)
    assert isinstance(net[2], nn.LayerNorm)
    assert isinstance(net[3], nn.ReLU)
    assert isinstance(net[4], nn.BatchNorm1d)
    assert isinstance(net[5], nn.Linear)
    assert net[5].in_features == 8
    assert net[5].out_features == 12
    assert isinstance(net[6], nn.LayerNorm)
    assert isinstance(net[7], nn.ReLU)
    assert isinstance(net[8], nn.BatchNorm1d)
    assert isinstance(net[-1], nn.Linear)
    assert net[-1].in_features == 12
    assert net[-1].out_features == 2
