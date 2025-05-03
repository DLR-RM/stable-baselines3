# tests/test_actor_critic_cnn_policy.py
import gymnasium as gym
import numpy as np
import pytest
import torch as th

from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN, create_mlp


@pytest.mark.parametrize(
    "action_space",
    [
        gym.spaces.Discrete(3),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    ],
)
def test_actor_critic_cnn_policy_forward(action_space):
    # 1) Build a dummy “image” observation space
    obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(3, 64, 64),
        dtype=np.uint8,
    )

    # 2) Create a constant learning-rate schedule
    lr_schedule = lambda _: 1e-3

    # 3) Instantiate the policy
    policy = ActorCriticCnnPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
    )
    policy.to("cpu")
    policy.eval()

    # 4) Make a batch of 5 random observations
    #    (normalized to [0,1], since SB3 will divide uint8 by 255 internally)
    obs = th.rand(5, *obs_space.shape)

    # 5) Forward pass: returns (actions, values, log_prob)
    with th.no_grad():
        actions, values, log_prob = policy.forward(obs)

    # 6) Check output shapes
    #   - for Discrete: actions is (batch,) of ints
    #   - for Box: actions is (batch, *action_shape)
    if isinstance(action_space, gym.spaces.Discrete):
        assert actions.shape == (5,)
    else:
        assert actions.shape == (5, *action_space.shape)

    # value function: one value per batch
    assert values.shape == (5, 1)

    # log probability: one scalar per batch
    assert log_prob.shape == (5,)

    # 7) Also test that `.evaluate_actions` works
    #    (returns value, log_prob, entropies)
    value2, log_prob2, entropy = policy.evaluate_actions(obs, actions)
    assert value2.shape == (5, 1)
    assert log_prob2.shape == (5,)
    assert entropy.shape == (5,)


def test_nature_cnn_with_batchnorm(tmp_path):
    import gymnasium as gym
    from gymnasium import spaces

    # create a fake image space: 3×64×64 uint8
    obs_space = spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    cnn = NatureCNN(obs_space, features_dim=64, use_batch_norm=True)
    # We should see at least one BatchNorm2d in cnn.cnn
    assert any(isinstance(m, th.nn.BatchNorm2d) for m in cnn.cnn), "No BatchNorm2d found"
    # Forward a dummy batch of one sample
    obs = th.as_tensor(obs_space.sample()[None]).float()
    out = cnn(obs)
    assert out.shape[-1] == 64


def test_create_mlp_with_batchnorm():
    layers = create_mlp(32, 4, [16, 8], use_batch_norm=True)
    # Check that first element is BatchNorm1d
    assert isinstance(layers[0], th.nn.BatchNorm1d)
    mlp = th.nn.Sequential(*layers)
    x = th.randn(5, 32)
    y = mlp(x)
    assert y.shape == (5, 4)


if __name__ == "__main__":
    pytest.main([__file__])
