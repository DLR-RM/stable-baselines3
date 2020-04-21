import pytest

from torchy_baselines import A2C, PPO, SAC, TD3
from torchy_baselines.common.identity_env import FakeImageEnv


@pytest.mark.parametrize('model_class', [A2C, PPO, SAC])
def test_cnn(model_class):
    # Fake grayscale with frameskip
    # Atari after preprocessing: 84x84x1, here we are using lower resolution
    # to check that the network handle it automatically
    env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=1,
                       discrete = model_class not in {SAC, TD3})
    if model_class in {A2C, PPO}:
        kwargs = dict(n_steps=100)
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features
        kwargs = dict(buffer_size=500, policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=40)))
    _ = model_class('CnnPolicy', env, **kwargs).learn(500)
