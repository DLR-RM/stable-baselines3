import pytest

from torchy_baselines import A2C, PPO, SAC, TD3
from torchy_baselines.common.identity_env import FakeImageEnv


@pytest.mark.parametrize('model_class', [A2C, PPO, SAC])
def test_cnn(model_class):
    # Fake grayscale with frameskip
    env = FakeImageEnv(screen_height=84, screen_width=84, n_channels=1,
                       discrete = model_class not in {SAC, TD3})
    if model_class in {A2C, PPO}:
        kwargs = dict(n_steps=100)
    else:
        kwargs = dict(buffer_size=500)
    _ = model_class('CnnPolicy', env, **kwargs).learn(500)
