import os

import numpy as np
import pytest

from stable_baselines3 import A2C, PPO, SAC, TD3, DQN
from stable_baselines3.common.identity_env import FakeImageEnv


@pytest.mark.parametrize('model_class', [A2C, PPO, SAC, TD3, DQN])
def test_cnn(tmp_path, model_class):
    SAVE_NAME = 'cnn_model.zip'
    # Fake grayscale with frameskip
    # Atari after preprocessing: 84x84x1, here we are using lower resolution
    # to check that the network handle it automatically
    env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=1,
                       discrete=model_class not in {SAC, TD3})
    if model_class in {A2C, PPO}:
        kwargs = dict(n_steps=100)
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features
        kwargs = dict(buffer_size=250,
                      policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32)))
    model = model_class('CnnPolicy', env, **kwargs).learn(250)

    obs = env.reset()

    action, _ = model.predict(obs, deterministic=True)

    model.save(tmp_path / SAVE_NAME)
    del model

    model = model_class.load(tmp_path / SAVE_NAME)

    # Check that the prediction is the same
    assert np.allclose(action, model.predict(obs, deterministic=True)[0])

    os.remove(str(tmp_path / SAVE_NAME))
