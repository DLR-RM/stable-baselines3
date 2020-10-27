import os
from copy import deepcopy

import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.identity_env import FakeImageEnv
from stable_baselines3.common.utils import zip_strict


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3, DQN])
def test_cnn(tmp_path, model_class):
    SAVE_NAME = "cnn_model.zip"
    # Fake grayscale with frameskip
    # Atari after preprocessing: 84x84x1, here we are using lower resolution
    # to check that the network handle it automatically
    env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=1, discrete=model_class not in {SAC, TD3})
    if model_class in {A2C, PPO}:
        kwargs = dict(n_steps=100)
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features
        kwargs = dict(buffer_size=250, policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32)))
    model = model_class("CnnPolicy", env, **kwargs).learn(250)

    obs = env.reset()

    action, _ = model.predict(obs, deterministic=True)

    model.save(tmp_path / SAVE_NAME)
    del model

    model = model_class.load(tmp_path / SAVE_NAME)

    # Check that the prediction is the same
    assert np.allclose(action, model.predict(obs, deterministic=True)[0])

    os.remove(str(tmp_path / SAVE_NAME))


def patch_dqn_names_(model):
    # Small hack to make the test work with DQN
    if isinstance(model, DQN):
        model.critic = model.q_net
        model.critic_target = model.q_net_target


def params_should_match(params, other_params):
    for param, other_param in zip_strict(params, other_params):
        assert th.allclose(param, other_param)


def params_should_differ(params, other_params):
    for param, other_param in zip_strict(params, other_params):
        assert not th.allclose(param, other_param)


def check_td3_feature_extractor_match(model):
    for (key, actor_param), critic_param in zip(model.actor_target.named_parameters(), model.critic_target.parameters()):
        if "features_extractor" in key:
            assert th.allclose(actor_param, critic_param), key


def check_td3_feature_extractor_differ(model):
    for (key, actor_param), critic_param in zip(model.actor_target.named_parameters(), model.critic_target.parameters()):
        if "features_extractor" in key:
            assert not th.allclose(actor_param, critic_param), key


@pytest.mark.parametrize("model_class", [SAC, TD3, DQN])
@pytest.mark.parametrize("share_features_extractor", [True, False])
def test_features_extractor_target_net(model_class, share_features_extractor):
    if model_class == DQN and share_features_extractor:
        pytest.skip()

    env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=1, discrete=model_class not in {SAC, TD3})
    # Avoid memory error when using replay buffer
    # Reduce the size of the features
    kwargs = dict(buffer_size=250, learning_starts=100, policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32)))
    if model_class != DQN:
        kwargs["policy_kwargs"]["share_features_extractor"] = share_features_extractor

    model = model_class("CnnPolicy", env, seed=0, **kwargs)

    patch_dqn_names_(model)

    if share_features_extractor:
        # Check that the objects are the same and not just copied
        assert id(model.policy.actor.features_extractor) == id(model.policy.critic.features_extractor)
        if model_class == TD3:
            assert id(model.policy.actor_target.features_extractor) == id(model.policy.critic_target.features_extractor)
        # Actor and critic feature extractor should be the same
        td3_features_extractor_check = check_td3_feature_extractor_match
    else:
        # Actor and critic feature extractor should differ same
        td3_features_extractor_check = check_td3_feature_extractor_differ
        # Check that the object differ
        if model_class != DQN:
            assert id(model.policy.actor.features_extractor) != id(model.policy.critic.features_extractor)

        if model_class == TD3:
            assert id(model.policy.actor_target.features_extractor) != id(model.policy.critic_target.features_extractor)

    # Critic and target should be equal at the begginning of training
    params_should_match(model.critic.parameters(), model.critic_target.parameters())

    # TD3 has also a target actor net
    if model_class == TD3:
        params_should_match(model.actor.parameters(), model.actor_target.parameters())

    model.learn(200)

    # Critic and target should differ
    params_should_differ(model.critic.parameters(), model.critic_target.parameters())

    if model_class == TD3:
        params_should_differ(model.actor.parameters(), model.actor_target.parameters())
        td3_features_extractor_check(model)

    # Re-initialize and collect some random data (without doing gradient steps,
    # since 10 < learning_starts = 100)
    model = model_class("CnnPolicy", env, seed=0, **kwargs).learn(10)

    patch_dqn_names_(model)

    original_param = deepcopy(list(model.critic.parameters()))
    original_target_param = deepcopy(list(model.critic_target.parameters()))
    if model_class == TD3:
        original_actor_target_param = deepcopy(list(model.actor_target.parameters()))

    # Deactivate copy to target
    model.tau = 0.0
    model.train(gradient_steps=1)

    # Target should be the same
    params_should_match(original_target_param, model.critic_target.parameters())

    if model_class == TD3:
        params_should_match(original_actor_target_param, model.actor_target.parameters())
        td3_features_extractor_check(model)

    # not the same for critic net (updated by gradient descent)
    params_should_differ(original_param, model.critic.parameters())

    # Update the reference as it should not change in the next step
    original_param = deepcopy(list(model.critic.parameters()))

    if model_class == TD3:
        original_actor_param = deepcopy(list(model.actor.parameters()))

    # Deactivate learning rate
    model.lr_schedule = lambda _: 0.0
    # Re-activate polyak update
    model.tau = 0.01
    # Special case for DQN: target net is updated in the `collect_rollout()`
    # not the `train()` method
    if model_class == DQN:
        model.target_update_interval = 1
        model._on_step()

    model.train(gradient_steps=1)

    # Target should have changed now (due to polyak update)
    params_should_differ(original_target_param, model.critic_target.parameters())

    # Critic should be the same
    params_should_match(original_param, model.critic.parameters())

    if model_class == TD3:
        params_should_differ(original_actor_target_param, model.actor_target.parameters())

        params_should_match(original_actor_param, model.actor.parameters())

        td3_features_extractor_check(model)
