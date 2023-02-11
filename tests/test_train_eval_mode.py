from typing import Union

import gymnasium as gym
import numpy as np
import pytest
import torch as th
import torch.nn as nn

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

MODEL_LIST = [
    PPO,
    A2C,
    TD3,
    SAC,
    DQN,
]


class FlattenBatchNormDropoutExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input and applies batch normalization and dropout.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(
            observation_space,
            get_flattened_obs_dim(observation_space),
        )
        self.flatten = nn.Flatten()
        self.batch_norm = nn.BatchNorm1d(self._features_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        result = self.flatten(observations)
        result = self.batch_norm(result)
        result = self.dropout(result)
        return result


def clone_batch_norm_stats(batch_norm: nn.BatchNorm1d) -> (th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the given batch norm layer.

    :param batch_norm:
    :return: the bias and running mean
    """
    return batch_norm.bias.clone(), batch_norm.running_mean.clone()


def clone_dqn_batch_norm_stats(model: DQN) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the Q-network and target network.

    :param model:
    :return: the bias and running mean from the Q-network and target network
    """
    q_net_batch_norm = model.policy.q_net.features_extractor.batch_norm
    q_net_bias, q_net_running_mean = clone_batch_norm_stats(q_net_batch_norm)

    q_net_target_batch_norm = model.policy.q_net_target.features_extractor.batch_norm
    q_net_target_bias, q_net_target_running_mean = clone_batch_norm_stats(q_net_target_batch_norm)

    return q_net_bias, q_net_running_mean, q_net_target_bias, q_net_target_running_mean


def clone_td3_batch_norm_stats(
    model: TD3,
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the actor and critic networks and actor-target and critic-target networks.

    :param model:
    :return: the bias and running mean from the actor and critic networks and actor-target and critic-target networks
    """
    actor_batch_norm = model.actor.features_extractor.batch_norm
    actor_bias, actor_running_mean = clone_batch_norm_stats(actor_batch_norm)

    critic_batch_norm = model.critic.features_extractor.batch_norm
    critic_bias, critic_running_mean = clone_batch_norm_stats(critic_batch_norm)

    actor_target_batch_norm = model.actor_target.features_extractor.batch_norm
    actor_target_bias, actor_target_running_mean = clone_batch_norm_stats(actor_target_batch_norm)

    critic_target_batch_norm = model.critic_target.features_extractor.batch_norm
    critic_target_bias, critic_target_running_mean = clone_batch_norm_stats(critic_target_batch_norm)

    return (
        actor_bias,
        actor_running_mean,
        critic_bias,
        critic_running_mean,
        actor_target_bias,
        actor_target_running_mean,
        critic_target_bias,
        critic_target_running_mean,
    )


def clone_sac_batch_norm_stats(
    model: SAC,
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the actor and critic networks and critic-target networks.

    :param model:
    :return: the bias and running mean from the actor and critic networks and critic-target networks
    """
    actor_batch_norm = model.actor.features_extractor.batch_norm
    actor_bias, actor_running_mean = clone_batch_norm_stats(actor_batch_norm)

    critic_batch_norm = model.critic.features_extractor.batch_norm
    critic_bias, critic_running_mean = clone_batch_norm_stats(critic_batch_norm)

    critic_target_batch_norm = model.critic_target.features_extractor.batch_norm
    critic_target_bias, critic_target_running_mean = clone_batch_norm_stats(critic_target_batch_norm)

    return (actor_bias, actor_running_mean, critic_bias, critic_running_mean, critic_target_bias, critic_target_running_mean)


def clone_on_policy_batch_norm(model: Union[A2C, PPO]) -> (th.Tensor, th.Tensor):
    return clone_batch_norm_stats(model.policy.features_extractor.batch_norm)


CLONE_HELPERS = {
    A2C: clone_on_policy_batch_norm,
    DQN: clone_dqn_batch_norm_stats,
    SAC: clone_sac_batch_norm_stats,
    TD3: clone_td3_batch_norm_stats,
    PPO: clone_on_policy_batch_norm,
}


def test_dqn_train_with_batch_norm():
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        seed=1,
        tau=0.0,  # do not clone the target
        target_update_interval=100,  # Copy the stats to the target
    )

    (
        q_net_bias_before,
        q_net_running_mean_before,
        q_net_target_bias_before,
        q_net_target_running_mean_before,
    ) = clone_dqn_batch_norm_stats(model)

    model.learn(total_timesteps=200)
    # Force stats copy
    model.target_update_interval = 1
    model._on_step()

    (
        q_net_bias_after,
        q_net_running_mean_after,
        q_net_target_bias_after,
        q_net_target_running_mean_after,
    ) = clone_dqn_batch_norm_stats(model)

    assert ~th.isclose(q_net_bias_before, q_net_bias_after).all()
    assert ~th.isclose(q_net_running_mean_before, q_net_running_mean_after).all()

    # No weight update
    assert th.isclose(q_net_bias_before, q_net_target_bias_after).all()
    assert th.isclose(q_net_target_bias_before, q_net_target_bias_after).all()
    # Running stat should be copied even when tau=0
    assert th.isclose(q_net_running_mean_before, q_net_target_running_mean_before).all()
    assert th.isclose(q_net_running_mean_after, q_net_target_running_mean_after).all()


def test_td3_train_with_batch_norm():
    model = TD3(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        tau=0,  # do not copy the target
        seed=1,
    )

    (
        actor_bias_before,
        actor_running_mean_before,
        critic_bias_before,
        critic_running_mean_before,
        actor_target_bias_before,
        actor_target_running_mean_before,
        critic_target_bias_before,
        critic_target_running_mean_before,
    ) = clone_td3_batch_norm_stats(model)

    model.learn(total_timesteps=200)

    (
        actor_bias_after,
        actor_running_mean_after,
        critic_bias_after,
        critic_running_mean_after,
        actor_target_bias_after,
        actor_target_running_mean_after,
        critic_target_bias_after,
        critic_target_running_mean_after,
    ) = clone_td3_batch_norm_stats(model)

    assert ~th.isclose(actor_bias_before, actor_bias_after).all()
    assert ~th.isclose(actor_running_mean_before, actor_running_mean_after).all()

    assert ~th.isclose(critic_bias_before, critic_bias_after).all()
    assert ~th.isclose(critic_running_mean_before, critic_running_mean_after).all()

    assert th.isclose(actor_target_bias_before, actor_target_bias_after).all()
    # Running stat should be copied even when tau=0
    assert th.isclose(actor_running_mean_after, actor_target_running_mean_after).all()

    assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
    # Running stat should be copied even when tau=0
    assert th.isclose(critic_running_mean_after, critic_target_running_mean_after).all()


def test_sac_train_with_batch_norm():
    model = SAC(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        tau=0,  # do not copy the target
        seed=1,
    )

    (
        actor_bias_before,
        actor_running_mean_before,
        critic_bias_before,
        critic_running_mean_before,
        critic_target_bias_before,
        critic_target_running_mean_before,
    ) = clone_sac_batch_norm_stats(model)

    model.learn(total_timesteps=200)

    (
        actor_bias_after,
        actor_running_mean_after,
        critic_bias_after,
        critic_running_mean_after,
        critic_target_bias_after,
        critic_target_running_mean_after,
    ) = clone_sac_batch_norm_stats(model)

    assert ~th.isclose(actor_bias_before, actor_bias_after).all()
    assert ~th.isclose(actor_running_mean_before, actor_running_mean_after).all()

    assert ~th.isclose(critic_bias_before, critic_bias_after).all()
    # Running stat should be copied even when tau=0
    assert th.isclose(critic_running_mean_before, critic_target_running_mean_before).all()

    assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
    # Running stat should be copied even when tau=0
    assert th.isclose(critic_running_mean_after, critic_target_running_mean_after).all()


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_a2c_ppo_train_with_batch_norm(model_class, env_id):
    model = model_class(
        "MlpPolicy",
        env_id,
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        seed=1,
    )

    bias_before, running_mean_before = clone_on_policy_batch_norm(model)

    model.learn(total_timesteps=200)

    bias_after, running_mean_after = clone_on_policy_batch_norm(model)

    assert ~th.isclose(bias_before, bias_after).all()
    assert ~th.isclose(running_mean_before, running_mean_after).all()


@pytest.mark.parametrize("model_class", [DQN, TD3, SAC])
def test_offpolicy_collect_rollout_batch_norm(model_class):
    if model_class in [DQN]:
        env_id = "CartPole-v1"
    else:
        env_id = "Pendulum-v1"

    clone_helper = CLONE_HELPERS[model_class]

    learning_starts = 10
    model = model_class(
        "MlpPolicy",
        env_id,
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=learning_starts,
        seed=1,
        gradient_steps=0,
        train_freq=1,
    )

    batch_norm_stats_before = clone_helper(model)

    model.learn(total_timesteps=100)

    batch_norm_stats_after = clone_helper(model)

    # No change in batch norm params
    for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
        assert th.isclose(param_before, param_after).all()


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_a2c_ppo_collect_rollouts_with_batch_norm(model_class, env_id):
    model = model_class(
        "MlpPolicy",
        env_id,
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        seed=1,
        n_steps=64,
    )

    bias_before, running_mean_before = clone_on_policy_batch_norm(model)

    total_timesteps, callback = model._setup_learn(total_timesteps=2 * 64)

    for _ in range(2):
        model.collect_rollouts(model.get_env(), callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

    bias_after, running_mean_after = clone_on_policy_batch_norm(model)

    assert th.isclose(bias_before, bias_after).all()
    assert th.isclose(running_mean_before, running_mean_after).all()


@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_predict_with_dropout_batch_norm(model_class, env_id):
    if env_id == "CartPole-v1":
        if model_class in [SAC, TD3]:
            return
    elif model_class in [DQN]:
        return

    model_kwargs = dict(seed=1)
    clone_helper = CLONE_HELPERS[model_class]

    if model_class in [DQN, TD3, SAC]:
        model_kwargs["learning_starts"] = 0
    else:
        model_kwargs["n_steps"] = 64

    policy_kwargs = dict(
        features_extractor_class=FlattenBatchNormDropoutExtractor,
        net_arch=[16, 16],
    )
    model = model_class("MlpPolicy", env_id, policy_kwargs=policy_kwargs, verbose=1, **model_kwargs)

    batch_norm_stats_before = clone_helper(model)

    env = model.get_env()
    observation = env.reset()
    first_prediction, _ = model.predict(observation, deterministic=True)
    for _ in range(5):
        prediction, _ = model.predict(observation, deterministic=True)
        np.testing.assert_allclose(first_prediction, prediction)

    batch_norm_stats_after = clone_helper(model)

    # No change in batch norm params
    for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
        assert th.isclose(param_before, param_after).all()
