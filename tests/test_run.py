import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from tests.test_predict import (
    clone_dqn_batch_norm_stats,
    clone_td3_batch_norm_stats,
    clone_sac_batch_norm_stats,
    clone_batch_norm_stats,
    FlattenBatchNormDropoutExtractor,
)

normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


@pytest.mark.parametrize("model_class", [TD3, DDPG])
@pytest.mark.parametrize("action_noise", [normal_action_noise, OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1 * np.ones(1))])
def test_deterministic_pg(model_class, action_noise):
    """
    Test for DDPG and variants (TD3).
    """
    model = model_class(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        verbose=1,
        create_eval_env=True,
        buffer_size=250,
        action_noise=action_noise,
    )
    model.learn(total_timesteps=300, eval_freq=250)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
def test_a2c(env_id):
    model = A2C("MlpPolicy", env_id, seed=0, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
@pytest.mark.parametrize("clip_range_vf", [None, 0.2, -0.2])
def test_ppo(env_id, clip_range_vf):
    if clip_range_vf is not None and clip_range_vf < 0:
        # Should throw an error
        with pytest.raises(AssertionError):
            model = PPO(
                "MlpPolicy",
                env_id,
                seed=0,
                policy_kwargs=dict(net_arch=[16]),
                verbose=1,
                create_eval_env=True,
                clip_range_vf=clip_range_vf,
            )
    else:
        model = PPO(
            "MlpPolicy",
            env_id,
            n_steps=512,
            seed=0,
            policy_kwargs=dict(net_arch=[16]),
            verbose=1,
            create_eval_env=True,
            clip_range_vf=clip_range_vf,
        )
        model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("ent_coef", ["auto", 0.01, "auto_0.01"])
def test_sac(ent_coef):
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        verbose=1,
        create_eval_env=True,
        buffer_size=250,
        ent_coef=ent_coef,
        action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
    )
    model.learn(total_timesteps=300, eval_freq=250)


@pytest.mark.parametrize("n_critics", [1, 3])
def test_n_critics(n_critics):
    # Test SAC with different number of critics, for TD3, n_critics=1 corresponds to DDPG
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64], n_critics=n_critics),
        learning_starts=100,
        buffer_size=10000,
        verbose=1,
    )
    model.learn(total_timesteps=300)


def test_dqn():
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        buffer_size=500,
        learning_rate=3e-4,
        verbose=1,
        create_eval_env=True,
    )
    model.learn(total_timesteps=500, eval_freq=250)


@pytest.mark.parametrize("train_freq", [4, (4, "step"), (1, "episode")])
def test_train_freq(tmp_path, train_freq):

    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64], n_critics=1),
        learning_starts=100,
        buffer_size=10000,
        verbose=1,
        train_freq=train_freq,
    )
    model.learn(total_timesteps=150)
    model.save(tmp_path / "test_save.zip")
    env = model.get_env()
    model = SAC.load(tmp_path / "test_save.zip", env=env)
    model.learn(total_timesteps=150)
    model = SAC.load(tmp_path / "test_save.zip", train_freq=train_freq, env=env)
    model.learn(total_timesteps=150)


@pytest.mark.parametrize("train_freq", ["4", ("1", "episode"), "non_sense", (1, "close")])
def test_train_freq_fail(train_freq):
    with pytest.raises(ValueError):
        model = SAC(
            "MlpPolicy",
            "Pendulum-v0",
            policy_kwargs=dict(net_arch=[64, 64], n_critics=1),
            learning_starts=100,
            buffer_size=10000,
            verbose=1,
            train_freq=train_freq,
        )
        model.learn(total_timesteps=250)


def test_dqn_train_with_batch_norm():
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        tau=0,
        seed=1,
    )

    (
        q_net_bias_before,
        q_net_running_mean_before,
        q_net_target_bias_before,
        q_net_target_running_mean_before,
    ) = clone_dqn_batch_norm_stats(model)

    model.learn(total_timesteps=200)

    (
        q_net_bias_after,
        q_net_running_mean_after,
        q_net_target_bias_after,
        q_net_target_running_mean_after,
    ) = clone_dqn_batch_norm_stats(model)

    assert ~th.isclose(q_net_bias_before, q_net_bias_after).all()
    assert ~th.isclose(q_net_running_mean_before, q_net_running_mean_after).all()

    assert th.isclose(q_net_target_bias_before, q_net_target_bias_after).all()
    assert th.isclose(q_net_target_running_mean_before, q_net_target_running_mean_after).all()


def test_dqn_collect_rollouts_with_batch_norm():
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        seed=1,
    )

    (
        q_net_bias_before,
        q_net_running_mean_before,
        q_net_target_bias_before,
        q_net_target_running_mean_before,
    ) = clone_dqn_batch_norm_stats(model)

    total_timesteps, callback = model._setup_learn(total_timesteps=100, eval_env=model.get_env())

    for _ in range(10):
        model.collect_rollouts(
            model.get_env(),
            train_freq=model.train_freq,
            action_noise=model.action_noise,
            callback=callback,
            learning_starts=model.learning_starts,
            replay_buffer=model.replay_buffer,
        )

    (
        q_net_bias_after,
        q_net_running_mean_after,
        q_net_target_bias_after,
        q_net_target_running_mean_after,
    ) = clone_dqn_batch_norm_stats(model)

    assert th.isclose(q_net_bias_before, q_net_bias_after).all()
    assert th.isclose(q_net_running_mean_before, q_net_running_mean_after).all()

    assert th.isclose(q_net_target_bias_before, q_net_target_bias_after).all()
    assert th.isclose(q_net_target_running_mean_before, q_net_target_running_mean_after).all()


def test_td3_train_with_batch_norm():
    model = TD3(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        tau=0,
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
    assert th.isclose(actor_target_running_mean_before, actor_target_running_mean_after).all()

    assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
    assert th.isclose(critic_target_running_mean_before, critic_target_running_mean_after).all()


def test_td3_collect_rollouts_with_batch_norm():
    model = TD3(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
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

    total_timesteps, callback = model._setup_learn(total_timesteps=100, eval_env=model.get_env())

    for _ in range(10):
        model.collect_rollouts(
            model.get_env(),
            train_freq=model.train_freq,
            action_noise=model.action_noise,
            callback=callback,
            learning_starts=model.learning_starts,
            replay_buffer=model.replay_buffer,
        )

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

    assert th.isclose(actor_bias_before, actor_bias_after).all()
    assert th.isclose(actor_running_mean_before, actor_running_mean_after).all()

    assert th.isclose(critic_bias_before, critic_bias_after).all()
    assert th.isclose(critic_running_mean_before, critic_running_mean_after).all()

    assert th.isclose(actor_target_bias_before, actor_target_bias_after).all()
    assert th.isclose(actor_target_running_mean_before, actor_target_running_mean_after).all()

    assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
    assert th.isclose(critic_target_running_mean_before, critic_target_running_mean_after).all()


def test_sac_train_with_batch_norm():
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        tau=0,
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
    assert ~th.isclose(critic_running_mean_before, critic_running_mean_after).all()

    assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
    assert th.isclose(critic_target_running_mean_before, critic_target_running_mean_after).all()


def test_sac_collect_rollouts_with_batch_norm():
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
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

    total_timesteps, callback = model._setup_learn(total_timesteps=100, eval_env=model.get_env())

    for _ in range(10):
        model.collect_rollouts(
            model.get_env(),
            train_freq=model.train_freq,
            action_noise=model.action_noise,
            callback=callback,
            learning_starts=model.learning_starts,
            replay_buffer=model.replay_buffer,
        )

    (
        actor_bias_after,
        actor_running_mean_after,
        critic_bias_after,
        critic_running_mean_after,
        critic_target_bias_after,
        critic_target_running_mean_after,
    ) = clone_sac_batch_norm_stats(model)

    assert th.isclose(actor_bias_before, actor_bias_after).all()
    assert th.isclose(actor_running_mean_before, actor_running_mean_after).all()

    assert th.isclose(critic_bias_before, critic_bias_after).all()
    assert th.isclose(critic_running_mean_before, critic_running_mean_after).all()

    assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
    assert th.isclose(critic_target_running_mean_before, critic_target_running_mean_after).all()


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env_id", ["Pendulum-v0", "CartPole-v1"])
def test_a2c_ppo_train_with_batch_norm(model_class, env_id):
    model = model_class(
        "MlpPolicy",
        env_id,
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        seed=1,
    )

    batch_norm = model.policy.features_extractor.batch_norm
    bias_before, running_mean_before = clone_batch_norm_stats(batch_norm)

    model.learn(total_timesteps=200)

    bias_after, running_mean_after = clone_batch_norm_stats(batch_norm)

    assert ~th.isclose(bias_before, bias_after).all()
    assert ~th.isclose(running_mean_before, running_mean_after).all()


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("env_id", ["Pendulum-v0", "CartPole-v1"])
def test_a2c_ppo_collect_rollouts_with_batch_norm(model_class, env_id):
    model = model_class(
        "MlpPolicy",
        env_id,
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        seed=1,
    )

    batch_norm = model.policy.features_extractor.batch_norm
    bias_before, running_mean_before = clone_batch_norm_stats(batch_norm)

    total_timesteps, callback = model._setup_learn(total_timesteps=100, eval_env=model.get_env())

    for _ in range(10):
        model.collect_rollouts(model.get_env(), callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

    bias_after, running_mean_after = clone_batch_norm_stats(batch_norm)

    assert th.isclose(bias_before, bias_after).all()
    assert th.isclose(running_mean_before, running_mean_after).all()
