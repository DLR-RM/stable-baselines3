import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

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
