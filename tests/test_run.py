import gymnasium as gym
import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


@pytest.mark.parametrize("model_class", [TD3, DDPG])
@pytest.mark.parametrize(
    "action_noise",
    [normal_action_noise, OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1 * np.ones(1))],
)
def test_deterministic_pg(model_class, action_noise):
    """
    Test for DDPG and variants (TD3).
    """
    model = model_class(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        verbose=1,
        buffer_size=250,
        action_noise=action_noise,
    )
    model.learn(total_timesteps=200)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_a2c(env_id):
    model = A2C("MlpPolicy", env_id, seed=0, policy_kwargs=dict(net_arch=[16]), verbose=1)
    model.learn(total_timesteps=64)


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("normalize_advantage", [False, True])
def test_advantage_normalization(model_class, normalize_advantage):
    model = model_class("MlpPolicy", "CartPole-v1", n_steps=64, normalize_advantage=normalize_advantage)
    model.learn(64)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
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
            clip_range_vf=clip_range_vf,
            n_epochs=2,
        )
        model.learn(total_timesteps=1000)


@pytest.mark.parametrize("ent_coef", ["auto", 0.01, "auto_0.01"])
def test_sac(ent_coef):
    model = SAC(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        verbose=1,
        buffer_size=250,
        ent_coef=ent_coef,
        action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
    )
    model.learn(total_timesteps=200)


@pytest.mark.parametrize("n_critics", [1, 3])
def test_n_critics(n_critics):
    # Test SAC with different number of critics, for TD3, n_critics=1 corresponds to DDPG
    model = SAC(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[64, 64], n_critics=n_critics),
        learning_starts=100,
        buffer_size=10000,
        verbose=1,
    )
    model.learn(total_timesteps=200)


def test_dqn():
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        buffer_size=500,
        learning_rate=3e-4,
        verbose=1,
    )
    model.learn(total_timesteps=200)


@pytest.mark.parametrize("train_freq", [4, (4, "step"), (1, "episode")])
def test_train_freq(tmp_path, train_freq):
    model = SAC(
        "MlpPolicy",
        "Pendulum-v1",
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
            "Pendulum-v1",
            policy_kwargs=dict(net_arch=[64, 64], n_critics=1),
            learning_starts=100,
            buffer_size=10000,
            verbose=1,
            train_freq=train_freq,
        )
        model.learn(total_timesteps=250)


@pytest.mark.parametrize("model_class", [SAC, TD3, DDPG, DQN])
def test_offpolicy_multi_env(model_class):
    kwargs = {}
    if model_class in [SAC, TD3, DDPG]:
        env_id = "Pendulum-v1"
        policy_kwargs = dict(net_arch=[64], n_critics=1)
        # Check auto-conversion to VectorizedActionNoise
        kwargs = dict(action_noise=NormalActionNoise(np.zeros(1), 0.1 * np.ones(1)))
        if model_class == SAC:
            kwargs["use_sde"] = True
            kwargs["sde_sample_freq"] = 4
    else:
        env_id = "CartPole-v1"
        policy_kwargs = dict(net_arch=[64])

    def make_env():
        env = gym.make(env_id)
        # to check that the code handling timeouts runs
        env = gym.wrappers.TimeLimit(env, 50)
        return env

    env = make_vec_env(make_env, n_envs=2)
    model = model_class(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_starts=100,
        buffer_size=10000,
        verbose=0,
        train_freq=5,
        **kwargs,
    )
    model.learn(total_timesteps=150)

    # Check that gradient_steps=-1 works as expected:
    # perform as many gradient_steps as transitions collected
    train_freq = 3
    model = model_class(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_starts=0,
        buffer_size=10000,
        verbose=0,
        train_freq=train_freq,
        gradient_steps=-1,
        **kwargs,
    )
    model.learn(total_timesteps=train_freq)
    assert model.logger.name_to_value["train/n_updates"] == train_freq * env.num_envs


def test_warn_dqn_multi_env():
    with pytest.warns(UserWarning, match="The number of environments used is greater"):
        DQN(
            "MlpPolicy",
            make_vec_env("CartPole-v1", n_envs=2),
            buffer_size=100,
            target_update_interval=1,
        )


def test_ppo_warnings():
    """Test that PPO warns and errors correctly on
    problematic rollout buffer sizes"""

    # Only 1 step: advantage normalization will return NaN
    with pytest.raises(AssertionError):
        PPO("MlpPolicy", "Pendulum-v1", n_steps=1)

    # batch_size of 1 is allowed when normalize_advantage=False
    model = PPO("MlpPolicy", "Pendulum-v1", n_steps=1, batch_size=1, normalize_advantage=False)
    model.learn(4)

    # Truncated mini-batch
    # Batch size 1 yields NaN with normalized advantage because
    # torch.std(some_length_1_tensor) == NaN
    # advantage normalization is automatically deactivated
    # in that case
    with pytest.warns(UserWarning, match="there will be a truncated mini-batch of size 1"):
        model = PPO("MlpPolicy", "Pendulum-v1", n_steps=64, batch_size=63, verbose=1)
        model.learn(64)

    loss = model.logger.name_to_value["train/loss"]
    assert loss > 0
    assert not np.isnan(loss)  # check not nan (since nan does not equal nan)
