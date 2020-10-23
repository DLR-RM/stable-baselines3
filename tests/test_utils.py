import os
import shutil

import gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise, OrnsteinUhlenbeckActionNoise, VectorizedActionNoise
from stable_baselines3.common.utils import polyak_update, zip_strict
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


@pytest.mark.parametrize("env_id", ["CartPole-v1", lambda: gym.make("CartPole-v1")])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("vec_env_cls", [None, SubprocVecEnv])
@pytest.mark.parametrize("wrapper_class", [None, gym.wrappers.TimeLimit])
def test_make_vec_env(env_id, n_envs, vec_env_cls, wrapper_class):
    env = make_vec_env(env_id, n_envs, vec_env_cls=vec_env_cls, wrapper_class=wrapper_class, monitor_dir=None, seed=0)

    assert env.num_envs == n_envs

    if vec_env_cls is None:
        assert isinstance(env, DummyVecEnv)
        if wrapper_class is not None:
            assert isinstance(env.envs[0], wrapper_class)
        else:
            assert isinstance(env.envs[0], Monitor)
    else:
        assert isinstance(env, SubprocVecEnv)
    # Kill subprocesses
    env.close()


@pytest.mark.parametrize("env_id", ["BreakoutNoFrameskip-v4"])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("wrapper_kwargs", [None, dict(clip_reward=False, screen_size=60)])
def test_make_atari_env(env_id, n_envs, wrapper_kwargs):
    env_id = "BreakoutNoFrameskip-v4"
    env = make_atari_env(env_id, n_envs, wrapper_kwargs=wrapper_kwargs, monitor_dir=None, seed=0)

    assert env.num_envs == n_envs

    obs = env.reset()

    new_obs, reward, _, _ = env.step([env.action_space.sample() for _ in range(n_envs)])

    assert obs.shape == new_obs.shape

    # Wrapped into DummyVecEnv
    wrapped_atari_env = env.envs[0]
    if wrapper_kwargs is not None:
        assert obs.shape == (n_envs, 60, 60, 1)
        assert wrapped_atari_env.observation_space.shape == (60, 60, 1)
        assert not isinstance(wrapped_atari_env.env, ClipRewardEnv)
    else:
        assert obs.shape == (n_envs, 84, 84, 1)
        assert wrapped_atari_env.observation_space.shape == (84, 84, 1)
        assert isinstance(wrapped_atari_env.env, ClipRewardEnv)
        assert np.max(np.abs(reward)) < 1.0


def test_vec_env_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, env_kwargs={"goal_velocity": 0.11})
    assert env.get_attr("goal_velocity")[0] == 0.11


def test_custom_vec_env(tmp_path):
    """
    Stand alone test for a special case (passing a custom VecEnv class) to avoid doubling the number of tests.
    """
    monitor_dir = tmp_path / "test_make_vec_env/"
    env = make_vec_env(
        "CartPole-v1",
        n_envs=1,
        monitor_dir=monitor_dir,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": None},
    )

    assert env.num_envs == 1
    assert isinstance(env, SubprocVecEnv)
    assert os.path.isdir(monitor_dir)
    # Kill subprocess
    env.close()
    # Cleanup folder
    shutil.rmtree(monitor_dir)

    # This should fail because DummyVecEnv does not have any keyword argument
    with pytest.raises(TypeError):
        make_vec_env("CartPole-v1", n_envs=1, vec_env_kwargs={"dummy": False})


def test_evaluate_policy():
    model = A2C("MlpPolicy", "Pendulum-v0", seed=0)
    n_steps_per_episode, n_eval_episodes = 200, 2
    model.n_callback_calls = 0

    def dummy_callback(locals_, _globals):
        locals_["model"].n_callback_calls += 1

    _, episode_lengths = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes,
        deterministic=True,
        render=False,
        callback=dummy_callback,
        reward_threshold=None,
        return_episode_rewards=True,
    )

    n_steps = sum(episode_lengths)
    assert n_steps == n_steps_per_episode * n_eval_episodes
    assert n_steps == model.n_callback_calls

    # Reaching a mean reward of zero is impossible with the Pendulum env
    with pytest.raises(AssertionError):
        evaluate_policy(model, model.get_env(), n_eval_episodes, reward_threshold=0.0)

    episode_rewards, _ = evaluate_policy(model, model.get_env(), n_eval_episodes, return_episode_rewards=True)
    assert len(episode_rewards) == n_eval_episodes


def test_vec_noise():
    num_envs = 4
    num_actions = 10
    mu = np.zeros(num_actions)
    sigma = np.ones(num_actions) * 0.4
    base: ActionNoise = OrnsteinUhlenbeckActionNoise(mu, sigma)
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(base, -1)
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(base, None)
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(base, "whatever")

    vec = VectorizedActionNoise(base, num_envs)
    assert vec.n_envs == num_envs
    assert vec().shape == (num_envs, num_actions)
    assert not (vec() == base()).all()
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(None, num_envs)
    with pytest.raises(TypeError):
        vec = VectorizedActionNoise(12, num_envs)
    with pytest.raises(AssertionError):
        vec.noises = []
    with pytest.raises(TypeError):
        vec.noises = None
    with pytest.raises(ValueError):
        vec.noises = [None] * vec.n_envs
    with pytest.raises(AssertionError):
        vec.noises = [base] * (num_envs - 1)
    assert all(isinstance(noise, type(base)) for noise in vec.noises)
    assert len(vec.noises) == num_envs


def test_polyak():
    param1, param2 = th.nn.Parameter(th.ones((5, 5))), th.nn.Parameter(th.zeros((5, 5)))
    target1, target2 = th.nn.Parameter(th.ones((5, 5))), th.nn.Parameter(th.zeros((5, 5)))
    tau = 0.1
    polyak_update([param1], [param2], tau)
    with th.no_grad():
        for param, target_param in zip([target1], [target2]):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    assert th.allclose(param1, target1)
    assert th.allclose(param2, target2)


def test_zip_strict():
    # Iterables with different lengths
    list_a = [0, 1]
    list_b = [1, 2, 3]
    # zip does not raise any error
    for _, _ in zip(list_a, list_b):
        pass

    # zip_strict does raise an error
    with pytest.raises(ValueError):
        for _, _ in zip_strict(list_a, list_b):
            pass

    # same length, should not raise an error
    for _, _ in zip_strict(list_a, list_b[: len(list_a)]):
        pass


def test_cmd_util_rename():
    """Test that importing cmd_util still works but raises warning"""
    with pytest.warns(FutureWarning):
        from stable_baselines3.common.cmd_util import make_vec_env  # noqa: F401
