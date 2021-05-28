import os
import shutil

import gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, MaxAndSkipEnv
from stable_baselines3.common.env_util import is_wrapped, make_atari_env, make_vec_env, unwrap_wrapper
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


def test_vec_env_wrapper_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, wrapper_class=MaxAndSkipEnv, wrapper_kwargs={"skip": 3})
    assert env.get_attr("_skip")[0] == 3


def test_vec_env_monitor_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": False})
    assert env.get_attr("allow_early_resets")[0] is False

    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": False})
    assert env.get_attr("allow_early_resets")[0] is False

    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": True})
    assert env.get_attr("allow_early_resets")[0] is True

    env = make_atari_env(
        "BreakoutNoFrameskip-v4",
        n_envs=1,
        seed=0,
        monitor_kwargs={"allow_early_resets": True},
    )
    assert env.get_attr("allow_early_resets")[0] is True


def test_env_auto_monitor_wrap():
    env = gym.make("Pendulum-v0")
    model = A2C("MlpPolicy", env)
    assert model.env.env_is_wrapped(Monitor)[0] is True

    env = Monitor(env)
    model = A2C("MlpPolicy", env)
    assert model.env.env_is_wrapped(Monitor)[0] is True

    model = A2C("MlpPolicy", "Pendulum-v0")
    assert model.env.env_is_wrapped(Monitor)[0] is True


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

    # Test that warning is given about no monitor
    eval_env = gym.make("Pendulum-v0")
    with pytest.warns(UserWarning):
        _ = evaluate_policy(model, eval_env, n_eval_episodes)


class ZeroRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward * 0


class AlwaysDoneWrapper(gym.Wrapper):
    # Pretends that environment only has single step for each
    # episode.
    def __init__(self, env):
        super(AlwaysDoneWrapper, self).__init__(env)
        self.last_obs = None
        self.needs_reset = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.needs_reset = done
        self.last_obs = obs
        return obs, reward, True, info

    def reset(self, **kwargs):
        if self.needs_reset:
            obs = self.env.reset(**kwargs)
            self.last_obs = obs
            self.needs_reset = False
        return self.last_obs


@pytest.mark.parametrize("n_envs", [1, 2, 5, 7])
def test_evaluate_vector_env(n_envs):
    # Tests that the number of episodes evaluated is correct
    n_eval_episodes = 6

    env = make_vec_env("CartPole-v1", n_envs)
    model = A2C("MlpPolicy", "CartPole-v1", seed=0)

    class CountCallback:
        def __init__(self):
            self.count = 0

        def __call__(self, locals_, globals_):
            if locals_["done"]:
                self.count += 1

    count_callback = CountCallback()

    evaluate_policy(model, env, n_eval_episodes, callback=count_callback)

    assert count_callback.count == n_eval_episodes


@pytest.mark.parametrize("vec_env_class", [None, DummyVecEnv, SubprocVecEnv])
def test_evaluate_policy_monitors(vec_env_class):
    # Make numpy warnings throw exception
    np.seterr(all="raise")
    # Test that results are correct with monitor environments.
    # Also test VecEnvs
    n_eval_episodes = 3
    n_envs = 2
    env_id = "CartPole-v0"
    model = A2C("MlpPolicy", env_id, seed=0)

    def make_eval_env(with_monitor, wrapper_class=gym.Wrapper):
        # Make eval environment with or without monitor in root,
        # and additionally wrapped with another wrapper (after Monitor).
        env = None
        if vec_env_class is None:
            # No vecenv, traditional env
            env = gym.make(env_id)
            if with_monitor:
                env = Monitor(env)
            env = wrapper_class(env)
        else:
            if with_monitor:
                env = vec_env_class([lambda: wrapper_class(Monitor(gym.make(env_id)))] * n_envs)
            else:
                env = vec_env_class([lambda: wrapper_class(gym.make(env_id))] * n_envs)
        return env

    # Test that evaluation with VecEnvs works as expected
    eval_env = make_eval_env(with_monitor=True)
    _ = evaluate_policy(model, eval_env, n_eval_episodes)
    eval_env.close()

    # Warning without Monitor
    eval_env = make_eval_env(with_monitor=False)
    with pytest.warns(UserWarning):
        _ = evaluate_policy(model, eval_env, n_eval_episodes)
    eval_env.close()

    # Test that we gather correct reward with Monitor wrapper
    # Sanity check that we get zero-reward without Monitor
    eval_env = make_eval_env(with_monitor=False, wrapper_class=ZeroRewardWrapper)
    average_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes, warn=False)
    assert average_reward == 0.0, "ZeroRewardWrapper wrapper for testing did not work"
    eval_env.close()

    # Should get non-zero-rewards with Monitor (true reward)
    eval_env = make_eval_env(with_monitor=True, wrapper_class=ZeroRewardWrapper)
    average_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes)
    assert average_reward > 0.0, "evaluate_policy did not get reward from Monitor"
    eval_env.close()

    # Test that we also track correct episode dones, not the wrapped ones.
    # Sanity check that we get only one step per episode.
    eval_env = make_eval_env(with_monitor=False, wrapper_class=AlwaysDoneWrapper)
    episode_rewards, episode_lengths = evaluate_policy(
        model, eval_env, n_eval_episodes, return_episode_rewards=True, warn=False
    )
    assert all(map(lambda l: l == 1, episode_lengths)), "AlwaysDoneWrapper did not fix episode lengths to one"
    eval_env.close()

    # Should get longer episodes with with Monitor (true episodes)
    eval_env = make_eval_env(with_monitor=True, wrapper_class=AlwaysDoneWrapper)
    episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes, return_episode_rewards=True)
    assert all(map(lambda l: l > 1, episode_lengths)), "evaluate_policy did not get episode lengths from Monitor"
    eval_env.close()


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


def test_is_wrapped():
    """Test that is_wrapped correctly detects wraps"""
    env = gym.make("Pendulum-v0")
    env = gym.Wrapper(env)
    assert not is_wrapped(env, Monitor)
    monitor_env = Monitor(env)
    assert is_wrapped(monitor_env, Monitor)
    env = gym.Wrapper(monitor_env)
    assert is_wrapped(env, Monitor)
    # Test that unwrap works as expected
    assert unwrap_wrapper(env, Monitor) == monitor_env


def test_ppo_warnings():
    """Test that PPO warns and errors correctly on
    problematic rollour buffer sizes"""

    # Only 1 step: advantage normalization will return NaN
    with pytest.raises(AssertionError):
        PPO("MlpPolicy", "Pendulum-v0", n_steps=1)

    # Truncated mini-batch
    with pytest.warns(UserWarning):
        PPO("MlpPolicy", "Pendulum-v0", n_steps=6, batch_size=8)
