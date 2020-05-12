import os
import shutil

import pytest
import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


@pytest.mark.parametrize("env_id", ['CartPole-v1', lambda: gym.make('CartPole-v1')])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("vec_env_cls", [None, SubprocVecEnv])
@pytest.mark.parametrize("wrapper_class", [None, gym.wrappers.TimeLimit])
def test_make_vec_env(env_id, n_envs, vec_env_cls, wrapper_class):
    env = make_vec_env(env_id, n_envs, vec_env_cls=vec_env_cls,
                       wrapper_class=wrapper_class, monitor_dir=None, seed=0)

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


@pytest.mark.parametrize("env_id", ['BreakoutNoFrameskip-v4'])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("wrapper_kwargs", [None, dict(clip_reward=False, screen_size=60)])
def test_make_atari_env(env_id, n_envs, wrapper_kwargs):
    env_id = 'BreakoutNoFrameskip-v4'
    env = make_atari_env(env_id, n_envs,
                         wrapper_kwargs=wrapper_kwargs, monitor_dir=None, seed=0)

    assert env.num_envs == n_envs

    obs = env.reset()

    new_obs, reward, _, _ = env.step([env.action_space.sample() for _ in range(n_envs)])

    assert obs.shape == new_obs.shape

    # Wrapped into DummyVecEnv
    wrapped_atari_env = env.envs[0]
    if wrapper_kwargs is not None:
        assert obs.shape == (n_envs, 60, 60, 1)
        assert wrapped_atari_env.observation_space.shape == (60, 60, 1)
        assert wrapped_atari_env.clip_reward is False
    else:
        assert obs.shape == (n_envs, 84, 84, 1)
        assert wrapped_atari_env.observation_space.shape == (84, 84, 1)
        assert wrapped_atari_env.clip_reward is True
        assert np.max(np.abs(reward)) < 1.0


def test_custom_vec_env(tmp_path):
    """
    Stand alone test for a special case (passing a custom VecEnv class) to avoid doubling the number of tests.
    """
    monitor_dir = tmp_path / 'test_make_vec_env/'
    env = make_vec_env('CartPole-v1', n_envs=1,
                       monitor_dir=monitor_dir, seed=0,
                       vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': None})

    assert env.num_envs == 1
    assert isinstance(env, SubprocVecEnv)
    assert os.path.isdir(monitor_dir)
    # Kill subprocess
    env.close()
    # Cleanup folder
    shutil.rmtree(monitor_dir)

    # This should fail because DummyVecEnv does not have any keyword argument
    with pytest.raises(TypeError):
        make_vec_env('CartPole-v1', n_envs=1, vec_env_kwargs={'dummy': False})


def test_evaluate_policy():
    model = A2C('MlpPolicy', 'Pendulum-v0', seed=0)
    n_steps_per_episode, n_eval_episodes = 200, 2
    model.n_callback_calls = 0

    def dummy_callback(locals_, _globals):
        locals_['model'].n_callback_calls += 1

    _, episode_lengths = evaluate_policy(model, model.get_env(), n_eval_episodes, deterministic=True,
                                         render=False, callback=dummy_callback, reward_threshold=None,
                                         return_episode_rewards=True)

    n_steps = sum(episode_lengths)
    assert n_steps == n_steps_per_episode * n_eval_episodes
    assert n_steps == model.n_callback_calls

    # Reaching a mean reward of zero is impossible with the Pendulum env
    with pytest.raises(AssertionError):
        evaluate_policy(model, model.get_env(), n_eval_episodes, reward_threshold=0.0)

    episode_rewards, _ = evaluate_policy(model, model.get_env(), n_eval_episodes, return_episode_rewards=True)
    assert len(episode_rewards) == n_eval_episodes
