import os
import shutil

import gym
import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import BitFlippingEnv, IdentityEnv
from stable_baselines3.common.vec_env import DummyVecEnv


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3, DQN, DDPG])
def test_callbacks(tmp_path, model_class):
    log_folder = tmp_path / "logs/callbacks/"

    # DQN only support discrete actions
    env_name = select_env(model_class)
    # Create RL model
    # Small network for fast test
    model = model_class("MlpPolicy", env_name, policy_kwargs=dict(net_arch=[32]))

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_folder)

    eval_env = gym.make(env_name)
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=log_folder,
        log_path=log_folder,
        eval_freq=100,
        warn=False,
    )
    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=log_folder, name_prefix="event")

    event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

    # Stop training if max number of episodes is reached
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=100, verbose=1)

    callback = CallbackList([checkpoint_callback, eval_callback, event_callback, callback_max_episodes])
    model.learn(500, callback=callback)

    # Check access to local variables
    assert model.env.observation_space.contains(callback.locals["new_obs"][0])
    # Check that the child callback was called
    assert checkpoint_callback.locals["new_obs"] is callback.locals["new_obs"]
    assert event_callback.locals["new_obs"] is callback.locals["new_obs"]
    assert checkpoint_on_event.locals["new_obs"] is callback.locals["new_obs"]
    # Check that internal callback counters match models' counters
    assert event_callback.num_timesteps == model.num_timesteps
    assert event_callback.n_calls == model.num_timesteps

    model.learn(500, callback=None)
    # Transform callback into a callback list automatically
    model.learn(500, callback=[checkpoint_callback, eval_callback])
    # Automatic wrapping, old way of doing callbacks
    model.learn(500, callback=lambda _locals, _globals: True)

    # Testing models that support multiple envs
    if model_class in [A2C, PPO]:
        max_episodes = 1
        n_envs = 2
        # Pendulum-v0 has a timelimit of 200 timesteps
        max_episode_length = 200
        envs = make_vec_env(env_name, n_envs=n_envs, seed=0)

        model = model_class("MlpPolicy", envs, policy_kwargs=dict(net_arch=[32]))

        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
        callback = CallbackList([callback_max_episodes])
        model.learn(1000, callback=callback)

        # Check that the actual number of episodes and timesteps per env matches the expected one
        episodes_per_env = callback_max_episodes.n_episodes // n_envs
        assert episodes_per_env == max_episodes
        timesteps_per_env = model.num_timesteps // n_envs
        assert timesteps_per_env == max_episode_length

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)


def select_env(model_class) -> str:
    if model_class is DQN:
        return "CartPole-v0"
    else:
        return "Pendulum-v0"


def test_eval_callback_vec_env():
    # tests that eval callback does not crash when given a vector
    n_eval_envs = 3
    train_env = IdentityEnv()
    eval_env = DummyVecEnv([lambda: IdentityEnv()] * n_eval_envs)
    model = A2C("MlpPolicy", train_env, seed=0)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=100,
        warn=False,
    )
    model.learn(300, callback=eval_callback)
    assert eval_callback.last_mean_reward == 100.0


def test_eval_success_logging(tmp_path):
    n_bits = 2
    n_envs = 2
    env = BitFlippingEnv(n_bits=n_bits)
    eval_env = DummyVecEnv([lambda: BitFlippingEnv(n_bits=n_bits)] * n_envs)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=250,
        log_path=tmp_path,
        warn=False,
    )
    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        learning_starts=100,
        seed=0,
        replay_buffer_kwargs=dict(max_episode_length=n_bits),
    )
    model.learn(500, callback=eval_callback)
    assert len(eval_callback._is_success_buffer) > 0
    # More than 50% success rate
    assert np.mean(eval_callback._is_success_buffer) > 0.5
