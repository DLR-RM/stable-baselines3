import os
import shutil

import gym
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3, DQN, DDPG])
def test_callbacks(tmp_path, model_class):
    log_folder = tmp_path / "logs/callbacks/"

    # Dyn only support discrete actions
    env_name = select_env(model_class)
    # Create RL model
    # Small network for fast test
    model = model_class("MlpPolicy", env_name, policy_kwargs=dict(net_arch=[32]))

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_folder)

    eval_env = gym.make(env_name)
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)

    eval_callback = EvalCallback(
        eval_env, callback_on_new_best=callback_on_best, best_model_save_path=log_folder, log_path=log_folder, eval_freq=100
    )
    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=log_folder, name_prefix="event")

    event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

    # Stop training if max number of episodes is reached
    callback_max_eps = StopTrainingOnMaxEpisodes(max_episodes=1, verbose=1)

    callback = CallbackList([checkpoint_callback, eval_callback, event_callback, callback_max_eps])
    model.learn(500, callback=callback)

    # Check access to local variables
    assert model.env.observation_space.contains(callback.locals["new_obs"][0])
    # Check that the child callback was called
    assert checkpoint_callback.locals["new_obs"] is callback.locals["new_obs"]
    assert event_callback.locals["new_obs"] is callback.locals["new_obs"]
    assert checkpoint_on_event.locals["new_obs"] is callback.locals["new_obs"]

    model.learn(500, callback=None)
    # Transform callback into a callback list automatically
    model.learn(500, callback=[checkpoint_callback, eval_callback])
    # Automatic wrapping, old way of doing callbacks
    model.learn(500, callback=lambda _locals, _globals: True)

    # Testing for models that support multiple envs
    if model_class in [A2C, PPO]:
        envs = make_vec_env(env_name, n_envs=2, seed=0, vec_env_cls=SubprocVecEnv)

        model = model_class("MlpPolicy", envs, policy_kwargs=dict(net_arch=[32]))

        callback = CallbackList([callback_max_eps])

        model.learn(1000, callback=callback)

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)


def select_env(model_class) -> str:
    if model_class is DQN:
        return "CartPole-v0"
    else:
        return "Pendulum-v0"
