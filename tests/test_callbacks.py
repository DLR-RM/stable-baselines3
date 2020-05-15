import os
import shutil

import pytest
import gym

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (CallbackList, CheckpointCallback, EvalCallback,
                                                EveryNTimesteps, StopTrainingOnRewardThreshold)


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3])
def test_callbacks(model_class):
    log_folder = './logs/callbacks/'
    # Create RL model
    # Small network for fast test
    model = model_class('MlpPolicy', 'Pendulum-v0', policy_kwargs=dict(net_arch=[32]))

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_folder)

    eval_env = gym.make('Pendulum-v0')
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)

    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best,
                                 best_model_save_path=log_folder,
                                 log_path=log_folder, eval_freq=100)

    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=log_folder,
                                             name_prefix='event')
    event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

    callback = CallbackList([checkpoint_callback, eval_callback, event_callback])

    model.learn(500, callback=callback)
    model.learn(500, callback=None)
    # Transform callback into a callback list automatically
    model.learn(500, callback=[checkpoint_callback, eval_callback])
    # Automatic wrapping, old way of doing callbacks
    model.learn(500, callback=lambda _locals, _globals: True)
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
