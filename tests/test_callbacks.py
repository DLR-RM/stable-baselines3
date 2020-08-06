import os
import shutil
import copy

import gym
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnRewardThreshold,
    BaseCallback
)

def call_counter_wrapper(fn):
    def internal(self, *args, **kwargs):
        v = fn(self, *args, **kwargs)
        setattr(self, f"_{fn.__name__}_called", getattr(self, f"_{fn.__name__}_called", 0)+1)
        return v
    return internal

@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3, DQN, DDPG])
def test_callbacks(tmp_path, model_class):
    log_folder = tmp_path / "logs/callbacks/"

    # Dyn only support discrete actions
    env_name = select_env(model_class)
    # Create RL model
    # Small network for fast test
    model = model_class("MlpPolicy", env_name, policy_kwargs=dict(net_arch=[32]))
    
    # Instrument the class to count the number of calls
    _CheckpointCallback = CheckpointCallback
    _CheckpointCallback.update_locals = call_counter_wrapper(_CheckpointCallback.update_locals)
    _CheckpointCallback.update_child_locals = call_counter_wrapper(_CheckpointCallback.update_child_locals)

    checkpoint_callback = _CheckpointCallback(save_freq=1000, save_path=log_folder)

    eval_env = gym.make(env_name)
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)

    eval_callback = EvalCallback(
        eval_env, callback_on_new_best=callback_on_best, best_model_save_path=log_folder, log_path=log_folder, eval_freq=100
    )
    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = _CheckpointCallback(save_freq=1, save_path=log_folder, name_prefix="event")

    # Instrument the class to count the number of calls
    _EveryNTimesteps = copy.deepcopy(EveryNTimesteps)
    _EveryNTimesteps.update_locals = call_counter_wrapper(_EveryNTimesteps.update_locals)
    _EveryNTimesteps.update_child_locals = call_counter_wrapper(_EveryNTimesteps.update_child_locals)
    
    event_callback = _EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

    # Instrument the class to count the number of calls
    _CallbackList = copy.deepcopy(CallbackList)
    _CallbackList.update_locals = call_counter_wrapper(_CallbackList.update_locals)
    _CallbackList.update_child_locals = call_counter_wrapper(_CallbackList.update_child_locals)

    callback = _CallbackList([checkpoint_callback, eval_callback, event_callback])
    model.learn(500, callback=callback)
    
    # ensure we call update locals the correct number of times.
    assert callback._update_locals_called == callback.n_calls == checkpoint_callback._update_locals_called == checkpoint_on_event._update_locals_called
    
    # ensure we update the child locals the correct number of times.
    assert callback._update_child_locals_called == callback.n_calls == checkpoint_callback._update_child_locals_called  == checkpoint_on_event._update_child_locals_called
    
    model.learn(500, callback=None)
    # Transform callback into a callback list automatically
    model.learn(500, callback=[checkpoint_callback, eval_callback])
    # Automatic wrapping, old way of doing callbacks
    model.learn(500, callback=lambda _locals, _globals: True)
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)

def select_env(model_class) -> str:
    if model_class is DQN:
        return "CartPole-v0"
    else:
        return "Pendulum-v0"
