import pytest

from torchy_baselines import SAC
from torchy_baselines.common.callbacks import (CallbackList, CheckpointCallback, EvalCallback,
    EveryNTimesteps, StopTrainingOnRewardThreshold)


@pytest.mark.parametrize("model_class", [SAC])
def test_callbacks(model_class):
    # Create RL model
    model = model_class('MlpPolicy', 'Pendulum-v0')

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')

    # For testing: use the same training env
    eval_env = model.get_env()
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)

    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best,
                                 best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=100)

    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/',
                                             name_prefix='event')
    event_callback = EveryNTimesteps(n_steps=1000, callback=checkpoint_on_event)

    callback = CallbackList([checkpoint_callback, eval_callback, event_callback])

    model.learn(1000, callback=callback)
    model.learn(500, callback=None)
    # Transform callback into a callback list automatically
    model.learn(500, callback=[checkpoint_callback, eval_callback])
    # Automatic wrapping, old way of doing callbacks
    model.learn(500, callback=lambda _locals, _globals : True)
