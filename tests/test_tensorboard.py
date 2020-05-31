import os
import shutil

import pytest

from stable_baselines3 import A2C, PPO, SAC, TD3

TENSORBOARD_DIR = '/tmp/tb_dir/'

if os.path.isdir(TENSORBOARD_DIR):
    shutil.rmtree(TENSORBOARD_DIR)

MODEL_DICT = {
    'a2c': (A2C, 'CartPole-v1'),
    'ppo': (PPO, 'CartPole-v1'),
    'sac': (SAC, 'Pendulum-v0'),
    'td3': (TD3, 'Pendulum-v0'),
}

N_STEPS = 1000


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_tensorboard(model_name):
    logname = model_name.upper()
    algo, env_id = MODEL_DICT[model_name]
    model = algo('MlpPolicy', env_id, verbose=1, tensorboard_log=TENSORBOARD_DIR)
    model.learn(N_STEPS)
    model.learn(N_STEPS, reset_num_timesteps=False)

    assert os.path.isdir(TENSORBOARD_DIR + logname + "_1")
    assert not os.path.isdir(TENSORBOARD_DIR + logname + "_2")

@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_multiple_runs(model_name):
    logname = "tb_multiple_runs_" + model_name
    algo, env_id = MODEL_DICT[model_name]
    model = algo('MlpPolicy', env_id, verbose=1, tensorboard_log=TENSORBOARD_DIR)
    model.learn(N_STEPS, tb_log_name=logname)
    model.learn(N_STEPS, tb_log_name=logname)

    assert os.path.isdir(TENSORBOARD_DIR + logname + "_1")
    # Check that the log dir name increments correctly
    assert os.path.isdir(TENSORBOARD_DIR + logname + "_2")