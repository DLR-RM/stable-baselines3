import os

import pytest

from stable_baselines3 import A2C, PPO, SAC, TD3

MODEL_DICT = {
    "a2c": (A2C, "CartPole-v1"),
    "ppo": (PPO, "CartPole-v1"),
    "sac": (SAC, "Pendulum-v0"),
    "td3": (TD3, "Pendulum-v0"),
}

N_STEPS = 100


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_tensorboard(tmp_path, model_name):
    # Skip if no tensorboard installed
    pytest.importorskip("tensorboard")

    logname = model_name.upper()
    algo, env_id = MODEL_DICT[model_name]
    model = algo("MlpPolicy", env_id, verbose=1, tensorboard_log=tmp_path)
    model.learn(N_STEPS)
    model.learn(N_STEPS, reset_num_timesteps=False)

    assert os.path.isdir(tmp_path / str(logname + "_1"))
    assert not os.path.isdir(tmp_path / str(logname + "_2"))

    logname = "tb_multiple_runs_" + model_name
    model.learn(N_STEPS, tb_log_name=logname)
    model.learn(N_STEPS, tb_log_name=logname)

    assert os.path.isdir(tmp_path / str(logname + "_1"))
    # Check that the log dir name increments correctly
    assert os.path.isdir(tmp_path / str(logname + "_2"))
