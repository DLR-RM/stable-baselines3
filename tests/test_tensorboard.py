import os
from typing import Dict, Union

import pytest

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.utils import get_latest_run_id

MODEL_DICT = {
    "a2c": (A2C, "CartPole-v1"),
    "ppo": (PPO, "CartPole-v1"),
    "sac": (SAC, "Pendulum-v1"),
    "td3": (TD3, "Pendulum-v1"),
}

N_STEPS = 100


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict: Dict[str, Union[str, float]] = {
            "algorithm": self.model.__class__.__name__,
            # Ignore type checking for gamma, see https://github.com/DLR-RM/stable-baselines3/pull/1194/files#r1035006458
            "gamma": self.model.gamma,  # type: ignore[attr-defined]
        }
        if isinstance(self.model.learning_rate, float):  # Can also be Schedule, in that case, we don't report
            hparam_dict["learning rate"] = self.model.learning_rate
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict: Dict[str, float] = {
            "rollout/ep_len_mean": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


@pytest.mark.parametrize("model_name", MODEL_DICT.keys())
def test_tensorboard(tmp_path, model_name):
    # Skip if no tensorboard installed
    pytest.importorskip("tensorboard")

    logname = model_name.upper()
    algo, env_id = MODEL_DICT[model_name]
    kwargs = {}
    if model_name == "ppo":
        kwargs["n_steps"] = 64
    elif model_name in {"sac", "td3"}:
        kwargs["train_freq"] = 2
    model = algo("MlpPolicy", env_id, verbose=1, tensorboard_log=tmp_path, **kwargs)
    model.learn(N_STEPS, callback=HParamCallback())
    model.learn(N_STEPS, reset_num_timesteps=False)

    assert os.path.isdir(tmp_path / str(logname + "_1"))
    assert not os.path.isdir(tmp_path / str(logname + "_2"))

    logname = "tb_multiple_runs_" + model_name
    model.learn(N_STEPS, tb_log_name=logname)
    model.learn(N_STEPS, tb_log_name=logname)

    assert os.path.isdir(tmp_path / str(logname + "_1"))
    # Check that the log dir name increments correctly
    assert os.path.isdir(tmp_path / str(logname + "_2"))


def test_escape_log_name(tmp_path):
    # Log name that must be escaped
    log_name = "filename[16, 16]"
    # Create folder
    os.makedirs(str(tmp_path) + f"/{log_name}_1", exist_ok=True)
    os.makedirs(str(tmp_path) + f"/{log_name}_2", exist_ok=True)
    last_run_id = get_latest_run_id(tmp_path, log_name)
    assert last_run_id == 2
