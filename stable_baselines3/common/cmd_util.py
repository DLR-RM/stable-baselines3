import warnings

from stable_baselines3.common.env_util import *  # noqa: F403,F401

warnings.warn(
    "Module ``common.cmd_util`` has been renamed to ``common.env_util`` and will be removed in the future.", FutureWarning
)
