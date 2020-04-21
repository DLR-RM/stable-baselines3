"""
Common aliases for type hint
"""
from typing import Union, Dict, Any, NamedTuple, Optional, List, Callable

import numpy as np
import torch as th
import gym

from torchy_baselines.common.vec_env import VecEnv
from torchy_baselines.common.callbacks import BaseCallback


GymEnv = Union[gym.Env, VecEnv]
TensorDict = Dict[str, th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[BaseCallback], BaseCallback]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class RolloutReturn(NamedTuple):
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool
