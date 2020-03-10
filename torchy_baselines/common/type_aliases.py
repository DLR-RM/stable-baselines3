"""
Common aliases for type hing
"""
from typing import Union, Type, Optional, Dict, Any, List, NamedTuple
from collections import namedtuple

import torch as th
import gym

from torchy_baselines.common.vec_env import VecEnv


GymEnv = Union[gym.Env, VecEnv]
TensorDict = Dict[str, th.Tensor]
OptimizerStateDict = Dict[str, Any]
# obs, action, old_values, old_log_prob, advantage, return_batch
class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


# obs, action, next_obs, done, reward
class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
