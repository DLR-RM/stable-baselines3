"""
Common aliases for type hing
"""
from typing import Union, Type, Optional, Dict, Any, List, Tuple

import torch as th
import gym

from torchy_baselines.common.vec_env import VecEnv


GymEnv = Union[gym.Env, VecEnv]
TensorDict = Dict[str, th.Tensor]
OptimizerStateDict = Dict[str, Any]
# obs, action, old_values, old_log_prob, advantage, return_batch
RolloutBufferSamples = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]
# obs, action, next_obs, done, reward
ReplayBufferSamples = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]
