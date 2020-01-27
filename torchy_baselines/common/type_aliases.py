"""
Common aliases for type hing
"""
from typing import Union, Type, Optional, Dict, Any, List, Tuple

import torch
import gym

from torchy_baselines.common.vec_env import VecEnv


GymEnv = Union[gym.Env, VecEnv]
TensorDict = Dict[str, torch.Tensor]
OptimizerStateDict = Dict[str, Any]
