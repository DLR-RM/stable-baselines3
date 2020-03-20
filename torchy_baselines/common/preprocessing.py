from typing import Tuple, Union

import numpy as np
import torch as th
from gym import spaces


def is_image(observation_space):
    return False


def preprocess_obs(obs: th.Tensor, observation_space: spaces.Space) -> th.Tensor:
    if isinstance(observation_space, spaces.Box):
        if is_image(observation_space):
            return obs / 255.0
        return obs
    elif isinstance(observation_space, spaces.Discrete):
        # TODO: one hot encoding
        raise NotImplementedError()
    else:
        # TODO: Multidiscrete, Binary, MultiBinary, Tuple, Dict
        raise NotImplementedError()


def get_obs_dim(observation_space: spaces.Space) -> Union[int, Tuple[int, ...]]:
    if isinstance(observation_space, spaces.Box):
        if is_image(observation_space):
            return observation_space.shape
        return np.prod(observation_space.shape)
    elif isinstance(observation_space, spaces.Discrete):
        return 1
    else:
        # TODO: Multidiscrete, Binary, MultiBinary, Tuple, Dict
        raise NotImplementedError()


def get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    else:
        raise NotImplementedError()
