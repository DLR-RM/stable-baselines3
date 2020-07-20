from typing import List, Optional, Sequence, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env import VecEnv


class ObsWrapper(VecEnv):
    """
    Wrapper for a VecEnv which overrides the observation space for Hindsight Experience Replay to support dict observations.

    :param env: (VecEnv) The vectorized environment to wrap.
    """

    def __init__(self, venv: VecEnv):
        super(ObsWrapper, self).__init__(
            num_envs=venv.num_envs, observation_space=venv.observation_space, action_space=venv.action_space
        )

        self.venv = venv

        self.spaces = list(venv.observation_space.spaces.values())

        # get dimensions of observation and goal
        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = venv.observation_space.spaces["achieved_goal"].shape
            self.obs_dim = venv.observation_space.spaces["observation"].shape[0]
            self.goal_dim = goal_space_shape[0]

        # new observation space with concatenated observation and (desired) goal
        # for the different types of spaces
        if isinstance(self.spaces[0], spaces.Box):
            low_values = np.concatenate(
                [venv.observation_space["observation"].low, venv.observation_space["desired_goal"].low]
            )
            high_values = np.concatenate(
                [venv.observation_space["observation"].high, venv.observation_space["desired_goal"].high]
            )
            self.observation_space = spaces.Box(low_values, high_values, dtype=np.float32)
        elif isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)
        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [venv.observation_space.spaces["observation"].n, venv.observation_space.spaces["desired_goal"].n]
            self.observation_space = spaces.MultiDiscrete(dimensions)
        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        return self.venv.close()

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def get_images(self) -> Sequence[np.ndarray]:
        return self.venv.get_images()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.venv.seed(seed)
