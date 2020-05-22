from typing import Union, Optional

import numpy as np
from gym import Env, Space
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box


from stable_baselines3.common.type_aliases import GymStepReturn, GymObs


class IdentityEnv(Env):
    def __init__(self,
                 dim: Optional[int] = None,
                 space: Optional[Space] = None,
                 ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of ``dim`` and ``space``. If both are
            None, then initialization proceeds with ``dim=1`` and ``space=None``.
        :param space: the action and observation space. Provide at most one of
            ``dim`` and ``space``.
        :param ep_length: the length of each episode in timesteps
        """
        if space is None:
            if dim is None:
                dim = 1
            space = Discrete(dim)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        self.action_space = self.observation_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self) -> GymObs:
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def _get_reward(self, action: Union[int, np.ndarray]) -> float:
        return 1.0 if np.all(self.state == action) else 0.0

    def render(self, mode: str = 'human') -> None:
        pass


class IdentityEnvBox(IdentityEnv):
    def __init__(self, low: float = -1.0,
                 high: float = 1.0, eps: float = 0.05,
                 ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param low: (float) the lower bound of the box dim
        :param high: (float) the upper bound of the box dim
        :param eps: (float) the epsilon bound for correct value
        :param ep_length: (int) the length of each episode in timesteps
        """
        space = Box(low=low, high=high, shape=(1,), dtype=np.float32)
        super().__init__(ep_length=ep_length, space=space)
        self.eps = eps

    def step(self, action: np.ndarray) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _get_reward(self, action: np.ndarray) -> float:
        return 1.0 if (self.state - self.eps) <= action <= (self.state + self.eps) else 0.0


class IdentityEnvMultiDiscrete(IdentityEnv):
    def __init__(self, dim: int = 1, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: (int) the size of the dimensions you want to learn
        :param ep_length: (int) the length of each episode in timesteps
        """
        space = MultiDiscrete([dim, dim])
        super().__init__(ep_length=ep_length, space=space)


class IdentityEnvMultiBinary(IdentityEnv):
    def __init__(self, dim: int = 1, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: (int) the size of the dimensions you want to learn
        :param ep_length: (int) the length of each episode in timesteps
        """
        space = MultiBinary(dim)
        super().__init__(ep_length=ep_length, space=space)


class FakeImageEnv(Env):
    """
    Fake image environment for testing purposes, it mimics Atari games.

    :param action_dim: (int) Number of discrete actions
    :param screen_height: (int) Height of the image
    :param screen_width: (int) Width of the image
    :param n_channels: (int) Number of color channels
    :param discrete: (bool)
    """
    def __init__(self, action_dim: int = 6,
                 screen_height: int = 84,
                 screen_width: int = 84,
                 n_channels: int = 1,
                 discrete: bool = True):

        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width,
                                                             n_channels), dtype=np.uint8)
        if discrete:
            self.action_space = Discrete(action_dim)
        else:
            self.action_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.ep_length = 10
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        return self.observation_space.sample()

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        reward = 0.0
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.observation_space.sample(), reward, done, {}

    def render(self, mode: str = 'human') -> None:
        pass
