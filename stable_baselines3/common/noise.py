from typing import Optional, List, Iterable
from abc import ABC, abstractmethod
import copy

import numpy as np


class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self):
        super(ActionNoise, self).__init__()

    def reset(self) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class NormalActionNoise(ActionNoise):
    """
    A Gaussian action noise

    :param mean: (np.ndarray) the mean value of the noise
    :param sigma: (np.ndarray) the scale of the noise (std here)
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray):
        self._mu = mean
        self._sigma = sigma
        super(NormalActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)

    def __repr__(self) -> str:
        return f'NormalActionNoise(mu={self._mu}, sigma={self._sigma})'


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: (np.ndarray) the mean of the noise
    :param sigma: (np.ndarray) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    :param initial_noise: (Optional[np.ndarray]) the initial value for the noise output, (if None: 0)
    """

    def __init__(self, mean: np.ndarray,
                 sigma: np.ndarray,
                 theta: float = .15,
                 dt: float = 1e-2,
                 initial_noise: Optional[np.ndarray] = None):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        super(OrnsteinUhlenbeckActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:
        noise = (self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt
                 + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape))
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return f'OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})'


class VectorizedActionNoise(ActionNoise):
    """
    A Vectorized action noise for parallel environments.

    :param base_noise: ActionNoise The noise generator to use
    :param n_envs: (int) The number of parallel environments
    """

    def __init__(self, base_noise: ActionNoise, n_envs: int):
        super().__init__()
        self.n_envs: int = n_envs
        self.noises = [copy.deepcopy(base_noise) for _ in range(n_envs)]

    def reset(self, indices: Optional[Iterable[int]] = None) -> None:
        """
        Reset all the noise processes, or those listed in indices

        :param indices: Optional[Iterable[int]] The indices to reset. Default: None.
            If the parameter is None, then all processes are reset to their initial position.
        """
        if indices is None:
            indices = range(len(self.noises))

        for index in indices:
            self.noises[index].reset()

    def __repr__(self) -> str:
        base_repr = repr(self.noises[0]) if len(self.noises) else ""
        return f"VecNoise(BaseNoise={base_repr}), n_envs={len(self.noises)})"

    def __call__(self) -> np.ndarray:
        """
        Generate and stack the action noise from each noise object
        if n_envs == 0, returns a 0x0 matrix.
        """
        if len(self.noises):
            noise = np.stack([noise() for noise in self.noises])
        else:
            noise = np.zeros((0, 0))
        return noise

    @property
    def noises(self) -> List[ActionNoise]:
        return self._noises

    @noises.setter
    def noises(self, noises: List[ActionNoise]) -> None:
        self._noises = noises
        for noise in noises:
            noise.reset()
