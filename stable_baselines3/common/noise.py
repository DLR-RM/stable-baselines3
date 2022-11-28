import copy
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np


class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self) -> None:
        super().__init__()

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

    :param mean: the mean value of the noise
    :param sigma: the scale of the noise (std here)
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray):
        self._mu = mean
        self._sigma = sigma
        super().__init__()

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: the mean of the noise
    :param sigma: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
    ):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        super().__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"


class VectorizedActionNoise(ActionNoise):
    """
    A Vectorized action noise for parallel environments.

    :param base_noise: ActionNoise The noise generator to use
    :param n_envs: The number of parallel environments
    """

    def __init__(self, base_noise: ActionNoise, n_envs: int):
        try:
            self.n_envs = int(n_envs)
            assert self.n_envs > 0
        except (TypeError, AssertionError) as e:
            raise ValueError(f"Expected n_envs={n_envs} to be positive integer greater than 0") from e

        self.base_noise = base_noise
        self.noises = [copy.deepcopy(self.base_noise) for _ in range(n_envs)]

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
        return f"VecNoise(BaseNoise={repr(self.base_noise)}), n_envs={len(self.noises)})"

    def __call__(self) -> np.ndarray:
        """
        Generate and stack the action noise from each noise object
        """
        noise = np.stack([noise() for noise in self.noises])
        return noise

    @property
    def base_noise(self) -> ActionNoise:
        return self._base_noise

    @base_noise.setter
    def base_noise(self, base_noise: ActionNoise) -> None:
        if base_noise is None:
            raise ValueError("Expected base_noise to be an instance of ActionNoise, not None", ActionNoise)
        if not isinstance(base_noise, ActionNoise):
            raise TypeError("Expected base_noise to be an instance of type ActionNoise", ActionNoise)
        self._base_noise = base_noise

    @property
    def noises(self) -> List[ActionNoise]:
        return self._noises

    @noises.setter
    def noises(self, noises: List[ActionNoise]) -> None:
        noises = list(noises)  # raises TypeError if not iterable
        assert len(noises) == self.n_envs, f"Expected a list of {self.n_envs} ActionNoises, found {len(noises)}."

        different_types = [i for i, noise in enumerate(noises) if not isinstance(noise, type(self.base_noise))]

        if len(different_types):
            raise ValueError(
                f"Noise instances at indices {different_types} don't match the type of base_noise", type(self.base_noise)
            )

        self._noises = noises
        for noise in noises:
            noise.reset()
