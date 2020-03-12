import numpy as np
from typing import Tuple, Optional, List


# TODO: add more from https://github.com/hardmaru/estool/blob/master/es.py
# or https://github.com/facebookresearch/nevergrad

class CEM(object):
    """
    Cross-entropy method with diagonal covariance (separable CEM).

    :param num_params: (int) Number of parameters per individual (dimension of the problem)
    :param mu_init: (np.ndarray) Initial mean of the population distribution
        Taken to be zero if None is passed.
    :param sigma_init: (float) Initial standard deviation of the population distribution
    :param pop_size: (int) Number of individuals in the population
    :param damping_init: (float) Initial value of damping for preventing from early convergence.
    :param damping_final: (float) Final value of damping
    :param parents: (int) Number of parents used to compute the new distribution
        of individuals.
    :param elitism: (bool) Keep the best known individual in the population
    :param antithetic: (bool) Use a finite difference like method for sampling
        (mu + epsilon, mu - epsilon)
    """
    def __init__(self,
                 num_params: int,
                 mu_init: Optional[np.ndarray] = None,
                 sigma_init: float = 1e-3,
                 pop_size: int = 256,
                 damping_init: float = 1e-3,
                 damping_final: float = 1e-5,
                 parents: Optional[int] = None,
                 elitism: bool = False,
                 antithetic: bool = False):
        super(CEM, self).__init__()

        self.num_params = num_params

        # Distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)

        self.sigma = sigma_init
        # Damping parameters
        self.damping = damping_init
        self.damping_final = damping_final
        # Exponential moving average decay for damping
        self.tau = 0.95
        # Covariance matrix, here only the diagonal
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling parameters
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"

        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents

        # Weighting for computing the new mean of the distributions
        # from the parents. The better the individual, the higher the weight
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size: int) -> List[np.ndarray]:
        """
        Returns a list of candidates parameters

        :param pop_size: (int)
        :return: ([np.ndarray])
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        individuals = self.mu + epsilon * np.sqrt(self.cov)

        # Keep the best known individual in the population
        if self.elitism:
            individuals[-1] = self.elite

        return individuals

    def tell(self, solutions: List[np.ndarray], scores: List[float]) -> None:
        """
        Updates the distribution

        :param solutions: ([np.ndarray])
        :param scores: ([float]) episode reward.
        """
        # Convert rewards (we want to maximize) to cost (we want to minimize)
        scores = np.array(scores)
        scores *= -1
        # Sort the individuals by fitness
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        # Update damping using a moving average
        self.damping = self.damping * self.tau + (1 - self.tau) * self.damping_final
        # self.mu = self.weights @ solutions[idx_sorted[:self.parents]]
        self.mu = self.weights.dot(solutions[idx_sorted[:self.parents]])

        # CMA-ES style would be to use the new mean here
        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * self.weights.dot(z * z) + self.damping * np.ones(self.num_params)

        # Retrieve the best individual
        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]

    def get_distrib_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the parameters of the distribution:
        the mean and standard deviation.

        :return: (np.ndarray, np.ndarray)
        """
        return np.copy(self.mu), np.copy(self.cov)
