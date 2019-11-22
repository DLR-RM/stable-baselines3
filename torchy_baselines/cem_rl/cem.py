import numpy as np

# TODO: add more from https://github.com/hardmaru/estool/blob/master/es.py
# or https://github.com/facebookresearch/nevergrad

class CEM(object):
    """
    Cross-entropy method with diagonal covariance (separable CEM).

    :param num_params: (int)
    :param mu_init: (np.ndarray) Initial mean of the population distribution
        Taken to be zero if None is passed.
    :param sigma_init: (float) Initial standard deviation of the population distribution
    :param pop_size: (int) Number of individuals in the population
    :param damp: (float) Damping for preventing from early convergence.
    :param damp_limit: (float) Final value of damping
    :param parents: (int)
    :param elitism: (bool)
    :param antithetic: (bool) Use a finite difference like method for sampling
        (mu + epsilon, mu - epsilon)
    """
    def __init__(self, num_params, mu_init=None, sigma_init=1e-3,
                 pop_size=256, damp=1e-3, damp_limit=1e-5,
                 parents=None, elitism=False, antithetic=False):
        super(CEM, self).__init__()
        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        # Exponential moving average decay for damping
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
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
        if self.elitism:
            individuals[-1] = self.elite

        return individuals

    def tell(self, solutions, scores):
        """
        Updates the distribution

        :param solutions: ([np.ndarray])
        :param scores: ([float]) episode reward.
        """
        # Convert rewards (we want to maximize) to cost (we want to minimize)
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        # self.mu = self.weights @ solutions[idx_sorted[:self.parents]]
        self.mu = self.weights.dot(solutions[idx_sorted[:self.parents]])

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * self.weights.dot(z * z) + self.damp * np.ones(self.num_params)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        # print(self.cov)

    def get_distrib_params(self):
        """
        Returns the parameters of the distribution:
        the mean and standard deviation.

        :return: (np.ndarray, np.ndarray)
        """
        return np.copy(self.mu), np.copy(self.cov)
