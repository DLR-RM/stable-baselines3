import torch as th
from torch.distributions import Normal


class Distribution(object):
    def __init__(self):
        super(Distribution, self).__init__()

    def log_prob(self, x):
        """
        returns the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        raise NotImplementedError

    def kl_div(self, other):
        """
        Calculates the Kullback-Leibler divergence from the given probabilty distribution

        :param other: ([float]) the distibution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probabilty distribution

        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError


class DiagGaussianDistribution(object):
    """docstring for DiagGaussianDistribution."""

    def __init__(self):
        super(DiagGaussianDistribution, self).__init__()
        self.distribution = None

    def proba_distribution_from_latent(self, latent, init_scale=1.0, init_bias=0.0):
        self.distribution = Normal()

    def sample(self):
        return self.distribution.rsample()
