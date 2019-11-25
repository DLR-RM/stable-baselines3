import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from gym import spaces

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

        :param other: ([float]) the distribution to compare with
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


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix,
    for continuous actions.

    :param action_dim: (int)  Number of continuous actions
    """
    def __init__(self, action_dim):
        super(DiagGaussianDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: (int) Dimension og the last layer of the policy (before the action layer)
        :param log_std_init: (float) Initial value for the log standard deviation
        :return: (nn.Linear, nn.Parameter)
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions, log_std, deterministic=False):
        """
        Create and sample for the distribution given its parameters (mean, std)

        :param mean_actions: (th.Tensor)
        :param log_std: (th.Tensor)
        :param deterministic: (bool)
        :return: (th.Tensor)
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        if deterministic:
            action = self.mode()
        else:
            action = self.sample()
        return action, self

    def mode(self):
        return self.distribution.mean

    def sample(self):
        return self.distribution.rsample()

    def entropy(self):
        return self.distribution.entropy()

    def log_prob_from_params(self, mean_actions, log_std):
        """
        Compute the log probabilty of taking an action
        given the distribution parameters.

        :param mean_actions: (th.Tensor)
        :param log_std: (th.Tensor)
        :return: (th.Tensor, th.Tensor)
        """
        action, _ = self.proba_distribution(mean_actions, log_std)
        log_prob = self.log_prob(action)
        return action, log_prob

    def log_prob(self, action):
        """
        Get the log probabilty of an action given a distribution.
        Note that you must call `proba_distribution()` method
        before.

        :param action: (th.Tensor)
        :return: (th.Tensor)
        """
        log_prob = self.distribution.log_prob(action)
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(axis=1)
        else:
            log_prob = log_prob.sum()
        return log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix,
    followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: (int) Number of continuous actions
    :param epsilon: (float) small value to avoid NaN due to numerical imprecision.
    """
    def __init__(self, action_dim, epsilon=1e-6):
        super(SquashedDiagGaussianDistribution, self).__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_action = None

    def proba_distribution(self, mean_actions, log_std, deterministic=False):
        action, _ = super(SquashedDiagGaussianDistribution, self).proba_distribution(mean_actions, log_std, deterministic)
        return action, self

    def mode(self):
        self.gaussian_action = self.distribution.mean
        # Squash the output
        return th.tanh(self.gaussian_action)

    def sample(self):
        self.gaussian_action = self.distribution.rsample()
        return th.tanh(self.gaussian_action)

    def log_prob_from_params(self, mean_actions, log_std):
        action, _ = self.proba_distribution(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_action)
        return action, log_prob

    def log_prob(self, action, gaussian_action=None):
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_action is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_action = TanhBijector.inverse(action)

        # Log likelihood for a gaussian distribution
        log_prob = super(SquashedDiagGaussianDistribution, self).log_prob(gaussian_action)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - action ** 2 + self.epsilon), dim=1)
        return log_prob


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: (int) Number of discrete actions
    """
    def __init__(self, action_dim):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim):
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilties using a softmax.

        :param latent_dim: (int) Dimension og the last layer of the policy (before the action layer)
        :return: (nn.Linear)
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits, deterministic=False):
        self.distribution = Categorical(logits=action_logits)
        if deterministic:
            action = self.mode()
        else:
            action = self.sample()
        return action, self

    def mode(self):
        return th.argmax(self.distribution.probs, dim=1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy()

    def log_prob_from_params(self, action_logits):
        action, _ = self.proba_distribution(action_logits)
        log_prob = self.log_prob(action)
        return action, log_prob

    def log_prob(self, action):
        log_prob = self.distribution.log_prob(action)
        return log_prob


class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using State Dependent Exploration (SDE).
    It is used to create the noise exploration matrix and
    compute the log probabilty of an action with that noise.

    :param action_dim: (int) Number of continuous actions
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: (bool) Use `expln()` function instead of `exp()` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, `exp()` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries.
    :param epsilon: (float) small value to avoid NaN due to numerical imprecision.
    """
    def __init__(self, action_dim, full_std=True, use_expln=False,
                 squash_output=False, epsilon=1e-6):
        super(StateDependentNoiseDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.latent_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        if squash_output:
            print("== Using TanhBijector ===")
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std):
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std: (th.Tensor)
        :return: (th.Tensor)
        """
        if self.use_expln:
            # From SDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            if log_std <= 0:
                std = th.exp(log_std)
            else:
                std = th.log(log_std + 1.0) + 1.0
        else:
            # Use normal exponential
            std = th.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return th.ones((self.latent_dim, self.action_dim)).to(log_std.device) * std

    def sample_weights(self, log_std):
        """
        Sample weights for the noise exploration matrix,
        using a centered gaussian distribution.

        :param log_std: (th.Tensor)
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        self.exploration_mat = self.weights_dist.rsample()

    def proba_distribution_net(self, latent_dim, log_std_init=-2.0):
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: (int) Dimension og the last layer of the policy (before the action layer)
        :param log_std_init: (float) Initial value for the log standard deviation
        :return: (nn.Linear, nn.Parameter)
        """
        # Network for the deterministic action, it represents the mean of the distribution
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)

        self.latent_dim = latent_dim
        # Reduce the number of parameters if needed
        log_std = th.ones(latent_dim, self.action_dim) if self.full_std else th.ones(latent_dim, 1)
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(self, mean_actions, log_std, latent_pi, deterministic=False):
        """
        Create and sample for the distribution given its parameters (mean, std)

        :param mean_actions: (th.Tensor)
        :param log_std: (th.Tensor)
        :param deterministic: (bool)
        :return: (th.Tensor)
        """
        variance = th.mm(latent_pi.detach() ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))

        if deterministic:
            action = self.mode()
        else:
            action = self.sample(latent_pi)
        return action, self

    def mode(self):
        action = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(action)
        return action

    def get_noise(self, latent_pi):
        return th.mm(latent_pi.detach(), self.exploration_mat)

    def sample(self, latent_pi):
        noise = self.get_noise(latent_pi)
        action = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(action)
        return action

    def entropy(self):
        # TODO: account for the squashing?
        return self.distribution.entropy()

    def log_prob_from_params(self, mean_actions, log_std, latent_pi):
        action, _ = self.proba_distribution(mean_actions, log_std, latent_pi)
        log_prob = self.log_prob(action)
        return action, log_prob

    def log_prob(self, action):
        if self.bijector is not None:
            gaussian_action = self.bijector.inverse(action)
        else:
            gaussian_action = action
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_action)

        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(axis=1)
        else:
            log_prob = log_prob.sum()

        if self.bijector is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= th.sum(self.bijector.log_prob_correction(gaussian_action), dim=1)
        return log_prob


class TanhBijector(object):
    """
    Bijective transformation of a probabilty distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: (float) small value to avoid NaN due to numerical imprecision.
    """
    def __init__(self, epsilon=1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return th.tanh(x)

    @staticmethod
    def atanh(x):
        """
        Inverse of Tanh

        Taken from pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y):
        """
        Inverse tanh.

        :param y: (th.Tensor)
        :return: (th.Tensor)
        """
        eps = th.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1. + eps, max=1. - eps))

    def log_prob_correction(self, x):
        # Squash correction (from original SAC implementation)
        return th.log(1 - th.tanh(x) ** 2 + self.epsilon)


def make_proba_distribution(action_space, use_sde=False, dist_kwargs=None):
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: (Gym Space) the input action space
    :param use_sde: (bool) Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: (dict) Keyword arguments to pass to the probabilty distribution
    :return: (Distribution) the approriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        if use_sde:
            return StateDependentNoiseDistribution(action_space.shape[0], **dist_kwargs)
        return DiagGaussianDistribution(action_space.shape[0], **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    # elif isinstance(action_space, spaces.MultiDiscrete):
    #     return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    # elif isinstance(action_space, spaces.MultiBinary):
    #     return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(action_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")
