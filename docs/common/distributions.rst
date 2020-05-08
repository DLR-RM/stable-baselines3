.. _distributions:

Probability Distributions
=========================

Probability distributions used for the different action spaces:

- ``CategoricalDistribution`` -> Discrete
- ``DiagGaussianDistribution`` -> Box (continuous actions)
- ``StateDependentNoiseDistribution`` -> Box (continuous actions) when ``use_sde=True``

.. - ``MultiCategoricalDistribution`` -> MultiDiscrete
.. - ``BernoulliDistribution`` -> MultiBinary

The policy networks output parameters for the distributions (named ``flat`` in the methods).
Actions are then sampled from those distributions.

For instance, in the case of discrete actions. The policy network outputs probability
of taking each action. The ``CategoricalDistribution`` allows to sample from it,
computes the entropy, the log probability (``log_prob``) and backpropagate the gradient.

In the case of continuous actions, a Gaussian distribution is used. The policy network outputs
mean and (log) std of the distribution (assumed to be a ``DiagGaussianDistribution``).

.. automodule:: stable_baselines3.common.distributions
  :members:
