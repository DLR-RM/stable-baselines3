import numpy as np
import torch as th

from torchy_baselines.common.utils import set_random_seed
from torchy_baselines.common.distributions import DiagGaussianDistribution, SquashedDiagGaussianDistribution,\
    CategoricalDistribution, TanhBijector, StateDependentNoiseDistribution

# TODO: more tests for the other distributions
def test_bijector():
    """
    Test TanhBijector
    """
    actions = th.ones(5) * 2.0

    bijector = TanhBijector()

    squashed_actions = bijector.forward(actions)
    # Check that the boundaries are not violated
    assert th.max(th.abs(squashed_actions)) <= 1.0
    # Check the inverse method
    assert th.isclose(TanhBijector.inverse(squashed_actions), actions).all()


def test_sde_distribution():
    n_samples = int(5e6)
    n_features = 2
    n_actions = 1
    deterministic_actions = th.ones(n_samples, n_actions) * 0.1
    state = th.ones(n_samples, n_features) * 0.3
    dist = StateDependentNoiseDistribution(n_actions, full_std=True, squash_output=False)

    set_random_seed(1)
    _, log_std = dist.proba_distribution_net(n_features)
    dist.sample_weights(log_std, batch_size=n_samples)

    actions, _ = dist.proba_distribution(deterministic_actions, log_std, state)

    assert th.allclose(actions.mean(), dist.distribution.mean.mean(), rtol=1e-3)
    assert th.allclose(actions.std(), dist.distribution.scale.mean(), rtol=1e-3)
