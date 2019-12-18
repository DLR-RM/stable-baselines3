import pytest
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


N_ACTIONS = 1

# TODO: fix for num action > 1
# TODO: analytical form for squashed Gaussian?
@pytest.mark.parametrize("dist", [
    DiagGaussianDistribution(N_ACTIONS),
    StateDependentNoiseDistribution(N_ACTIONS, squash_output=False),
])
def test_entropy(dist):
    # The entropy can be approximated by averaging the negative log likelihood
    # mean negative log likelihood == differential entropy
    n_samples = int(5e6)
    n_features = 3
    set_random_seed(1)
    state = th.rand(n_samples, n_features)
    deterministic_actions = th.rand(n_samples, N_ACTIONS)
    _, log_std = dist.proba_distribution_net(n_features, log_std_init=th.log(th.tensor(0.2)))

    if isinstance(dist, DiagGaussianDistribution):
        actions, dist = dist.proba_distribution(deterministic_actions, log_std)
    else:
        dist.sample_weights(log_std, batch_size=n_samples)
        actions, dist = dist.proba_distribution(deterministic_actions, log_std, state)

    entropy = dist.entropy()
    log_prob = dist.log_prob(actions)
    assert th.allclose(entropy.mean(), -log_prob.mean(), rtol=5e-3)
