import numpy as np
import torch as th

from torchy_baselines.common.distributions import DiagGaussianDistribution, SquashedDiagGaussianDistribution,\
    CategoricalDistribution, TanhBijector

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
