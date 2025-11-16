import random
import numpy as np
import torch

from stable_baselines3.common.utils import set_random_seed


def test_set_random_seed_reproducibility():
    """
    Ensure that calling set_random_seed with the same seed produces identical
    outputs from random, numpy and torch across repeated calls.
    """
    seed = 12345

    set_random_seed(seed)
    r1 = random.random()
    n1 = np.random.rand(3)
    t1 = torch.randn(3)

    # Reset with the same seed and sample again
    set_random_seed(seed)
    r2 = random.random()
    n2 = np.random.rand(3)
    t2 = torch.randn(3)

    assert r1 == r2, "Python random produced different results for identical seed"
    assert np.allclose(n1, n2), "NumPy produced different results for identical seed"
    assert torch.allclose(t1, t2), "Torch produced different results for identical seed"


def test_set_random_seed_differs_for_different_seeds():
    """
    Ensure that using different seeds results in different sequences (basic sanity check).
    """
    set_random_seed(1)
    a1 = np.random.rand(5)

    set_random_seed(2)
    a2 = np.random.rand(5)

    # It's extremely unlikely two different seeds produce the exact same float arrays
    assert not np.allclose(a1, a2), "Different seeds produced identical NumPy sequences"