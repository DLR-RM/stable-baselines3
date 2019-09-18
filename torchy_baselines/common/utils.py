import random

import scipy.signal
import torch as th
import numpy as np


def set_random_seed(seed, using_cuda=False):
    """
    Seed the different random generators
    :param seed: (int)
    :param using_cuda: (bool)
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if using_cuda:
        # Make CuDNN Determinist
        th.backends.cudnn.deterministic = True
        th.cuda.manual_seed(seed)


# From stable_baselines.common.math_util
# def discount(vector, gamma):
#     """
#     computes discounted sums along 0th dimension of vector x.
#         y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
#                 where k = len(x) - t - 1
#
#     :param vector: (np.ndarray) the input vector
#     :param gamma: (float) the discount value
#     :return: (np.ndarray) the output vector
#     """
#     assert vector.ndim >= 1
#     return scipy.signal.lfilter([1], [1, -gamma], vector[::-1], axis=0)[::-1]


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
