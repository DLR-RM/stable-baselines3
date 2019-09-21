import random

import numpy as np
import torch as th


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


# From stable baselines
def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
