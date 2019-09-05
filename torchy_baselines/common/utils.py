import random

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
