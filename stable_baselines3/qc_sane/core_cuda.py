import numpy as np
import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    print(sizes)
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPQFunction_quantile(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, quantiles):
        super().__init__()
        # print("create",[obs_dim + act_dim] + list(hidden_sizes) + [len(quantiles)])
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [len(quantiles)], activation
        )
        # self.out=mlp_quantile(quantiles)
        print("*q", self.q)

    def forward(self, obs, act):
        # print("pass Q_i/p",torch.cat([obs, act], dim=-1).shape)
        q = self.q(torch.cat([obs, act], dim=-1))
        # print("#",q.shape)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
