from functools import partial

import torch as th
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

from torchy_baselines.common.policies import BasePolicy, register_policy, create_mlp


class PPOPolicy(BasePolicy):
    def __init__(self, observation_space, action_space,
                 learning_rate=1e-3, net_arch=None, device='cpu',
                 activation_fn=nn.Tanh, adam_epsilon=1e-5):
        super(PPOPolicy, self).__init__(observation_space, action_space, device)
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.adam_epsilon = adam_epsilon
        self.net_args = {
            'input_dim': self.state_dim,
            'output_dim': -1,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
        }
        self.shared_net = None
        self._build(learning_rate)

    @staticmethod
    def init_weights(module, gain=1):
        if type(module) == nn.Linear:
            nn.init.orthogonal_(module.weight, gain=gain)
            module.bias.data.fill_(0.0)

    def _build(self, learning_rate):
        # TODO: support non-shared network
        shared_net = create_mlp(self.state_dim, output_dim=-1, net_arch=self.net_arch, activation_fn=self.activation_fn)
        self.shared_net = nn.Sequential(*shared_net).to(self.device)
        self.actor_net = nn.Linear(self.net_arch[-1], self.action_dim)
        self.value_net = nn.Linear(self.net_arch[-1], 1)
        self.log_std = nn.Parameter(th.zeros(self.action_dim))
        # Init weights: use orthogonal initialization
        for module in [self.shared_net, self.actor_net, self.value_net]:
            gain = 0.01 if module == self.actor_net else 1.0
            # Values from stable-baselines check why
            gain = {
                self.shared_net: np.sqrt(2),
                self.actor_net: 0.01,
                self.value_net: 1
            }[module]
            module.apply(partial(self.init_weights, gain=gain))
        # TODO: support linear decay of the learning rate
        self.optimizer = th.optim.Adam(self.parameters(), lr=learning_rate, eps=self.adam_epsilon)

    def forward(self, state, deterministic=False):
        state = th.FloatTensor(state).to(self.device)
        latent = self.shared_net(state)
        value = self.value_net(latent)
        action, action_distribution = self._get_action_dist_from_latent(latent, deterministic=deterministic)
        log_prob = self._get_log_prob(action_distribution, action)
        return action, value, log_prob

    def _get_action_dist_from_latent(self, latent, deterministic=False):
        mean_actions = self.actor_net(latent)
        action_std = th.ones_like(mean_actions) * self.log_std.exp()
        action_distribution = Normal(mean_actions, action_std)
        # Sample from the gaussian
        if deterministic:
            action = mean_actions
        else:
            action = action_distribution.rsample()
        return action, action_distribution

    @staticmethod
    def _get_log_prob(action_distribution, action):
        log_prob = action_distribution.log_prob(action)
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(axis=1)
        else:
            log_prob = log_prob.sum()
        return log_prob

    def actor_forward(self, state, deterministic=False):
        latent = self.shared_net(state)
        action, _ = self._get_action_dist_from_latent(latent, deterministic=deterministic)
        return action.detach().cpu().numpy()

    def get_policy_stats(self, state, action):
        latent = self.shared_net(state)
        _, action_distribution = self._get_action_dist_from_latent(latent)
        log_prob = self._get_log_prob(action_distribution, action)
        value = self.value_net(latent)
        return value, log_prob, action_distribution.entropy()

    def value_forward(self):
        pass

MlpPolicy = PPOPolicy

register_policy("MlpPolicy", MlpPolicy)
