from functools import partial

import torch as th
import torch.nn as nn
import numpy as np

from torchy_baselines.common.policies import BasePolicy, register_policy, create_mlp
from torchy_baselines.common.distributions import DiagGaussianDistribution, SquashedDiagGaussianDistribution


class PPOPolicy(BasePolicy):
    def __init__(self, observation_space, action_space,
                 learning_rate=1e-3, net_arch=None, device='cpu',
                 activation_fn=nn.Tanh, adam_epsilon=1e-5, ortho_init=True):
        super(PPOPolicy, self).__init__(observation_space, action_space, device)
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.adam_epsilon = adam_epsilon
        self.ortho_init = ortho_init
        self.net_args = {
            'input_dim': self.obs_dim,
            'output_dim': -1,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
        }
        self.shared_net = None
        self.pi_net, self.vf_net = None, None
        # Action distribution
        self.action_dist = DiagGaussianDistribution(self.action_dim)
        # self.action_dist = SquashedDiagGaussianDistribution(self.action_dim)
        self._build(learning_rate)

    def _build(self, learning_rate):
        # TODO: support shared network
        # shared_net = create_mlp(self.obs_dim, output_dim=-1, net_arch=self.net_arch, activation_fn=self.activation_fn)
        # self.shared_net = nn.Sequential(*shared_net).to(self.device)

        pi_net = create_mlp(self.obs_dim, output_dim=-1, net_arch=self.net_arch, activation_fn=self.activation_fn)
        self.pi_net = nn.Sequential(*pi_net).to(self.device)
        vf_net = create_mlp(self.obs_dim, output_dim=-1, net_arch=self.net_arch, activation_fn=self.activation_fn)
        self.vf_net = nn.Sequential(*vf_net).to(self.device)

        # self.action_net = nn.Linear(self.net_arch[-1], self.action_dim)
        # self.log_std = nn.Parameter(th.zeros(self.action_dim))
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.net_arch[-1])
        self.value_net = nn.Linear(self.net_arch[-1], 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            for module in [self.pi_net, self.vf_net, self.action_net, self.value_net]:
                # Values from stable-baselines check why
                gain = {
                    self.pi_net: np.sqrt(2),
                    self.vf_net: np.sqrt(2),
                    self.shared_net: np.sqrt(2),
                    self.action_net: 0.01,
                    self.value_net: 1
                }[module]
                module.apply(partial(self.init_weights, gain=gain))
        # TODO: support linear decay of the learning rate
        self.optimizer = th.optim.Adam(self.parameters(), lr=learning_rate, eps=self.adam_epsilon)

    def forward(self, obs, deterministic=False):
        if not isinstance(obs, th.Tensor):
            obs = th.FloatTensor(obs).to(self.device)
        latent_pi, latent_vf = self._get_latent(obs)
        value = self.value_net(latent_vf)
        action, action_distribution = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        return action, value, log_prob

    def _get_latent(self, obs):
        if self.shared_net is not None:
            latent = self.shared_net(obs)
            return latent, latent
        else:
            return self.pi_net(obs), self.vf_net(obs)

    def _get_action_dist_from_latent(self, latent, deterministic=False):
        mean_actions = self.action_net(latent)
        return self.action_dist.proba_distribution(mean_actions, self.log_std, deterministic=deterministic)

    def actor_forward(self, obs, deterministic=False):
        latent_pi, _ = self._get_latent(obs)
        action, _ = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        return action.detach().cpu().numpy()

    def get_policy_stats(self, obs, action):
        latent_pi, latent_vf = self._get_latent(obs)
        _, action_distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = action_distribution.log_prob(action)
        value = self.value_net(latent_vf)
        return value, log_prob, action_distribution.entropy()

    def value_forward(self):
        pass


MlpPolicy = PPOPolicy

register_policy("MlpPolicy", MlpPolicy)
