import torch as th
import torch.nn as nn
from torch.distributions import Normal

from torchy_baselines.common.policies import BasePolicy, register_policy, create_mlp


class PPOPolicy(BasePolicy):
    def __init__(self, observation_space, action_space,
                 learning_rate=1e-3, net_arch=None, device='cpu',
                 activation_fn=nn.Tanh):
        super(PPOPolicy, self).__init__(observation_space, action_space, device)
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            'input_dim': self.state_dim,
            'output_dim': -1,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
        }
        self.shared_net = None
        self._build(learning_rate)

    def _build(self, learning_rate):
        shared_net = create_mlp(self.state_dim, output_dim=-1, net_arch=self.net_arch, activation_fn=self.activation_fn)
        self.shared_net = nn.Sequential(*shared_net).to(self.device)
        self.actor_net = nn.Linear(self.net_arch[-1], self.action_dim)
        self.value_net = nn.Linear(self.net_arch[-1], 1)
        self.log_std = nn.Parameter(th.zeros(self.action_dim, 1))
        self.optimizer = th.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        state = th.FloatTensor(state).to(self.device)
        latent = self.shared_net(state)
        # TODO: initialize pi_mean weights properly
        mean_actions = self.actor_net(latent)
        action_distribution = Normal(mean_actions, self.log_std)
        # Sample from the gaussian
        action = action_distribution.rsample()
        log_prob = action_distribution.log_prob(action)
        # entropy = action_distribution.entropy()
        value = self.value_net(latent)
        return action, value, log_prob

    def actor_forward(self, state):
        latent = self.shared_net(state)
        # TODO: initialize pi_mean weights properly
        mean_actions = self.actor_net(latent)
        action_distribution = Normal(mean_actions, self.log_std)
        # Sample from the gaussian
        action = action_distribution.rsample()
        return action

    def value_forward(self):
        pass

MlpPolicy = PPOPolicy

register_policy("MlpPolicy", MlpPolicy)
