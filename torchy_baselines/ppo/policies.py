from functools import partial

import torch as th
import torch.nn as nn
import numpy as np

from torchy_baselines.common.policies import BasePolicy, register_policy, MlpExtractor
from torchy_baselines.common.distributions import make_proba_distribution,\
    DiagGaussianDistribution, CategoricalDistribution, StateDependentNoiseDistribution


class PPOPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for A2C and derivates (PPO).

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_dim: (gym.spaces.Space) Action space
    :param learning_rate: (callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (nn.Module) Activation function
    :param adam_epsilon: (float) Small values to avoid NaN in ADAM optimizer
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    """
    def __init__(self, observation_space, action_space,
                 learning_rate, net_arch=None, device='cpu',
                 activation_fn=nn.Tanh, adam_epsilon=1e-5,
                 ortho_init=True, use_sde=False, log_std_init=0.0):
        super(PPOPolicy, self).__init__(observation_space, action_space, device)
        self.obs_dim = self.observation_space.shape[0]


        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

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
        # In the future, feature_extractor will be replaced with a CNN
        self.features_extractor = nn.Flatten()
        self.features_dim = self.obs_dim
        self.log_std_init = log_std_init
        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

        self._build(learning_rate)

    def reset_noise_net(self):
        self.action_dist.sample_weights(self.log_std)

    def _build(self, learning_rate):
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch,
                                          activation_fn=self.activation_fn, device=self.device)

        if isinstance(self.action_dist, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi,
                                                                                    log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            for module in [self.mlp_extractor, self.action_net, self.value_net]:
                # Values from stable-baselines check why
                gain = {
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
                    self.value_net: 1
                }[module]
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = th.optim.Adam(self.parameters(), lr=learning_rate(1), eps=self.adam_epsilon)

    def forward(self, obs, deterministic=False):
        if not isinstance(obs, th.Tensor):
            obs = th.FloatTensor(obs).to(self.device)
        latent_pi, latent_vf = self._get_latent(obs)
        value = self.value_net(latent_vf)
        action, action_distribution = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        return action, value, log_prob

    def _get_latent(self, obs):
        return self.mlp_extractor(self.features_extractor(obs))

    def _get_action_dist_from_latent(self, latent_pi, deterministic=False):
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, deterministic=deterministic)

        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(mean_actions, deterministic=deterministic)

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi, deterministic=deterministic)

    def actor_forward(self, obs, deterministic=False):
        latent_pi, _ = self._get_latent(obs)
        action, _ = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        return action.detach().cpu().numpy()

    def evaluate_actions(self, obs, action, deterministic=False):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param action: (th.Tensor)
        :param deterministic: (bool)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf = self._get_latent(obs)
        _, action_distribution = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        value = self.value_net(latent_vf)
        return value, log_prob, action_distribution.entropy()

    def value_forward(self, obs):
        _, latent_vf = self._get_latent(obs)
        return self.value_net(latent_vf)


MlpPolicy = PPOPolicy

register_policy("MlpPolicy", MlpPolicy)
