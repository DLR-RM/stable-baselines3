from functools import partial

import torch as th
import torch.nn as nn
import numpy as np

from torchy_baselines.common.policies import BasePolicy, register_policy, MlpExtractor, create_mlp
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
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using SDE
    """
    def __init__(self, observation_space, action_space,
                 learning_rate, net_arch=None, device='cpu',
                 activation_fn=nn.Tanh, adam_epsilon=1e-5,
                 ortho_init=True, use_sde=False,
                 log_std_init=0.0, full_std=True, sde_net_arch=None):
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
        dist_kwargs = None
        # Keyword arguments for SDE distribution
        if use_sde:
            dist_kwargs = {
                'full_std': full_std,
                'squash_output': False,
                'use_expln': False,
                'learn_features': sde_net_arch is not None
            }

        self.sde_feature_extractor = None
        self.sde_net_arch = sde_net_arch

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(learning_rate)

    def reset_noise_net(self):
        """
        Sample new weights for the exploration matrix.
        """
        self.action_dist.sample_weights(self.log_std)

    def _build(self, learning_rate):
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch,
                                          activation_fn=self.activation_fn, device=self.device)

        # Separate feature extractor for SDE
        if self.sde_net_arch is not None:
            latent_sde = create_mlp(self.features_dim, -1, self.sde_net_arch,
                                    activation_fn=self.activation_fn, squash_out=False)
            self.sde_feature_extractor = nn.Sequential(*latent_sde)

        if isinstance(self.action_dist, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi,
                                                                                    log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = self.mlp_extractor.latent_dim_pi if self.sde_net_arch is None else self.sde_net_arch[-1]
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi,
                                                                                    latent_sde_dim=latent_sde_dim,
                                                                                    log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            for module in [self.mlp_extractor, self.action_net, self.value_net]:
                # Values from stable-baselines, TODO: check why
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
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        value = self.value_net(latent_vf)
        action, action_distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde,
                                                                        deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        return action, value, log_prob

    def _get_latent(self, obs):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Features for sde
        latent_sde = latent_pi
        if self.sde_feature_extractor is not None:
            latent_sde = self.sde_feature_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None, deterministic=False):
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, deterministic=deterministic)

        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(mean_actions, deterministic=deterministic)

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde, deterministic=deterministic)

    def actor_forward(self, obs, deterministic=False):
        latent_pi, _, latent_sde = self._get_latent(obs)
        action, _ = self._get_action_dist_from_latent(latent_pi, latent_sde, deterministic=deterministic)
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
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        _, action_distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        value = self.value_net(latent_vf)
        return value, log_prob, action_distribution.entropy()

    def value_forward(self, obs):
        _, latent_vf, _ = self._get_latent(obs)
        return self.value_net(latent_vf)


MlpPolicy = PPOPolicy

register_policy("MlpPolicy", MlpPolicy)
