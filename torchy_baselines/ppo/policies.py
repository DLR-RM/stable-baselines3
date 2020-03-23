from typing import Optional, List, Tuple, Callable, Union, Dict
from functools import partial

import gym
import torch as th
import torch.nn as nn
import numpy as np

from torchy_baselines.common.preprocessing import get_obs_dim
from torchy_baselines.common.policies import (BasePolicy, register_policy, MlpExtractor,
                                              create_sde_features_extractor)
from torchy_baselines.common.distributions import (make_proba_distribution, Distribution,
                                                   DiagGaussianDistribution, CategoricalDistribution,
                                                   StateDependentNoiseDistribution)


class PPOPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for A2C and derivates (PPO).

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (nn.Module) Activation function
    :param adam_epsilon: (float) Small values to avoid NaN in ADAM optimizer
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using SDE
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using SDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using SDE.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 device: Union[th.device, str] = 'cpu',
                 activation_fn: nn.Module = nn.Tanh,
                 adam_epsilon: float = 1e-5,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 normalize_images: bool = True):
        super(PPOPolicy, self).__init__(observation_space, action_space, device, squash_output=squash_output)

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.adam_epsilon = adam_epsilon
        self.ortho_init = ortho_init
        # In the future, feature_extractor will be replaced with a CNN
        self.features_extractor = nn.Flatten()
        self.features_dim = get_obs_dim(self.observation_space)
        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for SDE distribution
        if use_sde:
            dist_kwargs = {
                'full_std': full_std,
                'squash_output': squash_output,
                'use_expln': use_expln,
                'learn_features': sde_net_arch is not None
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs: (int)
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), 'reset_noise() is only available when using SDE'
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build(self, lr_schedule: Callable) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch,
                                          activation_fn=self.activation_fn, device=self.device)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for SDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(self.features_dim,
                                                                                        self.sde_net_arch,
                                                                                        self.activation_fn)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi,
                                                                                    log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi,
                                                                                    latent_sde_dim=latent_sde_dim,
                                                                                    log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

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
        # Setup optimizer with initial learning rate
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1), eps=self.adam_epsilon)

    def forward(self, obs: th.Tensor,
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        value = self.value_net(latent_vf)
        action, action_distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde,
                                                                        deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        return action, value, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: (th.Tensor) Observation
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) Latent codes
            for the actor, the value function and for SDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor,
                                     latent_sde: Optional[th.Tensor] = None,
                                     deterministic: bool = False) -> Tuple[th.Tensor, Distribution]:
        """
        Retrieve action and associated action distribution
        given the latent codes.

        :param latent_pi: (th.Tensor) Latent code for the actor
        :param latent_sde: (Optional[th.Tensor]) Latent code for the SDE exploration function
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, Distribution]) Action and action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, deterministic=deterministic)

        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(mean_actions, deterministic=deterministic)

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde,
                                                       deterministic=deterministic)
        else:
            raise ValueError('Invalid action distribution')

    def predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        action, _ = self._get_action_dist_from_latent(latent_pi, latent_sde, deterministic=deterministic)
        return action

    def evaluate_actions(self, obs: th.Tensor,
                         actions: th.Tensor,
                         deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :param deterministic: (bool)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        _, action_distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, deterministic=deterministic)
        log_prob = action_distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, action_distribution.entropy()


MlpPolicy = PPOPolicy

register_policy("MlpPolicy", MlpPolicy)
