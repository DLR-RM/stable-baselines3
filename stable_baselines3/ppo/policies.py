from typing import Optional, List, Tuple, Callable, Union, Dict, Type, Any
from functools import partial

import gym
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3.common.policies import (BasePolicy, register_policy, MlpExtractor,
                                               create_sde_features_extractor, NatureCNN,
                                               BaseFeaturesExtractor, FlattenExtractor)
from stable_baselines3.common.distributions import (make_proba_distribution, Distribution,
                                                    DiagGaussianDistribution, CategoricalDistribution,
                                                    MultiCategoricalDistribution, BernoulliDistribution,
                                                    StateDependentNoiseDistribution)


class PPOPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for A2C and derivates (PPO).

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in ADAM optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs['eps'] = 1e-5

        super(PPOPolicy, self).__init__(observation_space, action_space,
                                        device,
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        optimizer_class=optimizer_class,
                                        optimizer_kwargs=optimizer_kwargs,
                                        squash_output=squash_output)

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space,
                                                           **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
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
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            squash_output=self.dist_kwargs['squash_output'] if self.dist_kwargs else None,
            full_std=self.dist_kwargs['full_std'] if self.dist_kwargs else None,
            sde_net_arch=self.dist_kwargs['sde_net_arch'] if self.dist_kwargs else None,
            use_expln=self.dist_kwargs['use_expln'] if self.dist_kwargs else None,
            lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            ortho_init=self.ortho_init,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs
        ))
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs: (int)
        """
        assert isinstance(self.action_dist,
                          StateDependentNoiseDistribution), 'reset_noise() is only available when using gSDE'
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

        # Separate feature extractor for gSDE
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
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            for module in [self.features_extractor, self.mlp_extractor,
                           self.action_net, self.value_net]:
                # Values from stable-baselines, TODO: check why
                gain = {
                    self.features_extractor: np.sqrt(2),
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
                    self.value_net: 1
                }[module]
                module.apply(partial(self.init_weights, gain=gain))
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

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
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: (th.Tensor) Observation
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) Latent codes
            for the actor, the value function and for gSDE function
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
                                     latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: (th.Tensor) Latent code for the actor
        :param latent_sde: (Optional[th.Tensor]) Latent code for the gSDE exploration function
        :return: (Distribution) Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError('Invalid action distribution')

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor,
                         actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


MlpPolicy = PPOPolicy


class CnnPolicy(PPOPolicy):
    """
    CnnPolicy class (with both actor and critic) for A2C and derivates (PPO).

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super(CnnPolicy, self).__init__(observation_space,
                                        action_space,
                                        lr_schedule,
                                        net_arch,
                                        device,
                                        activation_fn,
                                        ortho_init,
                                        use_sde,
                                        log_std_init,
                                        full_std,
                                        sde_net_arch,
                                        use_expln,
                                        squash_output,
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        normalize_images,
                                        optimizer_class,
                                        optimizer_kwargs)


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
