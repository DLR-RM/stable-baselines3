from typing import Optional, List, Tuple, Callable, Union, Type, Any, Dict

import gym
import torch as th
import torch.nn as nn

from torchy_baselines.common.preprocessing import get_action_dim
from torchy_baselines.common.policies import (BasePolicy, register_policy, create_mlp,
                                              create_sde_features_extractor, NatureCNN,
                                              BaseFeaturesExtractor, FlattenExtractor)
from torchy_baselines.common.distributions import StateDependentNoiseDistribution, Distribution


class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: ([int]) Network architecture
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param clip_noise: (float) Clip the magnitude of the noise
    :param lr_sde: (float) Learning rate for the standard deviation of the noise
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using SDE.
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using SDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using SDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param device: (Union[th.device, str]) Device on which the code should run.
    """
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 net_arch: List[int],
                 features_extractor: nn.Module,
                 features_dim: int,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 clip_noise: Optional[float] = None,
                 lr_sde: float = 3e-4,
                 full_std: bool = False,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 normalize_images: bool = True,
                 device: Union[th.device, str] = 'auto'):
        super(Actor, self).__init__(observation_space, action_space,
                                    features_extractor=features_extractor,
                                    normalize_images=normalize_images,
                                    device=device,
                                    squash_output=not use_sde)

        self.latent_pi, self.log_std = None, None
        self.weights_dist, self.exploration_mat = None, None
        self.use_sde, self.sde_optimizer = use_sde, None
        self.full_std = full_std
        self.sde_features_extractor = None
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.clip_noise = clip_noise
        self.lr_sde = lr_sde
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std

        action_dim = get_action_dim(self.action_space)

        if use_sde:
            latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn, squash_output=False)
            self.latent_pi = nn.Sequential(*latent_pi_net)
            latent_sde_dim = net_arch[-1]
            learn_features = sde_net_arch is not None

            # Separate feature extractor for SDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(features_dim, sde_net_arch,
                                                                                            activation_fn)

            # Create state dependent noise matrix (SDE)
            self.action_dist = StateDependentNoiseDistribution(action_dim, full_std=full_std, use_expln=use_expln,
                                                               squash_output=False, learn_features=learn_features)

            action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=net_arch[-1],
                                                                               latent_sde_dim=latent_sde_dim,
                                                                               log_std_init=log_std_init)
            # Squash output
            self.mu = nn.Sequential(action_net, nn.Tanh())
            self.sde_optimizer = th.optim.Adam([self.log_std], lr=lr_sde)
            self.reset_noise()
        else:
            actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
            self.mu = nn.Sequential(*actor_net)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
             net_arch=self.net_arch,
             features_dim=self.features_dim,
             activation_fn=self.activation_fn,
             use_sde=self.use_sde,
             log_std_init=self.log_std_init,
             clip_noise=self.clip_noise,
             lr_sde=self.lr_sde,
             full_std=self.full_std,
             sde_net_arch=self.sde_net_arch,
             use_expln=self.use_expln,
             features_extractor=self.features_extractor
        ))
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using SDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return: (th.Tensor)
        """
        return self.action_dist.get_std(self.log_std)

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        latent_sde = self.sde_features_extractor(features) if self.sde_features_extractor is not None else latent_pi
        return latent_pi, latent_sde

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations. Only useful when using SDE.

        :param obs: (th.Tensor)
        :param action: (th.Tensor)
        :return: (th.Tensor, th.Tensor) log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_sde = self._get_latent(obs)
        mean_actions = self.mu(latent_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        log_prob = distribution.log_prob(actions)
        return log_prob, distribution.entropy()

    def reset_noise(self) -> None:
        """
        Sample new weights for the exploration matrix, when using SDE.
        """
        self.action_dist.sample_weights(self.log_std)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if self.use_sde:
            latent_pi, latent_sde = self._get_latent(obs)
            if deterministic:
                return self.mu(latent_pi)

            noise = self.action_dist.get_noise(latent_sde)
            if self.clip_noise is not None:
                noise = th.clamp(noise, -self.clip_noise, self.clip_noise)
            # TODO: Replace with squashing -> need to account for that in the sde update
            # -> set squash_output=True in the action_dist?
            # NOTE: the clipping is done in the rollout for now
            return self.mu(latent_pi) + noise
        else:
            features = self.extract_features(obs)
            return self.mu(features)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic=deterministic)


class Critic(BasePolicy):
    """
    Critic network for TD3,
    in fact it represents the action-state value function (Q-value function)

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: ([int]) Network architecture
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param activation_fn: (Type[nn.Module]) Activation function
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param device: (Union[th.device, str]) Device on which the code should run.
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 net_arch: List[int],
                 features_extractor: nn.Module,
                 features_dim: int,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True,
                 device: Union[th.device, str] = 'auto'):
        super(Critic, self).__init__(observation_space, action_space,
                                     features_extractor=features_extractor,
                                     normalize_images=normalize_images,
                                     device=device)

        action_dim = get_action_dim(self.action_space)

        q1_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
        self.q1_net = nn.Sequential(*q1_net)

        q2_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
        self.q2_net = nn.Sequential(*q2_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # Learn the features extractor using the policy loss only
        with th.no_grad():
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return self.q1_net(qvalue_input), self.q2_net(qvalue_input)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q1_net(th.cat([features, actions], dim=1))


class ValueFunction(BasePolicy):
    """
    Value function for TD3 when doing on-policy exploration with SDE.

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param net_arch: (Optional[List[int]]) Network architecture
    :param activation_fn: (Type[nn.Module]) Activation function
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 features_extractor: nn.Module,
                 features_dim: int,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 normalize_images: bool = True):
        super(ValueFunction, self).__init__(observation_space, action_space,
                                            features_extractor=features_extractor,
                                            normalize_images=normalize_images)

        if net_arch is None:
            net_arch = [64, 64]

        vf_net = create_mlp(features_dim, 1, net_arch, activation_fn)
        self.vf_net = nn.Sequential(*vf_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs)
        return self.vf_net(features)


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (Union[th.device, str]) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using SDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using SDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 clip_noise: Optional[float] = None,
                 lr_sde: float = 3e-4,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super(TD3Policy, self).__init__(observation_space, action_space, device, squash_output=True)

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [400, 300]
            else:
                net_arch = []

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        self.features_extractor = features_extractor_class(self.observation_space, **features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'features_extractor': self.features_extractor,
            'features_dim': self.features_dim,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn,
            'normalize_images': normalize_images,
            'device': device
        }
        self.actor_kwargs = self.net_args.copy()
        sde_kwargs = {
            'use_sde': use_sde,
            'log_std_init': log_std_init,
            'clip_noise': clip_noise,
            'lr_sde': lr_sde,
            'sde_net_arch': sde_net_arch,
            'use_expln': use_expln
        }
        self.actor_kwargs.update(sde_kwargs)

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        # For SDE only
        self.use_sde = use_sde
        self.vf_net = None
        self.log_std_init = log_std_init
        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1),
                                                    **self.optimizer_kwargs)
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1),
                                                     **self.optimizer_kwargs)
        if self.use_sde:
            self.vf_net = ValueFunction(self.observation_space, self.action_space,
                                        features_extractor=self.features_extractor,
                                        features_dim=self.features_dim)
            self.actor.sde_optimizer.add_param_group({'params': self.vf_net.parameters()})  # pytype: disable=attribute-error

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
             net_arch=self.net_args['net_arch'],
             activation_fn=self.net_args['activation_fn'],
             use_sde=self.actor_kwargs['use_sde'],
             log_std_init=self.actor_kwargs['log_std_init'],
             clip_noise=self.actor_kwargs['clip_noise'],
             lr_sde=self.actor_kwargs['lr_sde'],
             sde_net_arch=self.actor_kwargs['sde_net_arch'],
             use_expln=self.actor_kwargs['use_expln'],
             lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
             optimizer=self.optimizer_class,
             optimizer_kwargs=self.optimizer_kwargs,
             features_extractor_class=self.features_extractor_class,
             features_extractor_kwargs=self.features_extractor_kwargs
        ))
        return data

    def reset_noise(self) -> None:
        return self.actor.reset_noise()

    def make_actor(self) -> Actor:
        return Actor(**self.actor_kwargs).to(self.device)

    def make_critic(self) -> Critic:
        return Critic(**self.net_args).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False):
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)


MlpPolicy = TD3Policy


class CnnPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (Union[th.device, str]) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using SDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using SDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 clip_noise: Optional[float] = None,
                 lr_sde: float = 3e-4,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super(CnnPolicy, self).__init__(observation_space,
                                        action_space,
                                        lr_schedule,
                                        net_arch,
                                        device,
                                        activation_fn,
                                        use_sde,
                                        log_std_init,
                                        clip_noise,
                                        lr_sde,
                                        sde_net_arch,
                                        use_expln,
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        normalize_images,
                                        optimizer,
                                        optimizer_kwargs)

register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
