from typing import Optional, List, Tuple, Callable, Union, Type, Dict, Any

import gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import (BasePolicy, register_policy, create_mlp,
                                               create_sde_features_extractor, NatureCNN,
                                               BaseFeaturesExtractor, FlattenExtractor)
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: ([int]) Network architecture
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: (float) Clip the mean output when using gSDE to avoid numerical instability.
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
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
                 normalize_images: bool = True,
                 device: Union[th.device, str] = 'auto'):
        super(Actor, self).__init__(observation_space, action_space,
                                    features_extractor=features_extractor,
                                    normalize_images=normalize_images,
                                    device=device,
                                    squash_output=True)

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            latent_sde_dim = last_layer_dim
            # Separate feature extractor for gSDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(features_dim, sde_net_arch,
                                                                                            activation_fn)

            self.action_dist = StateDependentNoiseDistribution(action_dim, full_std=full_std, use_expln=use_expln,
                                                               learn_features=True, squash_output=True)
            self.mu, self.log_std = self.action_dist.proba_distribution_net(latent_dim=last_layer_dim,
                                                                            latent_sde_dim=latent_sde_dim,
                                                                            log_std_init=log_std_init)
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
            net_arch=self.net_arch,
            features_dim=self.features_dim,
            activation_fn=self.activation_fn,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            full_std=self.full_std,
            sde_net_arch=self.sde_net_arch,
            use_expln=self.use_expln,
            features_extractor=self.features_extractor,
            clip_mean=self.clip_mean
        ))
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return: (th.Tensor)
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), \
            'get_std() is only available when using gSDE'
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size: (int)
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), \
            'reset_noise() is only available when using gSDE'
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs: (th.Tensor)
        :return: (Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]])
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std,
                                                    deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


class Critic(BasePolicy):
    """
    Critic network (q-value function) for SAC.

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

        self.q_networks = [self.q1_net, self.q2_net]

    def forward(self, obs: th.Tensor, action: th.Tensor) -> List[th.Tensor]:
        # Learn the features extractor using the policy loss only
        # this is much faster
        with th.no_grad():
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, action], dim=1)
        return [q_net(qvalue_input) for q_net in self.q_networks]


class SACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: (float) Clip the mean output when using gSDE to avoid numerical instability.
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

    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super(SACPolicy, self).__init__(observation_space, action_space,
                                        device,
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        optimizer_class=optimizer_class,
                                        optimizer_kwargs=optimizer_kwargs,
                                        squash_output=True)

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [256, 256]
            else:
                net_arch = []

        # Create shared features extractor
        self.features_extractor = features_extractor_class(self.observation_space,
                                                           **self.features_extractor_kwargs)
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
            'sde_net_arch': sde_net_arch,
            'use_expln': use_expln,
            'clip_mean': clip_mean
        }
        self.actor_kwargs.update(sde_kwargs)
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None

        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1),
                                                    **self.optimizer_kwargs)

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Do not optimize the shared feature extractor with the critic loss
        # otherwise, there are gradient computation issues
        # Another solution: having duplicated features extractor but requires more memory and computation
        critic_parameters = [param for name, param in self.critic.named_parameters() if
                             'features_extractor' not in name]
        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1),
                                                     **self.optimizer_kwargs)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
            net_arch=self.net_args['net_arch'],
            activation_fn=self.net_args['activation_fn'],
            use_sde=self.actor_kwargs['use_sde'],
            log_std_init=self.actor_kwargs['log_std_init'],
            sde_net_arch=self.actor_kwargs['sde_net_arch'],
            use_expln=self.actor_kwargs['use_expln'],
            clip_mean=self.actor_kwargs['clip_mean'],
            lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs
        ))
        return data

    def make_actor(self) -> Actor:
        return Actor(**self.actor_kwargs).to(self.device)

    def make_critic(self) -> Critic:
        return Critic(**self.net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)


MlpPolicy = SACPolicy


class CnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: (float) Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
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
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
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
                                        use_sde,
                                        log_std_init,
                                        sde_net_arch,
                                        use_expln,
                                        clip_mean,
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        normalize_images,
                                        optimizer_class,
                                        optimizer_kwargs)


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
