from typing import Optional, List, Tuple, Callable, Union

import gym
import torch as th
import torch.nn as nn

from torchy_baselines.common.preprocessing import get_action_dim, get_obs_dim
from torchy_baselines.common.policies import (BasePolicy, register_policy, create_mlp,
                                              create_sde_features_extractor)
from torchy_baselines.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution

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
    :param activation_fn: (nn.Module) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using SDE.
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using SDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using SDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: (float) Clip the mean output when using SDE to avoid numerical instability.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 net_arch: List[int],
                 features_extractor: nn.Module,
                 features_dim: int,
                 activation_fn: nn.Module = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
                 normalize_images: bool = True):
        super(Actor, self).__init__(observation_space, action_space,
                                    features_extractor=features_extractor,
                                    normalize_images=normalize_images)

        action_dim = get_action_dim(self.action_space)

        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        self.use_sde = use_sde
        self.sde_features_extractor = None

        if self.use_sde:
            latent_sde_dim = net_arch[-1]
            # Separate feature extractor for SDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(features_dim, sde_net_arch,
                                                                                            activation_fn)

            self.action_dist = StateDependentNoiseDistribution(action_dim, full_std=full_std, use_expln=use_expln,
                                                               learn_features=True, squash_output=True)
            self.mu, self.log_std = self.action_dist.proba_distribution_net(latent_dim=net_arch[-1],
                                                                            latent_sde_dim=latent_sde_dim,
                                                                            log_std_init=log_std_init)
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(net_arch[-1], action_dim)
            self.log_std = nn.Linear(net_arch[-1], action_dim)

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using SDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return: (th.Tensor)
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), \
            'get_std() is only available when using SDE'
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using SDE.

        :param batch_size: (int)
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), \
            'reset_noise() is only available when using SDE'
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        latent_sde = self.sde_features_extractor(features) if self.sde_features_extractor is not None else latent_pi
        return latent_pi, latent_sde

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_sde = self._get_latent(obs)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            log_std = self.log_std
        else:
            log_std = self.log_std(latent_pi)
            # Original Implementation to cap the standard deviation
            log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, latent_sde

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, latent_sde = self.get_action_dist_params(obs)
        kwargs = dict(latent_sde=latent_sde) if self.use_sde else {}
        # Note: the action is squashed
        action, _ = self.action_dist.proba_distribution(mean_actions, log_std,
                                                        deterministic=deterministic, **kwargs)
        return action

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, latent_sde = self.get_action_dist_params(obs)
        kwargs = dict(latent_sde=latent_sde) if self.use_sde else {}
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)


class Critic(BasePolicy):
    """
    Critic network (q-value function) for SAC.

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: ([int]) Network architecture
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param activation_fn: (nn.Module) Activation function
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 net_arch: List[int],
                 features_extractor: nn.Module,
                 features_dim: int,
                 activation_fn: nn.Module = nn.ReLU,
                 normalize_images: bool = True):
        super(Critic, self).__init__(observation_space, action_space,
                                     features_extractor=features_extractor,
                                     normalize_images=normalize_images)

        action_dim = get_action_dim(self.action_space)

        q1_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
        self.q1_net = nn.Sequential(*q1_net)

        q2_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
        self.q2_net = nn.Sequential(*q2_net)

        self.q_networks = [self.q1_net, self.q2_net]

    def forward(self, obs: th.Tensor, action: th.Tensor) -> List[th.Tensor]:
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
    :param activation_fn: (nn.Module) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using SDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using SDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: (float) Clip the mean output when using SDE to avoid numerical instability.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'cpu',
                 activation_fn: nn.Module = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
                 normalize_images: bool = True):
        super(SACPolicy, self).__init__(observation_space, action_space, device, squash_output=True)

        if net_arch is None:
            net_arch = [256, 256]

        # In the future, features_extractor will be replaced with a CNN
        self.features_extractor = nn.Flatten()
        self.features_dim = get_obs_dim(self.observation_space)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'features_extractor': self.features_extractor,
            'features_dim': self.features_dim,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn,
            'normalize_images': normalize_images
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
        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = th.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))

    def make_actor(self) -> Actor:
        return Actor(**self.actor_kwargs).to(self.device)

    def make_critic(self) -> Critic:
        return Critic(**self.net_args).to(self.device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.predict(obs, deterministic=False)

    def predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)


MlpPolicy = SACPolicy

register_policy("MlpPolicy", MlpPolicy)
