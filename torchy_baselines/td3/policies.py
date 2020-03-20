from typing import Optional, List, Tuple, Callable, Union

import gym
import torch as th
import torch.nn as nn

from torchy_baselines.common.preprocessing import get_action_dim, get_obs_dim
from torchy_baselines.common.policies import (BasePolicy, register_policy, create_mlp, BaseNetwork,
                                              create_sde_feature_extractor)
from torchy_baselines.common.distributions import StateDependentNoiseDistribution


class Actor(BaseNetwork):
    """
    Actor network (policy) for TD3.

    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param net_arch: ([int]) Network architecture
    :param activation_fn: (nn.Module) Activation function
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
    """
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 net_arch: List[int],
                 activation_fn: nn.Module = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 clip_noise: Optional[float] = None,
                 lr_sde: float = 3e-4,
                 full_std: bool = False,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False):
        super(Actor, self).__init__()

        self.latent_pi, self.log_std = None, None
        self.weights_dist, self.exploration_mat = None, None
        self.use_sde, self.sde_optimizer = use_sde, None
        self.action_dim = action_dim
        self.full_std = full_std
        self.sde_feature_extractor = None

        if use_sde:
            latent_pi_net = create_mlp(obs_dim, -1, net_arch, activation_fn, squash_output=False)
            self.latent_pi = nn.Sequential(*latent_pi_net)
            latent_sde_dim = net_arch[-1]
            learn_features = sde_net_arch is not None

            # Separate feature extractor for SDE
            if sde_net_arch is not None:
                self.sde_feature_extractor, latent_sde_dim = create_sde_feature_extractor(obs_dim, sde_net_arch,
                                                                                          activation_fn)

            # Create state dependent noise matrix (SDE)
            self.action_dist = StateDependentNoiseDistribution(action_dim, full_std=full_std, use_expln=use_expln,
                                                               squash_output=False, learn_features=learn_features)
            action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=net_arch[-1],
                                                                               latent_sde_dim=latent_sde_dim,
                                                                               log_std_init=log_std_init)
            # Squash output
            self.mu = nn.Sequential(action_net, nn.Tanh())
            self.clip_noise = clip_noise
            self.sde_optimizer = th.optim.Adam([self.log_std], lr=lr_sde)
            self.reset_noise()
        else:
            actor_net = create_mlp(obs_dim, action_dim, net_arch, activation_fn, squash_output=True)
            self.mu = nn.Sequential(*actor_net)

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

    def _get_action_dist_from_latent(self, latent_pi, latent_sde):
        mean_actions = self.mu(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)

    def _get_latent(self, obs) -> Tuple[th.Tensor, th.Tensor]:
        latent_pi = self.latent_pi(obs)

        if self.sde_feature_extractor is not None:
            latent_sde = self.sde_feature_extractor(obs)
        else:
            latent_sde = latent_pi
        return latent_pi, latent_sde

    def evaluate_actions(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations. Only useful when using SDE.

        :param obs: (th.Tensor)
        :param action: (th.Tensor)
        :return: (th.Tensor, th.Tensor) log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_sde = self._get_latent(obs)
        _, distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(action)
        # value = self.value_net(latent_vf)
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
            # action, _ = self._get_action_dist_from_latent(latent_pi)
            # return action
        else:
            return self.mu(obs)


class Critic(BaseNetwork):
    """
    Critic network for TD3,
    in fact it represents the action-state value function (Q-value function)

    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param net_arch: ([int]) Network architecture
    :param activation_fn: (nn.Module) Activation function
    """
    def __init__(self, obs_dim: int, action_dim: int,
                 net_arch: List[int], activation_fn: nn.Module = nn.ReLU):
        super(Critic, self).__init__()

        q1_net = create_mlp(obs_dim + action_dim, 1, net_arch, activation_fn)
        self.q1_net = nn.Sequential(*q1_net)

        q2_net = create_mlp(obs_dim + action_dim, 1, net_arch, activation_fn)
        self.q2_net = nn.Sequential(*q2_net)

    def forward(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        qvalue_input = th.cat([obs, action], dim=1)
        return self.q1_net(qvalue_input), self.q2_net(qvalue_input)

    def q1_forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        return self.q1_net(th.cat([obs, action], dim=1))


class ValueFunction(BaseNetwork):
    """
    Value function for TD3 when doing on-policy exploration with SDE.

    :param obs_dim: (int) Dimension of the observation
    :param net_arch: (Optional[List[int]]) Network architecture
    :param activation_fn: (nn.Module) Activation function
    """
    def __init__(self, obs_dim: int, net_arch: Optional[List[int]] = None,
                 activation_fn: nn.Module = nn.Tanh):
        super(ValueFunction, self).__init__()

        if net_arch is None:
            net_arch = [64, 64]

        vf_net = create_mlp(obs_dim, 1, net_arch, activation_fn)
        self.vf_net = nn.Sequential(*vf_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.vf_net(obs)


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
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
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'cpu',
                 activation_fn: nn.Module = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 clip_noise: Optional[float] = None,
                 lr_sde: float = 3e-4,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False):
        super(TD3Policy, self).__init__(observation_space, action_space, device, squash_output=True)

        # Default network architecture, from the original paper
        if net_arch is None:
            net_arch = [400, 300]

        self.obs_dim = get_obs_dim(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
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
        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = th.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))

        if self.use_sde:
            self.vf_net = ValueFunction(self.obs_dim)
            self.actor.sde_optimizer.add_param_group({'params': self.vf_net.parameters()})  # pytype: disable=attribute-error

    def reset_noise(self) -> None:
        return self.actor.reset_noise()

    def make_actor(self) -> Actor:
        return Actor(**self.actor_kwargs).to(self.device)

    def make_critic(self) -> Critic:
        return Critic(**self.net_args).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False):
        return self.predict(observation, deterministic=deterministic)

    def predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)


MlpPolicy = TD3Policy

register_policy("MlpPolicy", MlpPolicy)
