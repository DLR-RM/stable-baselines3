import torch as th
import torch.nn as nn
from torch.distributions import Normal

from torchy_baselines.common.policies import BasePolicy, register_policy, create_mlp, BaseNetwork


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
        for the std instead of only (n_features,)
    """
    def __init__(self, obs_dim, action_dim, net_arch, activation_fn=nn.ReLU,
                 use_sde=False, log_std_init=-2, clip_noise=None,
                 lr_sde=3e-4, full_std=False):
        super(Actor, self).__init__()

        self.latent_pi, self.log_std = None, None
        self.weights_dist, self.exploration_mat = None, None
        self.use_sde, self.sde_optimizer = use_sde, None
        self.action_dim = action_dim
        self.full_std = full_std

        if use_sde:
            latent_dim = net_arch[-1]
            latent_pi = create_mlp(obs_dim, -1, net_arch, activation_fn, squash_out=False)
            self.latent_pi = nn.Sequential(*latent_pi)
            if full_std:
                self.log_std = nn.Parameter(th.ones(latent_dim, action_dim) * log_std_init, requires_grad=True)
            else:
                # Reduce the number of parameters:
                self.log_std = nn.Parameter(th.ones(latent_dim, 1) * log_std_init, requires_grad=True)

            self.latent_dim = latent_dim
            self.actor_net = nn.Sequential(nn.Linear(net_arch[-1], action_dim), nn.Tanh())
            self.clip_noise = clip_noise
            self.sde_optimizer = th.optim.Adam([self.log_std], lr=lr_sde)
            self.reset_noise()
        else:
            actor_net = create_mlp(obs_dim, action_dim, net_arch, activation_fn, squash_out=True)
            self.actor_net = nn.Sequential(*actor_net)

    def get_log_std(self):
        if self.full_std:
            return self.log_std
        # Reduce the number of parameters:
        return th.ones((self.latent_dim, self.action_dim)).to(self.log_std.device) * self.log_std

    def evaluate_actions(self, obs, action):
        """
        Evaluate actions according to the current policy,
        given the observations. Only useful when using SDE.

        :param obs: (th.Tensor)
        :param action: (th.Tensor)
        :return: (th.Tensor, th.Tensor) log likelihood of taking those actions
            and entropy of the action distribution.
        """
        with th.no_grad():
            latent_pi = self.latent_pi(obs)
            mean_actions = self.actor_net(latent_pi)

        variance = th.mm(latent_pi ** 2, th.exp(self.get_log_std()) ** 2)
        distribution = Normal(mean_actions, th.sqrt(variance + 1e-5))
        log_prob = distribution.log_prob(action)
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(axis=1)
        else:
            log_prob = log_prob.sum()
        return log_prob, distribution.entropy()

    def reset_noise(self):
        self.weights_dist = Normal(th.zeros_like(self.get_log_std()), th.exp(self.get_log_std()))
        self.exploration_mat = self.weights_dist.rsample()

    def forward(self, obs, deterministic=True):
        if self.use_sde:
            latent_pi = self.latent_pi(obs)
            if deterministic:
                return self.actor_net(latent_pi)
            noise = th.mm(latent_pi.detach(), self.exploration_mat)
            if self.clip_noise is not None:
                noise = th.clamp(noise, -self.clip_noise, self.clip_noise)
            # TODO: Replace with squashing -> need to account for that in the sde update
            # return th.clamp(self.actor_net(latent_pi) + noise, -1, 1)
            # NOTE: the clipping is done in the rollout for now
            return self.actor_net(latent_pi) + noise
        else:
            return self.actor_net(obs)


class Critic(BaseNetwork):
    """
    Critic network for TD3,
    in fact it represents the action-state value function (Q-value function)

    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param net_arch: ([int]) Network architecture
    :param activation_fn: (nn.Module) Activation function
    """
    def __init__(self, obs_dim, action_dim,
                 net_arch, activation_fn=nn.ReLU):
        super(Critic, self).__init__()

        q1_net = create_mlp(obs_dim + action_dim, 1, net_arch, activation_fn)
        self.q1_net = nn.Sequential(*q1_net)

        q2_net = create_mlp(obs_dim + action_dim, 1, net_arch, activation_fn)
        self.q2_net = nn.Sequential(*q2_net)

        self.q_networks = [self.q1_net, self.q2_net]

    def forward(self, obs, action):
        qvalue_input = th.cat([obs, action], dim=1)
        return [q_net(qvalue_input) for q_net in self.q_networks]

    def q1_forward(self, obs, action):
        return self.q_networks[0](th.cat([obs, action], dim=1))


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_dim: (gym.spaces.Space) Action space
    :param learning_rate: (callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (nn.Module) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    """
    def __init__(self, observation_space, action_space,
                 learning_rate, net_arch=None, device='cpu',
                 activation_fn=nn.ReLU, use_sde=False, log_std_init=-2,
                 clip_noise=None, lr_sde=3e-4):
        super(TD3Policy, self).__init__(observation_space, action_space, device)

        # Default network architecture, from the original paper
        if net_arch is None:
            net_arch = [400, 300]

        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs['use_sde'] = use_sde
        self.actor_kwargs['log_std_init'] = log_std_init
        self.actor_kwargs['clip_noise'] = clip_noise
        self.actor_kwargs['lr_sde'] = lr_sde

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.use_sde = use_sde
        self.log_std_init = log_std_init
        self._build(learning_rate)

    def _build(self, learning_rate):
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=learning_rate(1))

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = th.optim.Adam(self.critic.parameters(), lr=learning_rate(1))

    def reset_noise(self):
        return self.actor.reset_noise()

    def make_actor(self):
        return Actor(**self.actor_kwargs).to(self.device)

    def make_critic(self):
        return Critic(**self.net_args).to(self.device)

    def forward(self, obs, deterministic=True):
        return self.actor(obs, deterministic=deterministic)


MlpPolicy = TD3Policy

register_policy("MlpPolicy", MlpPolicy)
