import torch as th
import torch.nn as nn

from torchy_baselines.common.policies import BasePolicy, register_policy, create_mlp, BaseNetwork
from torchy_baselines.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(BaseNetwork):
    """
    Actor network (policy) for SAC.

    :param obs_dim: (int) Dimension of the observation
    :param action_dim: (int) Dimension of the action space
    :param net_arch: ([int]) Network architecture
    :param activation_fn: (nn.Module) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using SDE.
    """
    def __init__(self, obs_dim, action_dim, net_arch, activation_fn=nn.ReLU,
                 use_sde=False, log_std_init=-3, full_std=True):
        super(Actor, self).__init__()

        actor_net = create_mlp(obs_dim, -1, net_arch, activation_fn)
        self.actor_net = nn.Sequential(*actor_net)
        self.use_sde = use_sde

        if self.use_sde:
            # TODO: check for the learn_features
            self.action_dist = StateDependentNoiseDistribution(action_dim, full_std=full_std, use_expln=False,
                                                               learn_features=False, squash_output=True)
            self.mu, self.log_std = self.action_dist.proba_distribution_net(latent_dim=net_arch[-1],
                                                                            log_std_init=log_std_init)
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(net_arch[-1], action_dim)
            self.log_std = nn.Linear(net_arch[-1], action_dim)

    def get_std(self):
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using SDE.
        It corresponds to `th.exp(log_std)` in the normal case,
        but is slightly different when using `expln` function
        (cf StateDependentNoiseDistribution doc).

        :return: (th.Tensor)
        """
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self):
        """
        Sample new weights for the exploration matrix, when using SDE.
        """
        self.action_dist.sample_weights(self.log_std)

    def get_action_dist_params(self, obs):
        latent = self.actor_net(obs)

        if self.use_sde:
            mean_actions, log_std = self.mu(latent), self.log_std
        else:
            mean_actions, log_std = self.mu(latent), self.log_std(latent)
            # Original Implementation to cap the standard deviation
            log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, latent

    def forward(self, obs, deterministic=False):
        mean_actions, log_std, latent = self.get_action_dist_params(obs)
        if self.use_sde:
            # Note the action is squashed
            action, _ = self.action_dist.proba_distribution(mean_actions, log_std, latent, deterministic=deterministic)
        else:
            # Note the action is squashed
            action, _ = self.action_dist.proba_distribution(mean_actions, log_std, deterministic=deterministic)
        return action

    def action_log_prob(self, obs):
        mean_actions, log_std, latent = self.get_action_dist_params(obs)

        if self.use_sde:
            action, log_prob = self.action_dist.log_prob_from_params(mean_actions, self.log_std, latent)
        else:
            action, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std)
        return action, log_prob


class Critic(BaseNetwork):
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


class SACPolicy(BasePolicy):
    def __init__(self, observation_space, action_space,
                 learning_rate, net_arch=None, device='cpu',
                 activation_fn=nn.ReLU, use_sde=False, log_std_init=-3):
        super(SACPolicy, self).__init__(observation_space, action_space, device)

        if net_arch is None:
            net_arch = [256, 256]

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
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None

        self._build(learning_rate)

    def _build(self, learning_rate):
        self.actor = self.make_actor()
        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=learning_rate(1))

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = th.optim.Adam(self.critic.parameters(), lr=learning_rate(1))

    def make_actor(self):
        return Actor(**self.actor_kwargs).to(self.device)

    def make_critic(self):
        return Critic(**self.net_args).to(self.device)


MlpPolicy = SACPolicy

register_policy("MlpPolicy", MlpPolicy)
