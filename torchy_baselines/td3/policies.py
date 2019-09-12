import torch as th
import torch.nn as nn

from torchy_baselines.common.policies import BasePolicy, register_policy


def create_mlp(input_dim, output_dim, net_arch,
               activation_fn=nn.ReLU, squash_out=False):
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    modules.append(nn.Linear(net_arch[-1], output_dim))
    if squash_out:
        modules.append(nn.Tanh())
    return modules


class BaseNetwork(nn.Module):
    """docstring for BaseNetwork."""

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def load_from_vector(self, vector):
        """
        Load parameters from a 1D vector.

        :param vector: (np.ndarray)
        """
        device = next(self.parameters()).device
        th.nn.utils.vector_to_parameters(th.FloatTensor(vector).to(device), self.parameters())

    def parameters_to_vector(self):
        """
        Convert the parameters to a 1D vector.

        :return: (np.ndarray)
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()


class Actor(BaseNetwork):
    def __init__(self, state_dim, action_dim, net_arch=None, activation_fn=nn.ReLU):
        super(Actor, self).__init__()

        if net_arch is None:
            net_arch = [400, 300]

        # TODO: orthogonal initialization?
        actor_net = create_mlp(state_dim, action_dim, net_arch, activation_fn, squash_out=True)
        self.actor_net = nn.Sequential(*actor_net)

    def forward(self, x):
        return self.actor_net(x)


class Critic(BaseNetwork):
    def __init__(self, state_dim, action_dim,
                 net_arch=None, activation_fn=nn.ReLU):
        super(Critic, self).__init__()

        if net_arch is None:
            net_arch = [400, 300]

        # TODO: solve  pytorch parameter registration
        # for _ in range(n_critics):
        #     q_net = create_mlp(state_dim + action_dim, 1, net_arch, activation_fn)
        #     self.q_net = nn.Sequential(*q_net)
        #     self.q_networks.append(self.q_net)

        q1_net = create_mlp(state_dim + action_dim, 1, net_arch, activation_fn)
        self.q1_net = nn.Sequential(*q1_net)

        q2_net = create_mlp(state_dim + action_dim, 1, net_arch, activation_fn)
        self.q2_net = nn.Sequential(*q2_net)

        self.q_networks = [self.q1_net, self.q2_net]

    def forward(self, obs, action):
        qvalue_input = th.cat([obs, action], dim=1)
        return [q_net(qvalue_input) for q_net in self.q_networks]

    def q1_forward(self, obs, action):
        return self.q_networks[0](th.cat([obs, action], dim=1))


class TD3Policy(BasePolicy):
    def __init__(self, observation_space, action_space,
                 learning_rate=1e-3, net_arch=None, device='cpu',
                 activation_fn=nn.ReLU):
        super(TD3Policy, self).__init__(observation_space, action_space, device)
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
        }
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self._build(learning_rate)

    def _build(self, learning_rate):
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = th.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def make_actor(self):
        return Actor(**self.net_args).to(self.device)

    def make_critic(self):
        return Critic(**self.net_args).to(self.device)

MlpPolicy = TD3Policy

register_policy("MlpPolicy", MlpPolicy)
