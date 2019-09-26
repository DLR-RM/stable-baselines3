import torch as th
import torch.nn as nn


class BasePolicy(nn.Module):
    """
    The base policy object

    :param observation_space: (Gym Space) The observation space of the environment
    :param action_space: (Gym Space) The action space of the environment
    """

    def __init__(self, observation_space, action_space, device='cpu'):
        super(BasePolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @staticmethod
    def init_weights(module, gain=1):
        if type(module) == nn.Linear:
            nn.init.orthogonal_(module.weight, gain=gain)
            module.bias.data.fill_(0.0)

    def forward(self, *_args, **kwargs):
        raise NotImplementedError()

    def save(self, path):
        """
        Save model to a given location.

        :param path: (str)
        """
        th.save(self.state_dict(), path)

    def load(self, path):
        """
        Load saved model from path.

        :param path: (str)
        """
        self.load_state_dict(th.load(path))

    def load_from_vector(self, vector):
        """
        Load parameters from a 1D vector.

        :param vector: (np.ndarray)
        """
        th.nn.utils.vector_to_parameters(th.FloatTensor(vector).to(self.device), self.parameters())

    def parameters_to_vector(self):
        """
        Convert the parameters to a 1D vector.

        :return: (np.ndarray)
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()


def create_mlp(input_dim, output_dim, net_arch,
               activation_fn=nn.ReLU, squash_out=False):
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
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


_policy_registry = dict()


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    # For building the doc
    try:
        for cls in BasePolicy.__subclasses__():
            if issubclass(policy, cls):
                sub_class = cls
                break
    except AttributeError:
        sub_class = str(th.random.randint(100))
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy
