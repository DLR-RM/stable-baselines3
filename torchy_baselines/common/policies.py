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
        return th.nn.utils.parameters_to_vector(self.parameters())


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
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy
