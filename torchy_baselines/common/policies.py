from typing import Union, Type, Dict, List, Tuple, Optional

from itertools import zip_longest

import gym
import torch as th
import torch.nn as nn
import numpy as np

from torchy_baselines.common.preprocessing import preprocess_obs


class BasePolicy(nn.Module):
    """
    The base policy object

    :param observation_space: (gym.spaces.Space) The observation space of the environment
    :param action_space: (gym.spaces.Space) The action space of the environment
    :param device: (Union[th.device, str]) Device on which the code should run.
    :param squash_output: (bool) For continuous actions, whether the output is squashed
        or not using a `tanh()` function.
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 squash_output: bool = False,
                 features_extractor: Optional[nn.Module] = None,
                 normalize_images: bool = True):
        super(BasePolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self._squash_output = squash_output

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: (th.Tensor)
        :return: (th.Tensor)
        """
        assert self.features_extractor is not None, 'No feature extractor was set'
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)

    @property
    def squash_output(self) -> bool:
        """ (bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1):
        if type(module) == nn.Linear:
            nn.init.orthogonal_(module.weight, gain=gain)
            module.bias.data.fill_(0.0)

    def forward(self, *_args, **kwargs):
        raise NotImplementedError()

    def predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        raise NotImplementedError()

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path: (str)
        """
        th.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load saved model from path.

        :param path: (str)
        """
        self.load_state_dict(th.load(path))

    def load_from_vector(self, vector: np.ndarray):
        """
        Load parameters from a 1D vector.

        :param vector: (np.ndarray)
        """
        th.nn.utils.vector_to_parameters(th.FloatTensor(vector).to(self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return: (np.ndarray)
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()


def create_mlp(input_dim: int,
               output_dim: int,
               net_arch: List[int],
               activation_fn: nn.Module = nn.ReLU,
               squash_output: bool = False) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: (int) Dimension of the input vector
    :param output_dim: (int)
    :param net_arch: (List[int]) Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: (nn.Module) The activation function
        to use after each layer.
    :param squash_output: (bool) Whether to squash the output using a Tanh
        activation function
    :return: (List[nn.Module])
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        modules.append(nn.Linear(net_arch[-1], output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def create_sde_features_extractor(features_dim: int,
                                  sde_net_arch: List[int],
                                  activation_fn: nn.Module) -> Tuple[nn.Sequential, int]:
    """
    Create the neural network that will be used to extract features
    for the SDE exploration function.

    :param features_dim: (int)
    :param sde_net_arch: ([int])
    :param activation_fn: (nn.Module)
    :return: (nn.Sequential, int)
    """
    # Special case: when using states as features (i.e. sde_net_arch is an empty list)
    # don't use any activation function
    sde_activation = activation_fn if len(sde_net_arch) > 0 else None
    latent_sde_net = create_mlp(features_dim, -1, sde_net_arch, activation_fn=sde_activation, squash_output=False)
    latent_sde_dim = sde_net_arch[-1] if len(sde_net_arch) > 0 else features_dim
    sde_features_extractor = nn.Sequential(*latent_sde_net)
    return sde_features_extractor, latent_sde_dim


_policy_registry = dict()  # type: Dict[Type[BasePolicy], Dict[str, Type[BasePolicy]]]


def get_policy_from_name(base_policy_type: Type[BasePolicy], name: str) -> Type[BasePolicy]:
    """
    Returns the registered policy from the base type and name

    :param base_policy_type: (Type[BasePolicy]) the base policy class
    :param name: (str) the policy name
    :return: (Type[BasePolicy]) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError(f"Error: the policy type {base_policy_type} is not registered!")
    if name not in _policy_registry[base_policy_type]:
        raise ValueError(f"Error: unknown policy type {name},"
                         "the only registed policy type are: {list(_policy_registry[base_policy_type].keys())}!")
    return _policy_registry[base_policy_type][name]


def register_policy(name: str, policy: Type[BasePolicy]) -> None:
    """
    Register a policy, so it can be called using its name.
    e.g. SAC('MlpPolicy', ...) instead of SAC(MlpPolicy, ...)

    :param name: (str) the policy name
    :param policy: (Type[BasePolicy]) the policy class
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
        raise ValueError(f"Error: the policy {policy} is not of any known subclasses of BasePolicy!")

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError(f"Error: the name {name} is alreay registered for a different policy, will not override.")
    _policy_registry[sub_class][name] = policy


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: (int) Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: (nn.Module) The activation function to use for the networks.
    :param device: (th.device)
    """
    def __init__(self, feature_dim: int,
                 net_arch: List[Union[int, Dict[str, List[int]]]],
                 activation_fn: nn.Module,
                 device: Union[th.device, str] = 'cpu'):
        super(MlpExtractor, self).__init__()

        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if 'pi' in layer:
                    assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer['pi']

                if 'vf' in layer:
                    assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer['vf']
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)
