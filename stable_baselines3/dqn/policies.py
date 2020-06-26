from typing import Optional, List, Callable, Union, Type, Any, Dict

import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import create_mlp, NatureCNN, BaseFeaturesExtractor, FlattenExtractor


class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 features_extractor: nn.Module,
                 features_dim: int,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True):
        super(QNetwork, self).__init__(observation_space, action_space,
                                       features_extractor=features_extractor,
                                       normalize_images=normalize_images,
                                       device=device)

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: (th.Tensor) Observation
        :return: (th.Tensor) The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
            net_arch=self.net_arch,
            features_dim=self.features_dim,
            activation_fn=self.activation_fn,
            features_extractor=self.features_extractor,
            epsilon=self.epsilon,
        ))
        return data


class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
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
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super(DQNPolicy, self).__init__(observation_space, action_space,
                                        device,
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        optimizer_class=optimizer_class,
                                        optimizer_kwargs=optimizer_kwargs)

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.features_extractor = features_extractor_class(self.observation_space,
                                                           **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

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

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        """
        Create the network and the optimizer.

        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        return QNetwork(**self.net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
            net_arch=self.net_args['net_arch'],
            activation_fn=self.net_args['activation_fn'],
            lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs
        ))
        return data


MlpPolicy = DQNPolicy


class CnnPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
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
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        normalize_images,
                                        optimizer_class,
                                        optimizer_kwargs)


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
