from typing import Optional, List, Tuple, Callable, Union, Type, Any, Dict

import gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import (BasePolicy, register_policy, create_mlp,
                                               NatureCNN, BaseFeaturesExtractor, FlattenExtractor)


class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

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

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 net_arch: List[int],
                 features_extractor: nn.Module,
                 features_dim: int,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True,
                 device: Union[th.device, str] = 'auto'):
        super(Actor, self).__init__(observation_space, action_space,
                                    features_extractor=features_extractor,
                                    normalize_images=normalize_images,
                                    device=device,
                                    squash_output=True)

        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(dict(
            net_arch=self.net_arch,
            features_dim=self.features_dim,
            activation_fn=self.activation_fn,
            features_extractor=self.features_extractor
        ))
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return self.mu(features)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic=deterministic)


class Critic(BasePolicy):
    """
    Critic network for TD3,
    in fact it represents the action-state value function (Q-value function)

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

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # Learn the features extractor using the policy loss only
        with th.no_grad():
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return self.q1_net(qvalue_input), self.q2_net(qvalue_input)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q1_net(th.cat([features, actions], dim=1))


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (Union[th.device, str]) Device on which the code should run.
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
        super(TD3Policy, self).__init__(observation_space, action_space,
                                        device,
                                        features_extractor_class,
                                        features_extractor_kwargs,
                                        optimizer_class=optimizer_class,
                                        optimizer_kwargs=optimizer_kwargs,
                                        squash_output=True)

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [400, 300]
            else:
                net_arch = []

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
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None

        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1),
                                                    **self.optimizer_kwargs)
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1),
                                                     **self.optimizer_kwargs)

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

    def make_actor(self) -> Actor:
        return Actor(**self.net_args).to(self.device)

    def make_critic(self) -> Critic:
        return Critic(**self.net_args).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False):
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)


MlpPolicy = TD3Policy


class CnnPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (Union[th.device, str]) Device on which the code should run.
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
