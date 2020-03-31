from typing import Optional, List, Tuple, Callable, Union, Type

import gym
import torch as th
import torch.nn as nn

from torchy_baselines.common.preprocessing import get_action_dim, get_obs_dim
from torchy_baselines.common.policies import (BasePolicy, register_policy, create_mlp)


class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net for DQN

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param log_std_init: (float) Initial value for the log standard deviation
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'cpu',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 log_std_init: float = 0.0,
                 normalize_images: bool = True):
        super(DQNPolicy, self).__init__(observation_space, action_space, device)

        if net_arch is None:
            net_arch = [256, 256]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        # In the future, features_extractor will be replaced with a CNN
        self.features_extractor = nn.Flatten()
        # In the future, feature_extractor will be replaced with a CNN
        self.features_extractor = nn.Flatten()
        self.features_dim = get_obs_dim(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)
        self.normalize_images = normalize_images
        self.log_std_init = log_std_init # Not used by now, only discrete env supported

        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        """
        Create the network and the optimizer.

        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        q_net = create_mlp(self.features_dim, self.action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

        # Setup optimizer with initial learning rate
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Forward pass
        :param obs: (th.Tensor) Observation
        """
        features = self.extract_features(obs)
        return self.q_net(features)


MlpPolicy = DQNPolicy

register_policy("MlpPolicy", MlpPolicy)
