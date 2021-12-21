from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import torch
from torch.nn.parameter import Parameter

from stable_baselines3.common.type_aliases import ReplayBufferSamples


class ActorLossModifier(ABC):
    @abstractmethod
    def modify_loss(self, actor_loss: torch.Tensor, replay_data: ReplayBufferSamples) -> torch.Tensor:
        """Modify the actor loss.

        :param actor_loss: initial actor loss
        :type actor_loss: torch.Tensor
        :param replay_data: replay data
        :type replay_data: ReplayBufferSamples
        :return: nex actor loss
        :rtype: torch.Tensor
        """

    def parameters(self) -> Iterator[Parameter]:
        pass


class RewardModifier(ABC):
    @abstractmethod
    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        """Modify the reward.

        :param replay_data: the replay data
        :return: the replay data with modified reward.
        """
