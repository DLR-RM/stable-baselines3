from abc import ABC, abstractmethod
from typing import Iterator

import torch
from torch.nn.parameter import Parameter

from stable_baselines3.common.type_aliases import ReplayBufferSamples


class Surgeon(ABC):
    def modify_actor_loss(self, actor_loss: torch.Tensor, replay_data: ReplayBufferSamples) -> torch.Tensor:
        """
        Modify the actor loss.

        :param actor_loss: Initial actor loss
        :param replay_data: Replay data
        :return: New actor loss
        """
        return actor_loss

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        """
        Modify the reward.

        :param reward: Initial reward
        :param replay_data: Replay data
        :return: New reward
        """
        return replay_data

    def parameters(self) -> Iterator[Parameter]:
        return []

    def setup(self, model):
        self.logger = model.logger
        self.model = model

    def on_step(self):
        return