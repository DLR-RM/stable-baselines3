import os

import torch

from stable_baselines3.a2c import A2C
from stable_baselines3.ddpg import DDPG
from stable_baselines3.dqn import DQN
from stable_baselines3.her import HER
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3

# See https://www.youtube.com/watch?v=9mS1fIYj1So
# PyTorch Performance Tuning Guide

torch.backends.cudnn.benchmark = True

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
