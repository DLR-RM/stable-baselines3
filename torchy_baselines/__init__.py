import os

from torchy_baselines.a2c import A2C
from torchy_baselines.cem_rl import CEMRL
from torchy_baselines.ppo import PPO
from torchy_baselines.sac import SAC
from torchy_baselines.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_file, 'r') as file_handler:
    __version__ = file_handler.read()
