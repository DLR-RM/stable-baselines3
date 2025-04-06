import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, RSPPO
from stable_baselines3.common.utils import set_random_seed

set_random_seed(42)

env = gym.make('CartPole-v1')
model = RSPPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=1e6)
