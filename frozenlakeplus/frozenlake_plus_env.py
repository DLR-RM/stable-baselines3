import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np

class FrozenLakePlus(FrozenLakeEnv):
    def __init__(self, map_name="4x4", is_slippery=True, dynamic_slippery=False, slippery_change_freq=10):
        super().__init__(map_name=map_name, is_slippery=is_slippery)
        self.dynamic_slippery = dynamic_slippery
        self.slippery_change_freq = slippery_change_freq
        self.step_count = 0
        self.slippery = is_slippery
        
    def step(self, action):
        self.step_count += 1
        
        if self.dynamic_slippery and self.step_count % self.slippery_change_freq == 0:
            self.slippery = not self.slippery
            self.is_slippery = self.slippery
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Convert the observation to a numpy array
        obs = np.array(obs)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        self.step_count = 0
        return super().reset(seed=seed, options=options)

__all__ = ["FrozenLakePlus"]
