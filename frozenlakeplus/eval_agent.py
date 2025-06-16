from frozenlake_plus_env import FrozenLakePlus
from stable_baselines3 import PPO
import numpy as np

env = FrozenLakePlus(dynamic_slippery=False)
model = PPO.load("ppo_frozenlakeplus", env=env)

n_episodes = 10
for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Convert action to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()