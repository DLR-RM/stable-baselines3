from frozenlake_plus_env import FrozenLakePlus
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(lambda: FrozenLakePlus(dynamic_slippery=True), n_envs=1)

print("Starting training...")
model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=10_000)
print("Training finished. Model saved as ppo_frozenlakeplus.zip")

model.save("ppo_frozenlakeplus")