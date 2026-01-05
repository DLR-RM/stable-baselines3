import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.gdb.gdb import GDB
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

env_id = "Pendulum-v1"
total_timesteps = 5_000

# PPO
# env = gym.make(env_id, max_episode_steps=4096)
# ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)
# ppo_model.learn(total_timesteps=total_timesteps)
# ppo_reward, _ = evaluate_policy(ppo_model, env, n_eval_episodes=10)
# print(f"PPO Mean Reward: {ppo_reward:.2f}")
# env.close()

# SAC
# env = gym.make(env_id)
# sac_model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=100_000, batch_size=256)
# sac_model.learn(total_timesteps=total_timesteps)
# sac_reward, _ = evaluate_policy(sac_model, env, n_eval_episodes=10)
# print(f"SAC Mean Reward: {sac_reward:.2f}")
# env.close()

# # TD3
env = gym.make(env_id)
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
td3_model = TD3("MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=100_000, 
                batch_size=256, action_noise=action_noise)
td3_model.learn(total_timesteps=total_timesteps)
td3_reward, _ = evaluate_policy(td3_model, env, n_eval_episodes=10)
print(f"TD3 Mean Reward: {td3_reward:.2f}")
env.close()

import torch
import gymnasium as gym

class PendulumDynamics:
    def __init__(self, env=None, device="cpu"):
        if env is None:
            env = gym.make("Pendulum-v1")
        
        unwrapped = env.unwrapped
        self.dt = unwrapped.dt
        self.g = unwrapped.g
        self.m = unwrapped.m
        self.l = unwrapped.l
        self.max_speed = unwrapped.max_speed
        self.max_torque = float(unwrapped.max_torque)
        self.device = torch.device(device)

    def step(self, state, action):
        """
        Args:
            state: [batch, 3] -> [cos(θ), sin(θ), θ̇]  or [3]
            action: [batch, 1] -> torque  or [1]
        Returns:
            next_state: [batch, 3] -> [cos(θ'), sin(θ'), θ̇']
            reward: [batch] or scalar
        """
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)

        cos_theta, sin_theta, theta_dot = state[:, 0], state[:, 1], state[:, 2]
        theta = torch.atan2(sin_theta, cos_theta)
        u = action.squeeze(-1).clamp(-self.max_torque, self.max_torque)

        # Reward: -(θ² + 0.1*θ̇² + 0.001*u²)
        reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * u**2)

        # Dynamics
        theta_ddot = (3 * self.g / (2 * self.l)) * torch.sin(theta) + (3 / (self.m * self.l**2)) * u
        theta_dot_new = (theta_dot + theta_ddot * self.dt).clamp(-self.max_speed, self.max_speed)
        theta_new = theta + theta_dot_new * self.dt

        next_state = torch.stack([torch.cos(theta_new), torch.sin(theta_new), theta_dot_new], dim=-1)
        
        if squeeze:
            return next_state.squeeze(0), reward.squeeze(0)
        return next_state, reward

# GDB
env = gym.make(env_id, 
            #    render_mode="human", 
               max_episode_steps=512)
surrogate_env = PendulumDynamics(env=env, device="cpu")
gdb_model = GDB("MlpPolicy", env, surrogate_env=surrogate_env, verbose=1, learning_rate=3e-4, n_steps=64)
gdb_model.learn(total_timesteps=total_timesteps)
gdb_reward, _ = evaluate_policy(gdb_model, env, n_eval_episodes=10)
print(f"GDB Mean Reward: {gdb_reward:.2f}")
env.close()

print("Training completed.")


# Summary
# print(f"\n{'='*40}\nResults (higher is better):\n  PPO: {ppo_reward:.2f}\n  SAC: {sac_reward:.2f}\n  TD3: {td3_reward:.2f}")