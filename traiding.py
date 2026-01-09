import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple
from dataclasses import dataclass
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import polars as pl
import torch
from torch import nn

class TradingEnv(gym.Env):
    """Trading environment with enhanced features."""
    
    def __init__(
        self,
        price_data: np.ndarray,
        lookback: int,
        initial_balance: float,
        transaction_cost: float,
        max_position: float,
        action_smoothing_coef: float=0.9,

    ):
        super().__init__()
        
        self.current_step = lookback
        self.price_data = price_data
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.action_smoothing_coef = action_smoothing_coef 
        
        self.rewards = []
        
        self.observation_space = spaces.Dict({
            "price": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.lookback,), dtype=np.float32
            ),
            "position_size": spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "datetime": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            ),
        })
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.prev_action = 0.0
        self.reset()
    
    def reset(self, seed: Optional[int] = None, hard=False, options: Optional[dict] = None):
        super().reset(seed=seed)
        if hard:
            self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.portfolio_value = self.initial_balance
        self.prev_action = 0.0
        self.episode_trades = 0
        
        
        print(f"Ep Reward Mean: {np.mean(self.rewards) if self.rewards else 0}")
        self.rewards = []
        
        return self._get_observation(), {}
    
    def _get_observation(self, normalize: bool=True) -> dict:
        prices = self.price_data[self.current_step - self.lookback:self.current_step]
        
        if normalize:
            last_price = prices[-1]
            prices = prices - last_price
            prices = prices / np.abs(prices).max()
        
        return {
            "price": prices.astype(np.float32),
            "position_size": np.array([self.position_size], dtype=np.float32),
            "datetime": np.array([self.current_step], dtype=np.float32),
        }
    
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        
        target_position = float(np.clip(action[0], -1.0, 1.0))
        
        current_price = self.price_data[self.current_step]
        self.current_step += 1
        next_price = self.price_data[self.current_step]
        
        position_change = abs(target_position - self.position_size)
        cost = position_change * self.transaction_cost
        
        if position_change > 0.01:
            self.episode_trades += 1
        
        price_return = (next_price - current_price) / current_price
        pnl_ratio = self.position_size * price_return - cost
        pnl = pnl_ratio * self.portfolio_value 
        
        self.portfolio_value += pnl
        self.position_size = target_position
        
        # Base reward: log return
        reward = np.log(1 + pnl_ratio + 1e-8) * 100
        
        self.rewards.append(reward)
        
        if self.current_step >= len(self.price_data) - self.lookback:
            self.current_step = self.lookback
        
        # # Action smoothing penalty
        # action_change = abs(target_position - self.prev_action)
        # reward -= self.action_smoothing_coef * action_change
        # self.prev_action = target_position
        
        terminated = self.current_step >= len(self.price_data) - 1
        truncated = self.portfolio_value <= 0
        
        info = {
            "portfolio_value": self.portfolio_value,
            "position_size": self.position_size,
            "pnl": pnl,
            "cost": cost,
            "current_price": current_price,
            "episode_trades": self.episode_trades,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    

    
total_timesteps = 50_000

prices = pl.read_parquet("../data/futures/dataset/train/BTCUSDT.parquet")[-total_timesteps:]
prices = np.exp(prices.select("close").to_numpy().flatten())
    
env = TradingEnv(
    price_data=prices,
    lookback=60,    
    initial_balance=10000.0,
    transaction_cost=0.001,
    max_position=1.0,
    action_smoothing_coef=0.9,
)

env = gym.wrappers.TimeLimit(env, max_episode_steps=2048)

class TradingFeaturesExtractor(nn.Module):
    """Custom feature extractor for the Dict observation space."""
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__()
        
        self.observation_space = observation_space
        self.features_dim = features_dim    
        
        price_shape = observation_space["price"].shape[0]
        
        self.price_encoder = nn.Sequential(
            nn.Linear(price_shape, features_dim*2),
            nn.ReLU(),
            nn.Linear(features_dim*2, features_dim),
            nn.ReLU(),
        )
        
        self.combined = nn.Sequential(
            nn.Linear(features_dim + 1, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        price_features = self.price_encoder(observations["price"])
        combined = torch.cat([price_features, observations["position_size"]], dim=-1)
        return self.combined(combined)


net_arch = dict(pi=[256, 256], qf=[256, 256])

policy_kwargs = dict(
        features_extractor_class=TradingFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=net_arch,
        share_features_extractor=True,
    )

# SAC

sac_model = SAC("MultiInputPolicy", 
                env, 
                verbose=1, 
                learning_rate=3e-4, 
                buffer_size=100_000, 
                batch_size=256, 
                policy_kwargs=policy_kwargs, 
                device="cpu",
                )
sac_model.learn(total_timesteps=total_timesteps)
sac_reward, _ = evaluate_policy(sac_model, env, n_eval_episodes=10)
print(f"SAC Mean Reward: {sac_reward:.2f}")
env.close()

# PPO
from stable_baselines3 import PPO

ppo_model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, policy_kwargs=policy_kwargs, device="mps")
ppo_model.learn(total_timesteps=total_timesteps)
ppo_reward, _ = evaluate_policy(ppo_model, env, n_eval_episodes=10)
print(f"PPO Mean Reward: {ppo_reward:.2f}")
env.close()