import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import polars as pl
import torch
from torch import nn
from typing import Dict, Tuple

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class TradingEnv(gym.Env):
    """Trading environment with enhanced features."""
    
    def __init__(
        self,
        price_data: np.ndarray,
        lookback: int,
        initial_balance: float,
        transaction_cost: float,
        max_position: float,
        action_smoothing_coef: float = 0.9,
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
                low=-100.0, high=100.0, shape=(1,), dtype=np.float32
            ),
            "datetime": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            ),
        })
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(64,), dtype=np.float32
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
        
        self.rewards = []
        
        return self._get_observation(), {}
    
    def _get_observation(self, normalize: bool = True, details: bool = False) -> dict:
        prices = self.price_data[self.current_step - self.lookback:self.current_step]
        
        if normalize:
            last_price = prices[-1]
            prices = prices - last_price
            scalar_max = np.abs(prices).max()
            prices = prices / scalar_max

        obs = {
            "price": prices.astype(np.float32),
            "position_size": np.array([self.position_size], dtype=np.float32),
            "datetime": np.array([self.current_step], dtype=np.float32),
        }
        
        if details and normalize:
            obs.update({
                "scalar_max": np.array([scalar_max], dtype=np.float32),
            })
        
        return obs

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:

        current_price = self.price_data[self.current_step]
        prev_observation = self._get_observation(normalize=True, details=True)

        self.current_step += 1
        next_price = self.price_data[self.current_step]

        orders = np.linspace(
            -prev_observation["scalar_max"].item(),
            prev_observation["scalar_max"].item(),
            num=action.shape[0]
        ) + current_price

        if current_price > next_price:
            taken_action = ((orders < current_price) & (orders > next_price)).astype(np.float32)
        else:
            taken_action = -((orders > current_price) & (orders < next_price)).astype(np.float32)

        positions = action * taken_action.astype(np.float32)
        cost = positions.sum() * self.transaction_cost

        pnl_ratio = (
            (next_price - current_price) / current_price * self.position_size +
            np.sum(positions * (np.repeat(next_price, action.shape[0]) - orders) / orders) -
            cost
        )

        self.position_size += np.sum(positions)
        self.episode_trades += np.sum(taken_action)
        pnl = pnl_ratio * self.portfolio_value
        self.portfolio_value += pnl

        reward = np.log((1 + pnl_ratio).clip(min=1e-8)) * 100
        self.rewards.append(reward)

        if self.current_step >= len(self.price_data) - self.lookback:
            self.current_step = self.lookback

        terminated = self.current_step >= len(self.price_data) - 1
        truncated = self.portfolio_value <= 0
        
        info = {
            "portfolio_value": self.portfolio_value,
            "position_size": self.position_size,
            "pnl": pnl,
            "cost": cost,
            "current_price": current_price,
            "episode_trades": self.episode_trades,
            "ep_reward_mean": np.mean(self.rewards) if self.rewards else 0,
        }
        
        return self._get_observation(), reward, terminated, truncated, info


class TradingEnvTorch(nn.Module):
    """Differentiable trading environment in PyTorch."""
    
    def __init__(
        self,
        price_data: torch.Tensor,
        lookback: int,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        action_dim: int = 64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.register_buffer("price_data", price_data.to(self.device))
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.action_dim = action_dim
        
        self.reset()
    
    def reset(
        self, 
        batch_size: int = 1, 
        start_steps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        self.batch_size = batch_size
        
        if start_steps is None:
            self.current_step = torch.full(
                (batch_size,), self.lookback, dtype=torch.long, device=self.device
            )
        else:
            self.current_step = start_steps.to(self.device)
        
        self.balance = torch.full(
            (batch_size,), self.initial_balance, dtype=torch.float32, device=self.device
        )
        self.position_size = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.portfolio_value = self.balance.clone()
        self.episode_trades = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        return self._get_observation()
    
    def _get_observation(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        # Gather price windows: [batch_size, lookback]
        offsets = torch.arange(-self.lookback, 0, device=self.device)
        indices = self.current_step.unsqueeze(1) + offsets.unsqueeze(0)
        prices = self.price_data[indices]
        
        scalar_max = torch.ones(self.batch_size, device=self.device)
        
        if normalize:
            last_price = prices[:, -1:]
            prices = prices - last_price
            scalar_max = prices.abs().max(dim=1).values.clamp(min=1e-8)
            prices = prices / scalar_max.unsqueeze(1)
        
        return {
            "price": prices,
            "position_size": self.position_size.unsqueeze(1),
            "datetime": self.current_step.unsqueeze(1).float(),
            "scalar_max": scalar_max,
        }
    
    def step(
        self, 
        state,
        action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            action: [batch_size, action_dim] tensor of actions in [-1, 1]
        Returns:
            obs, reward, terminated, truncated, info
        """
        obs = self._get_observation(normalize=True)
        scalar_max = obs["scalar_max"]
        
        current_price = self.price_data[self.current_step]
        next_step = (self.current_step + 1).clamp(max=len(self.price_data) - 1)
        next_price = self.price_data[next_step]
        
        # Order grid: [batch_size, action_dim]
        order_offsets = torch.linspace(-1, 1, self.action_dim, device=self.device)
        orders = order_offsets.unsqueeze(0) * scalar_max.unsqueeze(1) + current_price.unsqueeze(1)
        
        # Determine which orders are triggered
        price_went_down = (current_price > next_price).unsqueeze(1)
        price_went_up = (current_price < next_price).unsqueeze(1)
        
        # Soft trigger using sigmoid for gradient flow
        temp = 0.01  # Temperature for soft comparison
        
        down_trigger = torch.sigmoid((current_price.unsqueeze(1) - orders) / temp) * \
                       torch.sigmoid((orders - next_price.unsqueeze(1)) / temp)
        up_trigger = torch.sigmoid((orders - current_price.unsqueeze(1)) / temp) * \
                     torch.sigmoid((next_price.unsqueeze(1) - orders) / temp)
        
        taken_action = torch.where(
            price_went_down,
            down_trigger,
            torch.where(price_went_up, -up_trigger, torch.zeros_like(orders))
        )
        
        positions = action * taken_action
        cost = positions.sum(dim=1).abs() * self.transaction_cost
        
        # PnL calculation
        price_return = (next_price - current_price) / current_price.clamp(min=1e-8)
        order_returns = (next_price.unsqueeze(1) - orders) / orders.clamp(min=1e-8)
        
        pnl_ratio = (
            price_return * self.position_size +
            (positions * order_returns).sum(dim=1) -
            cost
        )
        
        # Update state
        self.position_size = self.position_size + positions.sum(dim=1)
        self.episode_trades = self.episode_trades + taken_action.abs().sum(dim=1)
        
        pnl = pnl_ratio * self.portfolio_value
        self.portfolio_value = self.portfolio_value + pnl
        
        # Log return reward
        reward = torch.log((1 + pnl_ratio).clamp(min=1e-8)) * 100
        
        # Advance step with wraparound
        self.current_step = torch.where(
            next_step >= len(self.price_data) - self.lookback,
            torch.full_like(self.current_step, self.lookback),
            next_step
        )
        
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
    
    def get_portfolio_value(self) -> torch.Tensor:
        """For direct optimization of portfolio value."""
        return self.portfolio_value

    def reset(self, batch_size: int = 1, start_step: Optional[torch.Tensor] = None) -> dict:
        """Initialize state."""
        if start_step is None:
            start_step = torch.full((batch_size,), self.lookback, device=self.device)

        prices = torch.zeros(batch_size, self.lookback, device=self.device)
        for i in range(batch_size):
            idx = start_step[i].long()
            raw = self.price_data[idx - self.lookback:idx]
            last = raw[-1]
            prices[i] = (raw - last) / raw.abs().max()

        state = {
            "price": prices,
            "position_size": torch.zeros(batch_size, 1, device=self.device),
            "datetime": start_step.float().unsqueeze(-1),
            "portfolio_value": torch.full((batch_size, 1), self.initial_balance, device=self.device),
        }
        return state if batch_size > 1 else {k: v.squeeze(0) for k, v in state.items()}


class TradingFeaturesExtractor(nn.Module):
    """Custom feature extractor for the Dict observation space."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__()

        self.observation_space = observation_space
        self.features_dim = features_dim

        price_shape = observation_space["price"].shape[0]

        self.price_encoder = nn.Sequential(
            nn.Linear(price_shape, features_dim * 2),
            nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim),
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


class EnvInfoCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            self.logger.record("env_info/ep_reward_mean", info.get("ep_reward_mean", 0))
        return True


if __name__ == "__main__":
    from stable_baselines3.sgb.sgb import GDB

    total_timesteps = 500_000

    prices = pl.read_parquet("../data/futures/dataset/train/BTCUSDT.parquet")
    prices = np.exp(prices.select("close").to_numpy().flatten())

    env = TradingEnv(
        price_data=prices,
        lookback=60,
        initial_balance=10000.0,
        transaction_cost=0.0001,
        max_position=1.0,
        action_smoothing_coef=0.9,
    )

    env = gym.wrappers.TimeLimit(env, max_episode_steps=4096)
    env = Monitor(env)

    surrogate_env = TradingDynamics(env=env, device="cpu")

    net_arch = dict(pi=[256, 256], qf=[256, 256])
    policy_kwargs = dict(
        features_extractor_class=TradingFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=net_arch,
        share_features_extractor=True,
    )
    
    # PPO
    
    # ppo_model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="./logs/ppo_trading/",
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     policy_kwargs=policy_kwargs,
    #     device="cpu",
    # )
    # ppo_model.learn(total_timesteps=total_timesteps, callback=EnvInfoCallback())
    # ppo_reward, _ = evaluate_policy(ppo_model, env, n_eval_episodes=10)
    # print(f"PPO Mean Reward: {ppo_reward:.2f}")
    # env.close()

    # SAC
    sac_model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/sac_trading/",
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )
    sac_model.learn(total_timesteps=total_timesteps, callback=EnvInfoCallback())
    sac_reward, _ = evaluate_policy(sac_model, env, n_eval_episodes=10)
    print(f"SAC Mean Reward: {sac_reward:.2f}")
    env.close()
    
    

    # GDB
    gdb_model = GDB(
        "MultiInputPolicy",
        env,
        tensorboard_log="./logs/gdb_trading/",
        surrogate_env=surrogate_env,
        verbose=1,
        learning_rate=5e-5,
        n_steps=512,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )

    gdb_model.learn(total_timesteps=total_timesteps, callback=EnvInfoCallback())
    gdb_reward, _ = evaluate_policy(gdb_model, env, n_eval_episodes=10)
    print(f"GDB Mean Reward: {gdb_reward:.2f}")
    env.close()
