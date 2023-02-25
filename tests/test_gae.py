from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomEnv(gym.Env):
    def __init__(self, max_steps=8):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.max_steps = max_steps
        self.n_steps = 0

    def seed(self, seed):
        self.observation_space.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.observation_space.seed(seed)
        self.n_steps = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self.n_steps += 1

        terminated = truncated = False
        reward = 0.0
        if self.n_steps >= self.max_steps:
            reward = 1.0
            terminated = True
            # To simplify GAE computation checks,
            # we do not consider truncation here.
            # Truncations are checked in InfiniteHorizonEnv
            truncated = False

        return self.observation_space.sample(), reward, terminated, truncated, {}


class InfiniteHorizonEnv(gym.Env):
    def __init__(self, n_states=4):
        super().__init__()
        self.n_states = n_states
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.current_state = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            super().reset(seed=seed)

        self.current_state = 0
        return self.current_state, {}

    def step(self, action):
        self.current_state = (self.current_state + 1) % self.n_states
        return self.current_state, 1.0, False, False, {}


class CheckGAECallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        rollout_size = buffer.size()

        max_steps = self.training_env.envs[0].max_steps
        gamma = self.model.gamma
        gae_lambda = self.model.gae_lambda
        value = self.model.policy.constant_value
        # We know in advance that the agent will get a single
        # reward at the very last timestep of the episode,
        # so we can pre-compute the lambda-return and advantage
        deltas = np.zeros((rollout_size,))
        advantages = np.zeros((rollout_size,))
        # Reward should be 1.0 on final timestep of episode
        rewards = np.zeros((rollout_size,))
        rewards[max_steps - 1 :: max_steps] = 1.0
        # Note that these are episode starts (+1 timestep from done)
        episode_starts = np.zeros((rollout_size,))
        episode_starts[::max_steps] = 1.0

        # Final step is always terminal (next would episode_start = 1)
        deltas[-1] = rewards[-1] - value
        advantages[-1] = deltas[-1]
        for n in reversed(range(rollout_size - 1)):
            # Values are constants
            episode_start_mask = 1.0 - episode_starts[n + 1]
            deltas[n] = rewards[n] + gamma * value * episode_start_mask - value
            advantages[n] = deltas[n] + gamma * gae_lambda * advantages[n + 1] * episode_start_mask

        # TD(lambda) estimate, see Github PR #375
        lambda_returns = advantages + value

        assert np.allclose(buffer.advantages.flatten(), advantages)
        assert np.allclose(buffer.returns.flatten(), lambda_returns)

    def _on_step(self):
        return True


class CustomPolicy(ActorCriticPolicy):
    """Custom Policy with a constant value function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant_value = 0.0

    def forward(self, obs, deterministic=False):
        actions, values, log_prob = super().forward(obs, deterministic)
        # Overwrite values with ones
        values = th.ones_like(values) * self.constant_value
        return actions, values, log_prob


@pytest.mark.parametrize("env_cls", [CustomEnv, InfiniteHorizonEnv])
def test_env(env_cls):
    # Check the env used for testing
    check_env(env_cls(), skip_render_check=True)


@pytest.mark.parametrize("model_class", [A2C, PPO])
@pytest.mark.parametrize("gae_lambda", [1.0, 0.9])
@pytest.mark.parametrize("gamma", [1.0, 0.99])
@pytest.mark.parametrize("num_episodes", [1, 3])
def test_gae_computation(model_class, gae_lambda, gamma, num_episodes):
    env = CustomEnv(max_steps=64)
    rollout_size = 64 * num_episodes
    model = model_class(
        CustomPolicy,
        env,
        seed=1,
        gamma=gamma,
        n_steps=rollout_size,
        gae_lambda=gae_lambda,
    )
    model.learn(rollout_size, callback=CheckGAECallback())

    # Change constant value so advantage != returns
    model.policy.constant_value = 1.0
    model.learn(rollout_size, callback=CheckGAECallback())


@pytest.mark.parametrize("model_class", [A2C, SAC])
@pytest.mark.parametrize("handle_timeout_termination", [False, True])
def test_infinite_horizon(model_class, handle_timeout_termination):
    max_steps = 8
    gamma = 0.98
    env = gym.wrappers.TimeLimit(InfiniteHorizonEnv(n_states=4), max_steps)
    kwargs = {}
    if model_class == SAC:
        policy_kwargs = dict(net_arch=[64], n_critics=1)
        kwargs = dict(
            replay_buffer_kwargs=dict(handle_timeout_termination=handle_timeout_termination),
            tau=0.5,
            learning_rate=0.005,
        )
    else:
        policy_kwargs = dict(net_arch=[64])
        kwargs = dict(learning_rate=0.002)
        # A2C always handle timeouts
        if not handle_timeout_termination:
            return

    model = model_class("MlpPolicy", env, gamma=gamma, seed=1, policy_kwargs=policy_kwargs, **kwargs)
    model.learn(1500)
    # Value of the initial state
    obs_tensor = model.policy.obs_to_tensor(0)[0]
    if model_class == A2C:
        value = model.policy.predict_values(obs_tensor).item()
    else:
        value = model.critic(obs_tensor, model.actor(obs_tensor))[0].item()
    # True value (geometric series with a reward of one at each step)
    infinite_horizon_value = 1 / (1 - gamma)

    if handle_timeout_termination:
        # true value +/- 1
        assert abs(infinite_horizon_value - value) < 1.0
    else:
        # wrong estimation
        assert abs(infinite_horizon_value - value) > 1.0
