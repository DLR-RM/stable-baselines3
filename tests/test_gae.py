import gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomEnv(gym.Env):
    def __init__(self, max_steps=8):
        super(CustomEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.max_steps = max_steps
        self.n_steps = 0

    def seed(self, seed):
        self.observation_space.seed(seed)

    def reset(self):
        self.n_steps = 0
        return self.observation_space.sample()

    def step(self, action):
        self.n_steps += 1

        done = False
        reward = 0.0
        if self.n_steps >= self.max_steps:
            reward = 1.0
            done = True

        return self.observation_space.sample(), reward, done, {}


class CheckGAECallback(BaseCallback):
    def __init__(self):
        super(CheckGAECallback, self).__init__(verbose=0)

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
        super(CustomPolicy, self).__init__(*args, **kwargs)
        self.constant_value = 0.0

    def forward(self, obs, deterministic=False):
        actions, values, log_prob = super().forward(obs, deterministic)
        # Overwrite values with ones
        values = th.ones_like(values) * self.constant_value
        return actions, values, log_prob


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
