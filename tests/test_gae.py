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
    """docstring for CheckGAECallback."""

    def __init__(self):
        super(CheckGAECallback, self).__init__(verbose=0)

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer

        max_steps = self.training_env.envs[0].max_steps
        gamma = self.model.gamma
        gae_lambda = self.model.gae_lambda
        value = self.model.policy.constant_value
        # We know in advance that the agent will get a single
        # reward at the very last timestep of the episode,
        # so we can pre-compute the return
        # returns = np.array([gamma ** n for n in range(max_steps)])[::-1]

        # the same goes for the advantage
        deltas = np.zeros((max_steps,))
        advantages = np.zeros((max_steps,))
        rewards = np.array([0.0] * (max_steps - 1) + [1.0])
        deltas[-1] = rewards[-1] - value
        advantages[-1] = deltas[-1]
        for n in reversed(range(max_steps - 1)):
            deltas[n] = rewards[n] + gamma * value - value
            advantages[n] = deltas[n] + gamma * gae_lambda * advantages[n + 1]

        returns = advantages + value

        assert np.allclose(buffer.advantages.flatten(), advantages)
        assert np.allclose(buffer.returns.flatten(), returns)

        # Previous implementation
        # if self.value_is_zero:
        #     # Constant value function that outputs zeros
        #     # advantage and return must be the same
        #     assert np.allclose(buffer.returns, buffer.advantages)

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
def test_gae_computation(model_class, gae_lambda, gamma):
    env = CustomEnv(max_steps=64)
    model = model_class(
        CustomPolicy,
        env,
        seed=1,
        gamma=gamma,
        n_steps=env.max_steps,
        gae_lambda=gae_lambda,
    )
    model.learn(env.max_steps, callback=CheckGAECallback())

    # Change constant value so advantage != returns
    model.policy.constant_value = 1.0
    model.learn(env.max_steps, callback=CheckGAECallback())
