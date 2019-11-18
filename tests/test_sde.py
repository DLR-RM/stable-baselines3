import pytest

import gym
import torch as th
from torch.distributions import Normal

from torchy_baselines import A2C, TD3
from torchy_baselines.common.vec_env import DummyVecEnv, VecNormalize
from torchy_baselines.common.monitor import Monitor


def test_state_dependent_exploration():
    """
    Check that the gradient correspond to the expected one
    """
    n_states = 2
    state_dim = 3
    # TODO: fix for action_dim > 1
    action_dim = 1
    sigma = th.ones(state_dim, 1, requires_grad=True)
    # Reduce the number of parameters
    # sigma_ = th.ones(state_dim, action_dim) * sigma_

    # weights_dist = Normal(th.zeros_like(log_sigma), th.exp(log_sigma))
    th.manual_seed(2)
    weights_dist = Normal(th.zeros_like(sigma), sigma)

    weights = weights_dist.rsample()
    state = th.rand(n_states, state_dim)
    mu = th.ones(action_dim)
    # print(weights.shape, state.shape)
    noise = th.mm(state, weights)

    variance = th.mm(state ** 2, sigma ** 2)
    action_dist = Normal(mu, th.sqrt(variance))

    loss = action_dist.log_prob((mu + noise).detach()).mean()
    loss.backward()

    # From Rueckstiess paper
    grad = th.zeros_like(sigma)
    for j in range(action_dim):
        for i in range(state_dim):
            a = ((noise[:, j] ** 2 - variance[:, j]) / (variance[:, j] ** 2)) * (state[:, i] ** 2 * sigma[i, j])
            grad[i, j] = a.mean()

    # sigma.grad should be equal to grad
    assert sigma.grad.allclose(grad)


@pytest.mark.parametrize("model_class", [A2C])
def test_state_dependent_noise(model_class):
    env_id = 'MountainCarContinuous-v0'

    env = VecNormalize(DummyVecEnv([lambda: Monitor(gym.make(env_id))]), norm_reward=True)
    eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(gym.make(env_id))]), training=False, norm_reward=False)

    model = model_class('MlpPolicy', env, n_steps=200, use_sde=True, ent_coef=0.00, verbose=1, learning_rate=3e-4,
                        policy_kwargs=dict(log_std_init=0.0, ortho_init=False), seed=None)
    model.learn(total_timesteps=int(1000), log_interval=5, eval_freq=500, eval_env=eval_env)


@pytest.mark.parametrize("model_class", [TD3])
def test_state_dependent_offpolicy_noise(model_class):
    model = model_class('MlpPolicy', 'Pendulum-v0', use_sde=True, seed=None, create_eval_env=True,
                        verbose=1, policy_kwargs=dict(log_std_init=-2))
    model.learn(total_timesteps=int(1000), eval_freq=500)
