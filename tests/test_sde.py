import pytest

import torch as th
from torch.distributions import Normal

from torchy_baselines import A2C


def test_state_dependent_exploration():
    state_dim = 3
    # TODO: fix for action_dim > 1
    action_dim = 1
    sigma = th.ones(state_dim, action_dim, requires_grad=True)

    # log_sigma = th.ones(2, 1, requires_grad=True)

    # weights_dist = Normal(th.zeros_like(log_sigma), th.exp(log_sigma))
    th.manual_seed(2)
    weights_dist = Normal(th.zeros_like(sigma), sigma)

    weights = weights_dist.rsample()
    state = th.rand(1, state_dim)
    # state = (th.ones(state_dim,) * 2).view(1, -1)
    mu = th.ones(action_dim)
    # print(weights.shape, state.shape)
    noise = th.mm(state, weights)
    # variance = th.mm(state ** 2, th.exp(log_sigma) ** 2)
    variance = th.mm(state ** 2, sigma ** 2)
    action_dist = Normal(mu, th.sqrt(variance))

    loss = action_dist.log_prob((mu + noise).detach()).mean()
    loss.backward()

    # From Rueckstiess paper
    grad = th.zeros_like(sigma)
    for j in range(action_dim):
        for i in range(state_dim):
            grad[i, j] = ((noise[:, j] ** 2 - variance[:, j]) / (variance[:, j] ** 2)) * (state[:, i] ** 2 * sigma[i, j])

    # sigma.grad should be equal to grad
    assert sigma.grad.allclose(grad)


@pytest.mark.parametrize("model_class", [A2C])
def test_state_dependent_noise(model_class):
    model = model_class('MlpPolicy', 'Pendulum-v0', n_steps=200,
                        use_sde=True, ent_coef=0.0, verbose=1, create_eval_env=True)
    model.learn(total_timesteps=int(1e6), log_interval=10, eval_freq=10000)
