import pytest
import torch as th
from torch.distributions import Normal

from stable_baselines3 import A2C, SAC, PPO


def test_state_dependent_exploration_grad():
    """
    Check that the gradient correspond to the expected one
    """
    n_states = 2
    state_dim = 3
    action_dim = 10
    sigma_hat = th.ones(state_dim, action_dim, requires_grad=True)
    # Reduce the number of parameters
    # sigma_ = th.ones(state_dim, action_dim) * sigma_
    # weights_dist = Normal(th.zeros_like(log_sigma), th.exp(log_sigma))
    th.manual_seed(2)
    weights_dist = Normal(th.zeros_like(sigma_hat), sigma_hat)
    weights = weights_dist.rsample()

    state = th.rand(n_states, state_dim)
    mu = th.ones(action_dim)
    noise = th.mm(state, weights)

    action = mu + noise

    variance = th.mm(state ** 2, sigma_hat ** 2)
    action_dist = Normal(mu, th.sqrt(variance))

    # Sum over the action dimension because we assume they are independent
    loss = action_dist.log_prob(action.detach()).sum(dim=-1).mean()
    loss.backward()

    # From Rueckstiess paper: check that the computed gradient
    # correspond to the analytical form
    grad = th.zeros_like(sigma_hat)
    for j in range(action_dim):
        # sigma_hat is the std of the gaussian distribution of the noise matrix weights
        # sigma_j = sum_j(state_i **2 * sigma_hat_ij ** 2)
        # sigma_j is the standard deviation of the policy gaussian distribution
        sigma_j = th.sqrt(variance[:, j])
        for i in range(state_dim):
            # Derivative of the log probability of the jth component of the action
            # w.r.t. the standard deviation sigma_j
            d_log_policy_j = (noise[:, j] ** 2 - sigma_j ** 2) / sigma_j ** 3
            # Derivative of sigma_j w.r.t. sigma_hat_ij
            d_log_sigma_j = (state[:, i] ** 2 * sigma_hat[i, j]) / sigma_j
            # Chain rule, average over the minibatch
            grad[i, j] = (d_log_policy_j * d_log_sigma_j).mean()

    # sigma.grad should be equal to grad
    assert sigma_hat.grad.allclose(grad)


@pytest.mark.parametrize("model_class", [SAC, A2C, PPO])
@pytest.mark.parametrize("sde_net_arch", [None, [32, 16], []])
@pytest.mark.parametrize("use_expln", [False, True])
def test_state_dependent_offpolicy_noise(model_class, sde_net_arch, use_expln):
    model = model_class('MlpPolicy', 'Pendulum-v0', use_sde=True, seed=None, create_eval_env=True,
                        verbose=1, policy_kwargs=dict(log_std_init=-2, sde_net_arch=sde_net_arch, use_expln=use_expln))
    model.learn(total_timesteps=int(500), eval_freq=250)
