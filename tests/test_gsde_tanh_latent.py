import torch as th

from stable_baselines3.common.distributions import StateDependentNoiseDistribution


def _saturated_distribution():
    dist = StateDependentNoiseDistribution(action_dim=1, squash_output=True, epsilon=1e-6)
    # latent feature 1 and log_std 0 give a unit-variance Gaussian
    log_std = th.zeros(1, 1)
    latent = th.ones(1, 1)
    mean = th.tensor([[12.0]], requires_grad=True)
    dist.proba_distribution(mean, log_std, latent)
    return dist, mean


def test_saturated_tanh_inverse_is_not_the_drawn_sample():
    gaussian = th.tensor([[12.783661]])
    action = th.tanh(gaussian)
    # float32 tanh of a large sample rounds to 1, and the clipped inverse is a different point
    action32 = action.to(th.float32)
    assert action32.item() == 1.0
    inverted = StateDependentNoiseDistribution(1, squash_output=True).bijector.inverse(action32)
    assert abs(inverted.item() - gaussian.item()) > 1.0


def test_log_prob_uses_the_drawn_gaussian_sample():
    dist, mean = _saturated_distribution()
    gaussian = th.tensor([[12.783661]])
    action = th.tanh(gaussian).to(th.float32)
    # The distribution parameters stay float32 so the saturated action is the stored one.
    dist.distribution.loc.data = dist.distribution.loc.data.to(th.float32)
    dist.distribution.scale.data = dist.distribution.scale.data.to(th.float32)
    mean32 = mean.detach().to(th.float32).requires_grad_(True)
    dist.distribution.loc = mean32

    retained = dist.log_prob(action, gaussian.to(th.float32))
    retained.backward()
    # d/d(mean) log N(x; mean, 1) = x - mean. The PPO loss negates this.
    expected = gaussian.to(th.float32) - mean32.detach()
    assert th.allclose(mean32.grad, expected, atol=1e-3)

    mean32.grad = None
    reconstructed = dist.log_prob(action)
    reconstructed.backward()
    assert mean32.grad.item() * expected.item() < 0
