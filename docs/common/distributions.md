(distributions)=

# Probability Distributions

Probability distributions used for the different action spaces:

- `CategoricalDistribution` -> Discrete
- `DiagGaussianDistribution` -> Box (continuous actions)
- `BetaDistribution` -> Box (continuous actions) when `use_beta=True`
- `StateDependentNoiseDistribution` -> Box (continuous actions) when `use_sde=True`

% - ``MultiCategoricalDistribution`` -> MultiDiscrete

% - ``BernoulliDistribution`` -> MultiBinary

The policy networks output parameters for the distributions (named `flat` in the methods).
Actions are then sampled from those distributions.

For instance, in the case of discrete actions. The policy network outputs probability
of taking each action. The `CategoricalDistribution` allows sampling from it,
computes the entropy, the log probability (`log_prob`) and backpropagate the gradient.

In the case of continuous actions, a Gaussian distribution is used by default. The policy network outputs
mean and (log) std of the distribution (assumed to be a `DiagGaussianDistribution`).

Alternatively, a `BetaDistribution` can be used for continuous actions in bounded spaces
by passing `policy_kwargs=dict(use_beta=True)`. The Beta distribution has bounded support on [0, 1],
so sampled actions naturally respect bounds without clipping. Actions are rescaled from [0, 1]
to the environment's action space `[low, high]`. The policy network outputs raw α and β parameters,
which are passed through softplus + 1 to ensure α, β ≥ 1 (unimodal regime).
The idea was first introduced by [Chou et al. (2017)](https://proceedings.mlr.press/v70/chou17a.html)
and further explored in [*The Beta Policy for Continuous Control Reinforcement Learning*](https://arxiv.org/abs/2111.02202).

```{eval-rst}
.. automodule:: stable_baselines3.common.distributions
  :members:
```
