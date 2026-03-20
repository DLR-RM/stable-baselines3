# RL Algorithms

This table displays the RL algorithms that are implemented in the Stable Baselines3 project,
along with some useful characteristics: support for discrete/continuous actions, multiprocessing.

| Name               | `Box` | `Discrete` | `MultiDiscrete` | `MultiBinary` | Multi Processing |
| ------------------ | ----- | ---------- | --------------- | ------------- | ---------------- |
| ARS [^f1]          | ✔️    | ✔️         | ❌              | ❌            | ✔️               |
| A2C                | ✔️    | ✔️         | ✔️              | ✔️            | ✔️               |
| CrossQ [^f1]       | ✔️    | ❌         | ❌              | ❌            | ✔️               |
| DDPG               | ✔️    | ❌         | ❌              | ❌            | ✔️               |
| DQN                | ❌    | ✔️         | ❌              | ❌            | ✔️               |
| HER                | ✔️    | ✔️         | ❌              | ❌            | ✔️               |
| PPO                | ✔️    | ✔️         | ✔️              | ✔️            | ✔️               |
| QR-DQN [^f1]       | ❌    | ️✔️        | ❌              | ❌            | ✔️               |
| RecurrentPPO [^f1] | ✔️    | ✔️         | ✔️              | ✔️            | ✔️               |
| SAC                | ✔️    | ❌         | ❌              | ❌            | ✔️               |
| TD3                | ✔️    | ❌         | ❌              | ❌            | ✔️               |
| TQC [^f1]          | ✔️    | ❌         | ❌              | ❌            | ✔️               |
| TRPO [^f1]         | ✔️    | ✔️         | ✔️              | ✔️            | ✔️               |
| Maskable PPO [^f1] | ❌    | ✔️         | ✔️              | ✔️            | ✔️               |

[^f1]: Implemented in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)

:::{note}
`Tuple` observation spaces are not supported by any environment,
however, single-level `Dict` spaces are (cf. {ref}`Examples <examples>`).
:::

Actions `gym.spaces`:

- `Box`: A N-dimensional box that contains every point in the action
  space.
- `Discrete`: A list of possible actions, where each timestep only
  one of the actions can be used.
- `MultiDiscrete`: A list of possible actions, where each timestep only one action of each discrete set can be used.
- `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.

:::{note}
More algorithms (like QR-DQN or TQC) are implemented in our [contrib repo](sb3_contrib.md)
and in our {ref}`SBX (SB3 + Jax) repo <sbx>` (DroQ, CrossQ, SimBa, ...).
:::

:::{note}
Some logging values (like `ep_rew_mean`, `ep_len_mean`) are only available when using a `Monitor` wrapper
See [Issue #339](https://github.com/hill-a/stable-baselines/issues/339) for more info.
:::

:::{note}
When using off-policy algorithms, [Time Limits](https://arxiv.org/abs/1712.00378) (aka timeouts) are handled
properly (cf. [issue #284](https://github.com/DLR-RM/stable-baselines3/issues/284)).
You can revert to SB3 < 2.1.0 behavior by passing `handle_timeout_termination=False`
via the `replay_buffer_kwargs` argument.
:::

## Reproducibility

Completely reproducible results are not guaranteed across PyTorch releases or different platforms.
Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.

In order to make computations deterministics, on your specific problem on one specific platform,
you need to pass a `seed` argument at the creation of a model.
If you pass an environment to the model using `set_env()`, then you also need to seed the environment first.

Credit: part of the *Reproducibility* section comes from [PyTorch Documentation](https://pytorch.org/docs/stable/notes/randomness.html)

## Training exceeds `total_timesteps`

When you train an agent using SB3, you pass a `total_timesteps` parameter to the `learn()` method which defines the training budget for the agent (how many interactions with the environment are allowed).
For example:

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1").learn(total_timesteps=1_000)
```

Because of the way the algorithms work, `total_timesteps` is a lower bound (see [issue #1150](https://github.com/DLR-RM/stable-baselines3/issues/1150)).
In the example above, PPO will effectively collect `n_steps * n_envs = 2048 * 1` steps despite `total_timesteps=1_000`
In more details:

- PPO/A2C and derivates collect `n_steps * n_envs` of experience
  before performing an update, so if you want to have exactly
  `total_timesteps`, you will need to adjust those values
- SAC/DQN/TD3 and other off-policy algorithms collect
  `train_freq * n_envs` steps before doing an update (when `train_freq` is in steps and not episodes), so if you want to have exactly `total_timesteps`
  you have to adjust these values (`train_freq=4` by default for DQN)
- ARS and other population-based algorithms evaluate the policy for
  `n_episodes` with `n_envs`, so unless the number of steps per
  episode is fixed, it is not possible to exactly achieve
  `total_timesteps`
- when using multiple envs, each call to `env.step()` corresponds to
  `n_envs` timesteps, so it is no longer possible to use the
  `EvaluationCallback` at an exact timestep
