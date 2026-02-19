(ppo2)=

```{eval-rst}
.. automodule:: stable_baselines3.ppo
```

# PPO

The [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far from the old policy.
For that, ppo uses clipping to avoid too large update.

:::{note}
PPO contains several modifications from the original algorithm not documented
by OpenAI: advantages are normalized and value function can be also clipped.
:::

## Notes

- Original paper: <https://arxiv.org/abs/1707.06347>
- Clear explanation of PPO on Arxiv Insights channel: <https://www.youtube.com/watch?v=5P7I-xPq8u8>
- OpenAI blog post: <https://openai.com/research/openai-baselines-ppo>
- Spinning Up guide: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>
- 37 implementation details blog: <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/>

## Can I use?

:::{note}
A recurrent version of PPO is available in our contrib repo: <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html>

However we advise users to start with simple frame-stacking as a simpler, faster
and usually competitive alternative, more info in our report: <https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO-aka-PPO-LSTM-on-environments-with-masked-velocity--VmlldzoxOTI4NjE4>
See also [Procgen paper appendix Fig 11.](https://arxiv.org/abs/1912.01588).
In practice, you can stack multiple observations using `VecFrameStack`.
:::

- Recurrent policies: ❌
- Multi processing: ✔️
- Gym spaces:

| Space         | Action | Observation |
| ------------- | ------ | ----------- |
| Discrete      | ✔️     | ✔️          |
| Box           | ✔️     | ✔️          |
| MultiDiscrete | ✔️     | ✔️          |
| MultiBinary   | ✔️     | ✔️          |
| Dict          | ❌     | ✔️          |

## Example

This example is only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized hyperparameters can be found in RL Zoo [repository](https://github.com/DLR-RM/rl-baselines3-zoo).

Train a PPO agent on `CartPole-v1` using 4 environments.

```python
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
```

:::{note}
PPO is meant to be run primarily on the CPU, especially when you are not using a CNN. To improve CPU utilization, try turning off the GPU and using `SubprocVecEnv` instead of the default `DummyVecEnv`:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__=="__main__":
    env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, device="cpu")
    model.learn(total_timesteps=25_000)
```

For more information, see [Vectorized Environments](../guide/vec_envs.md), [Issue #1245](https://github.com/DLR-RM/stable-baselines3/issues/1245#issuecomment-1435766949) or the [Multiprocessing notebook](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb).
:::

:::{note}
Using gSDE (Generalized State-Dependent Exploration) during inference (see [PR #1767](https://github.com/DLR-RM/stable-baselines3/pull/1767)):

When using PPO models trained with `use_sde=True`, the automatic noise resetting that occurs during training (controlled by `sde_sample_freq`) does not happen when using `model.predict()` for inference. This results in deterministic behavior even when `deterministic=False`.

For continuous control tasks, it is recommended to use deterministic behavior during inference (`deterministic=True`). If you need stochastic behavior during inference, you must manually reset the noise by calling `model.policy.reset_noise(env.num_envs)` at appropriate intervals based on your desired `sde_sample_freq`.
:::

## Results

### Atari Games

The complete learning curves are available in the [associated PR #110](https://github.com/DLR-RM/stable-baselines3/pull/110).

### PyBullet Environments

Results on the PyBullet benchmark (2M steps) using 6 seeds.
The complete learning curves are available in the [associated issue #48](https://github.com/DLR-RM/stable-baselines3/issues/48).

:::{note}
Hyperparameters from the [gSDE paper](https://arxiv.org/abs/2005.05719) were used (as they are tuned for PyBullet envs).
:::

*Gaussian* means that the unstructured Gaussian noise is used for exploration,
*gSDE* (generalized State-Dependent Exploration) is used otherwise.

| Environments | A2C          | A2C          | PPO          | PPO         |
| ------------ | ------------ | ------------ | ------------ | ----------- |
|              | Gaussian     | gSDE         | Gaussian     | gSDE        |
| HalfCheetah  | 2003 +/- 54  | 2032 +/- 122 | 1976 +/- 479 | 2826 +/- 45 |
| Ant          | 2286 +/- 72  | 2443 +/- 89  | 2364 +/- 120 | 2782 +/- 76 |
| Hopper       | 1627 +/- 158 | 1561 +/- 220 | 1567 +/- 339 | 2512 +/- 21 |
| Walker2D     | 577 +/- 65   | 839 +/- 56   | 1230 +/- 147 | 2019 +/- 64 |

### How to replicate the results?

Clone the [rl-zoo repo](https://github.com/DLR-RM/rl-baselines3-zoo):

```bash
git clone https://github.com/DLR-RM/rl-baselines3-zoo
cd rl-baselines3-zoo/
```

Run the benchmark (replace `$ENV_ID` by the envs mentioned above):

```bash
python train.py --algo ppo --env $ENV_ID --eval-episodes 10 --eval-freq 10000
```

Plot the results (here for PyBullet envs only):

```bash
python scripts/all_plots.py -a ppo -e HalfCheetah Ant Hopper Walker2D -f logs/ -o logs/ppo_results
python scripts/plot_from_file.py -i logs/ppo_results.pkl -latex -l PPO
```

## Parameters

```{eval-rst}
.. autoclass:: PPO
  :members:
  :inherited-members:

```

(ppo_policies)=

## PPO Policies

```{eval-rst}
.. autoclass:: MlpPolicy
  :members:
  :inherited-members:
```

```{eval-rst}
.. autoclass:: stable_baselines3.common.policies.ActorCriticPolicy
  :members:
  :noindex:
```

```{eval-rst}
.. autoclass:: CnnPolicy
  :members:
```

```{eval-rst}
.. autoclass:: stable_baselines3.common.policies.ActorCriticCnnPolicy
  :members:
  :noindex:
```

```{eval-rst}
.. autoclass:: MultiInputPolicy
  :members:
```

```{eval-rst}
.. autoclass:: stable_baselines3.common.policies.MultiInputActorCriticPolicy
  :members:
  :noindex:
```
