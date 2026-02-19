(vec-env)=

```{eval-rst}
.. automodule:: stable_baselines3.common.vec_env
```

# Vectorized Environments

Vectorized Environments are a method for stacking multiple independent environments into a single environment.
Instead of training an RL agent on 1 environment per step, it allows us to train it on `n` environments per step.
Because of this, `actions` passed to the environment are now a vector (of dimension `n`).
It is the same for `observations`, `rewards` and end of episode signals (`dones`).
In the case of non-array observation spaces such as `Dict` or `Tuple`, where different sub-spaces
may have different shapes, the sub-observations are vectors (of dimension `n`).

| Name          | `Box` | `Discrete` | `Dict` | `Tuple` | Multi Processing |
| ------------- | ----- | ---------- | ------ | ------- | ---------------- |
| DummyVecEnv   | ✔️    | ✔️         | ✔️     | ✔️      | ❌️               |
| SubprocVecEnv | ✔️    | ✔️         | ✔️     | ✔️      | ✔️               |

:::{note}
Vectorized environments are required when using wrappers for frame-stacking or normalization.
:::

:::{note}
When using vectorized environments, the environments are automatically reset at the end of each episode.
Thus, the observation returned for the i-th environment when `done[i]` is true will in fact be the first observation of the next episode, not the last observation of the episode that has just terminated.
You can access the "real" final observation of the terminated episode—that is, the one that accompanied the `done` event provided by the underlying environment—using the `terminal_observation` keys in the info dicts returned by the `VecEnv`.
:::

:::{warning}
When defining a custom `VecEnv` (for instance, using gym3 `ProcgenEnv`), you should provide `terminal_observation` keys in the info dicts returned by the `VecEnv`
(cf. note above).
:::

:::{warning}
When using `SubprocVecEnv`, users must wrap the code in an `if __name__ == "__main__":` if using the `forkserver` or `spawn` start method (default on Windows).
On Linux, the default start method is `fork` which is not thread safe and can create deadlocks.

For more information, see Python's [multiprocessing guidelines](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).
:::

## VecEnv API vs Gym API

For consistency across Stable-Baselines3 (SB3) versions and because of its special requirements and features,
SB3 VecEnv API is not the same as Gym API.
SB3 VecEnv API is actually close to Gym 0.21 API but differs to Gym 0.26+ API:

- the `reset()` method only returns the observation (`obs = vec_env.reset()`) and not a tuple, the info at reset are stored in `vec_env.reset_infos`.

- only the initial call to `vec_env.reset()` is required, environments are reset automatically afterward (and `reset_infos` is updated automatically).

- the `vec_env.step(actions)` method expects an array as input
  (with a batch size corresponding to the number of environments) and returns a 4-tuple (and not a 5-tuple): `obs, rewards, dones, infos` instead of `obs, reward, terminated, truncated, info`
  where `dones = terminated or truncated` (for each env).
  `obs, rewards, dones` are NumPy arrays with shape `(n_envs, shape_for_single_env)` (so with a batch dimension).
  Additional information is passed via the `infos` value which is a list of dictionaries.

- at the end of an episode, `infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated`
  tells the user if an episode was truncated or not:
  you should bootstrap if `infos[env_idx]["TimeLimit.truncated"] is True` (episode over due to a timeout/truncation)
  or `dones[env_idx] is False` (episode not finished).
  Note: compared to Gym 0.26+ `infos[env_idx]["TimeLimit.truncated"]` and `terminated` [are mutually exclusive](https://github.com/openai/gym/issues/3102).
  The conversion from SB3 to Gym API is

  ```python
  # done is True at the end of an episode
  # dones[env_idx] = terminated[env_idx] or truncated[env_idx]
  # In SB3, truncated and terminated are mutually exclusive
  # infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
  # terminated[env_idx] tells you whether you should bootstrap or not:
  # when the episode has not ended or when the termination was a timeout/truncation
  terminated[env_idx] = dones[env_idx] and not infos[env_idx]["TimeLimit.truncated"]
  should_bootstrap[env_idx] = not terminated[env_idx]
  ```

- at the end of an episode, because the environment resets automatically,
  we provide `infos[env_idx]["terminal_observation"]` which contains the last observation
  of an episode (and can be used when bootstrapping, see note in the previous section)

- to overcome the current Gymnasium limitation (only one render mode allowed per env instance, see [issue #100](https://github.com/Farama-Foundation/Gymnasium/issues/100)),
  we recommend using `render_mode="rgb_array"` since we can both have the image as a numpy array and display it with OpenCV.
  if no mode is passed or `mode="rgb_array"` is passed when calling `vec_env.render` then we use the default mode, otherwise, we use the OpenCV display.
  Note that if `render_mode != "rgb_array"`, you can only call `vec_env.render()` (without argument or with `mode=env.render_mode`).

- the `reset()` method doesn't take any parameter. If you want to seed the pseudo-random generator or pass options,
  you should call `vec_env.seed(seed=seed)`/`vec_env.set_options(options)` and `obs = vec_env.reset()` afterward (seed and options are discarded after each call to `reset()`).

- methods and attributes of the underlying Gym envs can be accessed, called and set using `vec_env.get_attr("attribute_name")`,
  `vec_env.env_method("method_name", args1, args2, kwargs1=kwargs1)` and `vec_env.set_attr("attribute_name", new_value)`.

## Modifying Vectorized Environments Attributes

If you plan to [modify the attributes of an environment](https://github.com/DLR-RM/stable-baselines3/issues/1573) while it is used (e.g., modifying an attribute specifying the task carried out for a portion of training when doing multi-task learning, or
a parameter of the environment dynamics), you must expose a setter method.
In fact, directly accessing the environment attribute in the callback can lead to unexpected behavior because environments can be wrapped (using gym or VecEnv wrappers, the `Monitor` wrapper being one example).

Consider the following example for a custom env:

```python
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_util import make_vec_env


class MyMultiTaskEnv(gym.Env):

  def __init__(self):
      super().__init__()
      """
      A state and action space for robotic locomotion.
      The multi-task twist is that the policy would need to adapt to different terrains, each with its own
      friction coefficient, mu.
      The friction coefficient is the only parameter that changes between tasks.
      mu is a scalar between 0 and 1, and during training a callback is used to update mu.
      """
      ...

  def step(self, action):
    # Do something, depending on the action and current value of mu the next state is computed
    return self._get_obs(), reward, done, truncated, info

  def set_mu(self, new_mu: float) -> None:
      # Note: this value should be used only at the next reset
      self.mu = new_mu

# Example of wrapped env
# env is of type <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
env = gym.make("CartPole-v1")
# To access the base env, without wrapper, you should use `.unwrapped`
# or env.get_wrapper_attr("gravity") to include wrappers
env.unwrapped.gravity
# SB3 uses VecEnv for training, where `env.unwrapped.x = new_value` cannot be used to set an attribute
# therefore, you should expose a setter like `set_mu` to properly set an attribute
vec_env = make_vec_env(MyMultiTaskEnv)
# Print current mu value
# Note: you should use vec_env.env_method("get_wrapper_attr", "mu") in Gymnasium v1.0
print(vec_env.env_method("get_wrapper_attr", "mu"))
# Change `mu` attribute via the setter
vec_env.env_method("set_mu", 0.1)
# If the variable exists, you can also use `set_wrapper_attr` to set it
assert vec_env.has_attr("mu")
vec_env.env_method("set_wrapper_attr", "mu", 0.1)
```

In this example `env.mu` cannot be accessed/changed directly because it is wrapped in a `VecEnv` and because it could be wrapped with other wrappers (see [GH#1573](https://github.com/DLR-RM/stable-baselines3/issues/1573) for a longer explanation).
Instead, the callback should use the `set_mu` method via the `env_method` method for Vectorized Environments.

```python
from itertools import cycle

class ChangeMuCallback(BaseCallback):
  """
  This callback changes the value of mu during training looping
  through a list of values until training is aborted.
  The environment is implemented so that the impact of changing
  the value of mu mid-episode is visible only after the episode is over
  and the reset method has been called.
  """
  def __init__(self):
    super().__init__()
    # An iterator that contains the different of the friction coefficient
    self.mus = cycle([0.1, 0.2, 0.5, 0.13, 0.9])

  def _on_step(self):
    # Note: in practice, you should not change this value at every step
    # but rather depending on some events/metrics like agent performance/episode termination
    # both accessible via the `self.logger` or `self.locals` variables
    self.training_env.env_method("set_mu", next(self.mus))
```

This callback can then be used to safely modify environment attributes during training since
it calls the environment setter method.

## Vectorized Environments Wrappers

If you want to alter or augment a `VecEnv` without redefining it completely (e.g. stack multiple frames, monitor the `VecEnv`, normalize the observation, ...), you can use `VecEnvWrapper` for that.
They are the vectorized equivalents (i.e., they act on multiple environments at the same time) of `gym.Wrapper`.

You can find below an example for extracting one key from the observation:

```python
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs[self.key]

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs[self.key], reward, done, info

env = DummyVecEnv([lambda: gym.make("FetchReach-v1")])
# Wrap the VecEnv
env = VecExtractDictObs(env, key="observation")
```

:::{note}
When creating a vectorized environment, you can also specify ordinary gymnasium
wrappers to wrap each of the sub-environments. See the
{func}`make_vec_env <stable_baselines3.common.env_util.make_vec_env>`
documentation for details.
Example:

```python
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_util import make_vec_env

# Use gym wrapper for each sub-env of the VecEnv
wrapper_kwargs = dict(min_action=-1.0, max_action=1.0)
vec_env = make_vec_env(
    "Pendulum-v1", n_envs=2, wrapper_class=RescaleAction, wrapper_kwargs=wrapper_kwargs
)
```
:::

## VecEnv

```{eval-rst}
.. autoclass:: VecEnv
  :members:
```

## DummyVecEnv

```{eval-rst}
.. autoclass:: DummyVecEnv
  :members:
```

## SubprocVecEnv

```{eval-rst}
.. autoclass:: SubprocVecEnv
  :members:
```

## Wrappers

### VecFrameStack

```{eval-rst}
.. autoclass:: VecFrameStack
  :members:
```

### StackedObservations

```{eval-rst}
.. autoclass:: stable_baselines3.common.vec_env.stacked_observations.StackedObservations
  :members:
```

### VecNormalize

```{eval-rst}
.. autoclass:: VecNormalize
  :members:

```

### VecVideoRecorder

```{eval-rst}
.. autoclass:: VecVideoRecorder
  :members:

```

### VecCheckNan

```{eval-rst}
.. autoclass:: VecCheckNan
  :members:

```

### VecTransposeImage

```{eval-rst}
.. autoclass:: VecTransposeImage
  :members:
```

### VecMonitor

```{eval-rst}
.. autoclass:: VecMonitor
  :members:
```

### VecExtractDictObs

```{eval-rst}
.. autoclass:: VecExtractDictObs
  :members:
```
