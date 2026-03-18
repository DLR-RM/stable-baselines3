(custom-env)=

# Using Custom Environments

To use the RL baselines with custom environments, they just need to follow the *gymnasium* [interface](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py).
That is to say, your environment must implement the following methods (and inherits from Gym Class):

:::{note}
If you are using images as input, the observation must be of type `np.uint8` and be within a space `Box` bounded by [0, 255] (`Box(low=0, high=255, shape=(<your image shape>)`).
By default, the observation is normalized by SB3 pre-processing (dividing by 255 to have values in [0, 1], i.e. `Box(low=0, high=1)`) when using CNN policies.
Images can be either channel-first or channel-last.

If you want to use `CnnPolicy` or `MultiInputPolicy` with image-like observation (3D tensor) that are already normalized, you must pass `normalize_images=False`
to the policy (using `policy_kwargs` parameter, `policy_kwargs=dict(normalize_images=False)`)
and make sure your image is in the **channel-first** format.
:::

:::{note}
Although SB3 supports both channel-last and channel-first images as input, we recommend using the channel-first convention when possible.
Under the hood, when a channel-last image is passed, SB3 uses a `VecTransposeImage` wrapper to re-order the channels.
:::

:::{note}
SB3 doesn't support `Discrete` and `MultiDiscrete` spaces with `start!=0`. However, you can update your environment or use a wrapper to make your env compatible with SB3:

```python
import gymnasium as gym

class ShiftWrapper(gym.Wrapper):
    """Allow to use Discrete() action spaces with start!=0"""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(env.action_space.n, start=0)
    
    def step(self, action: int):
        return self.env.step(action + self.env.action_space.start)
```
:::

:::{note}
SB3 doesn't support `MultiDiscrete` spaces with multi-dimensional arrays. However, you can update your environment or use a wrapper to make your env compatible with SB3:

```python
import numpy as np
import gymnasium as gym

class ReshapeWrapper(gym.Wrapper):
    """Allow to use MultiDiscrete() action spaces with len(nvec.shape) > 1:"""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.original_shape = env.action_space.nvec.shape
        self.action_space = gym.spaces.MultiDiscrete(env.action_space.nvec.flatten())
    
    def step(self, action: np.ndarray):
        return self.env.step(action.reshape(self.original_shape))
```
:::

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, arg1, arg2, ...):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...
```

Then you can define and train a RL agent with:

```python
# Instantiate the env
env = CustomEnv(arg1, ...)
# Define and Train the agent
model = A2C("CnnPolicy", env).learn(total_timesteps=1000)
```

To check that your environment follows the Gym interface that SB3 supports, please use:

```python
from stable_baselines3.common.env_checker import check_env

env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)
```

Gymnasium also have its own [env checker](https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env) but it checks a superset of what SB3 supports (SB3 does not support all Gym features).

We have created a [colab notebook](https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb) for a concrete example of creating a custom environment along with an example of using it with Stable-Baselines3 interface.

Alternatively, you may look at Gymnasium [built-in environments](https://gymnasium.farama.org).

Optionally, you can also register the environment with gym, that will allow you to create the RL agent in one line (and use `gym.make()` to instantiate the env):

```python
from gymnasium.envs.registration import register
# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="CartPole-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="gym.envs.classic_control:CartPoleEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)
```

In the project, for testing purposes, we use a custom environment named `IdentityEnv`
defined [in this file](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/envs/identity_env.py).
An example of how to use it can be found [here](https://github.com/DLR-RM/stable-baselines3/blob/master/tests/test_identity.py).
