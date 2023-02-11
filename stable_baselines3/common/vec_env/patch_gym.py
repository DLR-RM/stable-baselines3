import warnings
from typing import Callable

import gymnasium

try:
    import gym

    gym_installed = True
except ImportError:
    gym_installed = False

def _patch_env_generator(env_fn: Callable[[], gymnasium.Env]) -> Callable[[], gymnasium.Env]:
    """
    Taken from https://github.com/thu-ml/tianshou.

    Takes an environment generator and patches it to return Gymnasium envs.
    This function takes the environment generator ``env_fn`` and returns a patched
    generator, without invoking ``env_fn``. The original generator may return
    Gymnasium or OpenAI Gym environments, but the patched generator wraps
    the result of ``env_fn`` in a shimmy wrapper to convert it to Gymnasium,
    if necessary.

    :param env_fn: a function that returns an environment
    :return: Patched generator
    """

    def patched() -> gymnasium.Env:
        env = env_fn()

        # Gymnasium env, no patching to be done
        if isinstance(env, gymnasium.Env):
            return env

        if not gym_installed or not isinstance(env, gym.Env):
            raise ValueError(
                f"Environment generator returned a {type(env)}, not a Gymnasium "
                f"environment. In this case, we expect OpenAI Gym to be "
                f"installed and the environment to be an OpenAI Gym environment."
            )

        try:
            import shimmy
        except ImportError as e:
            raise ImportError(
                "Missing shimmy installation. You provided an environment generator "
                "that returned an OpenAI Gym environment. "
                "Stable-Baselines3 (SB3) has transitioned to using Gymnasium internally. "
                "In order to use OpenAI Gym environments with SB3, you need to "
                "install shimmy (`pip install shimmy`)."
            ) from e

        warnings.warn(
            "You provided an environment generator that returned an OpenAI Gym "
            "environment. We strongly recommend transitioning to Gymnasium "
            "environments. "
            "Stable-Baselines3 is automatically wrapping your environments in a compatibility "
            "layer, which could potentially cause issues."
        )

        # gym version only goes to 0.26.2
        gym_version = int(gym.__version__.split(".")[1])
        if gym_version >= 26:
            return shimmy.GymV26CompatibilityV0(env=env)
        elif gym_version >= 21:
            # TODO: rename to GymV21CompatibilityV0
            return shimmy.GymV22CompatibilityV0(env=env)
        else:
            raise Exception(
                f"Found OpenAI Gym version {gym.__version__}. " f"SB3 only supports OpenAI Gym environments of version>=0.21.0"
            )

    return patched
