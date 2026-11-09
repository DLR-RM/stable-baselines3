import warnings
from typing import Any

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.env_checker import _check_box_obs, _check_unsupported_spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def _check_vecenv_spaces(vec_env: VecEnv) -> None:
    """
    Check that the VecEnv has valid observation and action spaces.
    """
    assert hasattr(vec_env, "observation_space"), "VecEnv must have an observation_space attribute"
    assert hasattr(vec_env, "action_space"), "VecEnv must have an action_space attribute"
    assert hasattr(vec_env, "num_envs"), "VecEnv must have a num_envs attribute"

    assert isinstance(
        vec_env.observation_space, spaces.Space
    ), f"The observation space must inherit from gymnasium.spaces, got {type(vec_env.observation_space)}"
    assert isinstance(
        vec_env.action_space, spaces.Space
    ), f"The action space must inherit from gymnasium.spaces, got {type(vec_env.action_space)}"
    assert (
        isinstance(vec_env.num_envs, int) and vec_env.num_envs > 0
    ), f"num_envs must be a positive integer, got {vec_env.num_envs} (type: {type(vec_env.num_envs)})"


def _check_vecenv_reset(vec_env: VecEnv) -> Any:
    """
    Check that VecEnv reset method works correctly and returns properly shaped observations.
    """
    obs = vec_env.reset()

    # Check observation shape matches expected vectorized shape
    if isinstance(vec_env.observation_space, spaces.Box):
        assert isinstance(obs, np.ndarray), f"For Box observation space, reset() must return np.ndarray, got {type(obs)}"
        expected_shape = (vec_env.num_envs, *vec_env.observation_space.shape)
        assert obs.shape == expected_shape, (
            f"Expected observation shape {expected_shape}, got {obs.shape}. "
            f"VecEnv observations should have batch dimension first."
        )
    elif isinstance(vec_env.observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"For Dict observation space, reset() must return dict, got {type(obs)}"
        for key, space in vec_env.observation_space.spaces.items():
            assert key in obs, f"Missing key '{key}' in observation dict"
            if isinstance(space, spaces.Box):
                expected_shape = (vec_env.num_envs, *space.shape)
                assert (
                    obs[key].shape == expected_shape
                ), f"Expected observation['{key}'] shape {expected_shape}, got {obs[key].shape}"
    elif isinstance(vec_env.observation_space, spaces.Discrete):
        assert isinstance(obs, np.ndarray), f"For Discrete observation space, reset() must return np.ndarray, got {type(obs)}"
        expected_shape = (vec_env.num_envs,)
        assert obs.shape == expected_shape, f"Expected observation shape {expected_shape}, got {obs.shape}"

    return obs


def _check_vecenv_step(vec_env: VecEnv, obs: Any) -> None:
    """
    Check that VecEnv step method works correctly and returns properly shaped values.
    """
    # Generate valid actions
    actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])

    obs, rewards, dones, infos = vec_env.step(actions)

    # Check rewards
    assert isinstance(rewards, np.ndarray), f"step() must return rewards as np.ndarray, got {type(rewards)}"
    assert rewards.shape == (vec_env.num_envs,), f"Expected rewards shape ({vec_env.num_envs},), got {rewards.shape}"

    # Check dones
    assert isinstance(dones, np.ndarray), f"step() must return dones as np.ndarray, got {type(dones)}"
    assert dones.shape == (vec_env.num_envs,), f"Expected dones shape ({vec_env.num_envs},), got {dones.shape}"
    assert dones.dtype == bool, f"dones must have dtype bool, got {dones.dtype}"

    # Check infos
    assert isinstance(infos, (list, tuple)), f"step() must return infos as list or tuple, got {type(infos)}"
    assert len(infos) == vec_env.num_envs, f"Expected infos length {vec_env.num_envs}, got {len(infos)}"
    for i, info in enumerate(infos):
        assert isinstance(info, dict), f"infos[{i}] must be dict, got {type(info)}"

    # Check observation shape consistency (similar to reset)
    if isinstance(vec_env.observation_space, spaces.Box):
        assert isinstance(obs, np.ndarray), f"For Box observation space, step() must return np.ndarray, got {type(obs)}"
        expected_shape = (vec_env.num_envs, *vec_env.observation_space.shape)
        assert obs.shape == expected_shape, (
            f"Expected observation shape {expected_shape}, got {obs.shape}. "
            f"VecEnv observations should have batch dimension first."
        )
    elif isinstance(vec_env.observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"For Dict observation space, step() must return dict, got {type(obs)}"
        for key, space in vec_env.observation_space.spaces.items():
            assert key in obs, f"Missing key '{key}' in observation dict"
            if isinstance(space, spaces.Box):
                expected_shape = (vec_env.num_envs, *space.shape)
                assert (
                    obs[key].shape == expected_shape
                ), f"Expected observation['{key}'] shape {expected_shape}, got {obs[key].shape}"


class _DummyVecEnvForSpaceCheck:
    """Dummy class to pass to _check_unsupported_spaces function."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        self.observation_space = observation_space
        self.action_space = action_space


def _check_vecenv_unsupported_spaces(observation_space: spaces.Space, action_space: spaces.Space) -> bool:
    """
    Emit warnings when the observation space or action space used is not supported by Stable-Baselines
    for VecEnv. Reuses the existing _check_unsupported_spaces function.

    :return: True if return value tests should be skipped.
    """
    # Create a dummy env object to pass to the existing function
    dummy_env = _DummyVecEnvForSpaceCheck(observation_space, action_space)
    return _check_unsupported_spaces(dummy_env, observation_space, action_space)  # type: ignore[arg-type]


def check_vecenv(vec_env: VecEnv, warn: bool = True) -> None:
    """
    Check that a VecEnv follows the VecEnv API and is compatible with Stable-Baselines3.

    This checker verifies that:
    - The VecEnv has proper observation_space, action_space, and num_envs attributes
    - The reset() method returns observations with correct vectorized shape
    - The step() method returns observations, rewards, dones, and infos with correct shapes
    - All return values have the expected types and dimensions

    :param vec_env: The vectorized environment to check
    :param warn: Whether to output additional warnings mainly related to
        the interaction with Stable Baselines
    """
    assert isinstance(vec_env, VecEnv), "Your environment must inherit from stable_baselines3.common.vec_env.VecEnv"

    # ============= Check basic VecEnv attributes ================
    _check_vecenv_spaces(vec_env)

    # Define aliases for convenience
    observation_space = vec_env.observation_space
    action_space = vec_env.action_space

    # Warn the user if needed - reuse existing space checking logic
    if warn:
        should_skip = _check_vecenv_unsupported_spaces(observation_space, action_space)
        if should_skip:
            warnings.warn("VecEnv contains unsupported spaces, skipping some checks")
            return

        obs_spaces = observation_space.spaces if isinstance(observation_space, spaces.Dict) else {"": observation_space}
        for key, space in obs_spaces.items():
            if isinstance(space, spaces.Box):
                _check_box_obs(space, key)

        # Check for the action space
        if isinstance(action_space, spaces.Box) and (
            np.any(np.abs(action_space.low) != np.abs(action_space.high))
            or np.any(action_space.low != -1)
            or np.any(action_space.high != 1)
        ):
            warnings.warn(
                "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
                "cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"
            )

        if isinstance(action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([action_space.low, action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        if isinstance(action_space, spaces.Box) and action_space.dtype != np.dtype(np.float32):
            warnings.warn(
                f"Your action space has dtype {action_space.dtype}, we recommend using np.float32 to avoid cast errors."
            )

    # ============ Check the VecEnv methods ===============
    obs = _check_vecenv_reset(vec_env)
    _check_vecenv_step(vec_env, obs)
