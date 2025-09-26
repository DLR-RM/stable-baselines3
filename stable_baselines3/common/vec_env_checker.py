import warnings
from typing import Any, Union

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def _is_oneof_space(space: spaces.Space) -> bool:
    """
    Return True if the provided space is a OneOf space,
    False if not or if the current version of Gym doesn't support this space.
    """
    try:
        return isinstance(space, spaces.OneOf)  # type: ignore[attr-defined]
    except AttributeError:
        # Gym < v1.0
        return False


def _is_numpy_array_space(space: spaces.Space) -> bool:
    """
    Returns False if provided space is not representable as a single numpy array
    (e.g. Dict and Tuple spaces return False)
    """
    return not isinstance(space, (spaces.Dict, spaces.Tuple))


def _starts_at_zero(space: Union[spaces.Discrete, spaces.MultiDiscrete]) -> bool:
    """
    Return False if a (Multi)Discrete space has a non-zero start.
    """
    return np.allclose(space.start, np.zeros_like(space.start))


def _check_non_zero_start(space: spaces.Space, space_type: str = "observation", key: str = "") -> None:
    """
    :param space: Observation or action space
    :param space_type: information about whether it is an observation or action space
        (for the warning message)
    :param key: When the observation space comes from a Dict space, we pass the
        corresponding key to have more precise warning messages. Defaults to "".
    """
    if isinstance(space, (spaces.Discrete, spaces.MultiDiscrete)) and not _starts_at_zero(space):
        maybe_key = f"(key='{key}')" if key else ""
        warnings.warn(
            f"{type(space).__name__} {space_type} space {maybe_key} with a non-zero start (start={space.start}) "
            "is not supported by Stable-Baselines3. "
            "You can use a wrapper (see https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html) "
            f"or update your {space_type} space."
        )


def _check_image_input(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the input will be compatible with Stable-Baselines
    when the observation is apparently an image.

    :param observation_space: Observation space
    :param key: When the observation space comes from a Dict space, we pass the
        corresponding key to have more precise warning messages. Defaults to "".
    """
    if observation_space.dtype != np.uint8:
        warnings.warn(
            f"It seems that your observation {key} is an image but its `dtype` "
            f"is ({observation_space.dtype}) whereas it has to be `np.uint8`. "
            "If your observation is not an image, we recommend you to flatten the observation "
            "to have only a 1D vector"
        )

    if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
        warnings.warn(
            f"It seems that your observation space {key} is an image but the "
            "upper and lower bounds are not in [0, 255]. "
            "Because the CNN policy normalize automatically the observation "
            "you may encounter issue if the values are not in that range."
        )


def _check_box_obs(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the observation space is correctly formatted
    when dealing with a ``Box()`` space. In particular, it checks:
    - that the dimensions are big enough when it is an image, and that the type matches
    - that the observation has an expected shape (warn the user if not)
    """
    # If image, check the low and high values, the type and the number of channels
    # and the shape (minimal value)
    if len(observation_space.shape) == 3:
        _check_image_input(observation_space, key)

    if len(observation_space.shape) not in [1, 3]:
        warnings.warn(
            f"Your observation {key} has an unconventional shape (neither an image, nor a 1D vector). "
            "We recommend you to flatten the observation "
            "to have only a 1D vector or use a custom policy to properly process the data."
        )


def _check_vecenv_spaces(vec_env: VecEnv) -> None:
    """
    Check that the VecEnv has valid observation and action spaces.
    """
    assert hasattr(vec_env, "observation_space"), "VecEnv must have an observation_space attribute"
    assert hasattr(vec_env, "action_space"), "VecEnv must have an action_space attribute"
    assert hasattr(vec_env, "num_envs"), "VecEnv must have a num_envs attribute"

    assert isinstance(
        vec_env.observation_space, spaces.Space
    ), "The observation space must inherit from gymnasium.spaces"
    assert isinstance(vec_env.action_space, spaces.Space), "The action space must inherit from gymnasium.spaces"
    assert isinstance(vec_env.num_envs, int) and vec_env.num_envs > 0, "num_envs must be a positive integer"


def _check_vecenv_reset(vec_env: VecEnv) -> Any:
    """
    Check that VecEnv reset method works correctly and returns properly shaped observations.
    """
    try:
        obs = vec_env.reset()
    except Exception as e:
        raise RuntimeError(f"VecEnv reset() failed: {e}") from e

    # Check observation shape matches expected vectorized shape
    if isinstance(vec_env.observation_space, spaces.Box):
        assert isinstance(obs, np.ndarray), f"For Box observation space, reset() must return np.ndarray, got {type(obs)}"
        expected_shape = (vec_env.num_envs,) + vec_env.observation_space.shape
        assert obs.shape == expected_shape, (
            f"Expected observation shape {expected_shape}, got {obs.shape}. "
            f"VecEnv observations should have batch dimension first."
        )
    elif isinstance(vec_env.observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"For Dict observation space, reset() must return dict, got {type(obs)}"
        for key, space in vec_env.observation_space.spaces.items():
            assert key in obs, f"Missing key '{key}' in observation dict"
            if isinstance(space, spaces.Box):
                expected_shape = (vec_env.num_envs,) + space.shape
                assert obs[key].shape == expected_shape, (
                    f"Expected observation['{key}'] shape {expected_shape}, got {obs[key].shape}"
                )
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
    if isinstance(vec_env.action_space, spaces.Box):
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
    elif isinstance(vec_env.action_space, spaces.Discrete):
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
    elif isinstance(vec_env.action_space, spaces.MultiDiscrete):
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
    elif isinstance(vec_env.action_space, spaces.MultiBinary):
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
    else:
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])

    try:
        obs, rewards, dones, infos = vec_env.step(actions)
    except Exception as e:
        raise RuntimeError(f"VecEnv step() failed: {e}") from e

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
        expected_shape = (vec_env.num_envs,) + vec_env.observation_space.shape
        assert obs.shape == expected_shape, (
            f"Expected observation shape {expected_shape}, got {obs.shape}. "
            f"VecEnv observations should have batch dimension first."
        )
    elif isinstance(vec_env.observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"For Dict observation space, step() must return dict, got {type(obs)}"
        for key, space in vec_env.observation_space.spaces.items():
            assert key in obs, f"Missing key '{key}' in observation dict"
            if isinstance(space, spaces.Box):
                expected_shape = (vec_env.num_envs,) + space.shape
                assert obs[key].shape == expected_shape, (
                    f"Expected observation['{key}'] shape {expected_shape}, got {obs[key].shape}"
                )


def _check_vecenv_unsupported_spaces(observation_space: spaces.Space, action_space: spaces.Space) -> bool:
    """
    Emit warnings when the observation space or action space used is not supported by Stable-Baselines
    for VecEnv. This is a VecEnv-specific version of _check_unsupported_spaces.

    :return: True if return value tests should be skipped.
    """
    should_skip = graph_space = sequence_space = False
    if isinstance(observation_space, spaces.Dict):
        nested_dict = False
        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Dict):
                nested_dict = True
            elif isinstance(space, spaces.Graph):
                graph_space = True
            elif isinstance(space, spaces.Sequence):
                sequence_space = True
            _check_non_zero_start(space, "observation", key)

        if nested_dict:
            warnings.warn(
                "Nested observation spaces are not supported by Stable Baselines3 "
                "(Dict spaces inside Dict space). "
                "You should flatten it to have only one level of keys."
                "For example, `dict(space1=dict(space2=Box(), space3=Box()), spaces4=Discrete())` "
                "is not supported but `dict(space2=Box(), spaces3=Box(), spaces4=Discrete())` is."
            )

    if isinstance(observation_space, spaces.MultiDiscrete) and len(observation_space.nvec.shape) > 1:
        warnings.warn(
            f"The MultiDiscrete observation space uses a multidimensional array {observation_space.nvec} "
            "which is currently not supported by Stable-Baselines3. "
            "Please convert it to a 1D array using a wrapper: "
            "https://github.com/DLR-RM/stable-baselines3/issues/1836."
        )

    if isinstance(observation_space, spaces.Tuple):
        warnings.warn(
            "The observation space is a Tuple, "
            "this is currently not supported by Stable Baselines3. "
            "However, you can convert it to a Dict observation space "
            "(cf. https://gymnasium.farama.org/api/spaces/composite/#dict). "
            "which is supported by SB3."
        )
        # Check for Sequence spaces inside Tuple
        for space in observation_space.spaces:
            if isinstance(space, spaces.Sequence):
                sequence_space = True
            elif isinstance(space, spaces.Graph):
                graph_space = True

    # Check for Sequence spaces inside OneOf
    if _is_oneof_space(observation_space):
        warnings.warn(
            "OneOf observation space is not supported by Stable-Baselines3. "
            "Note: The checks for returned values are skipped."
        )
        should_skip = True

    _check_non_zero_start(observation_space, "observation")

    if isinstance(observation_space, spaces.Sequence) or sequence_space:
        warnings.warn(
            "Sequence observation space is not supported by Stable-Baselines3. "
            "You can pad your observation to have a fixed size instead.\n"
            "Note: The checks for returned values are skipped."
        )
        should_skip = True

    if isinstance(observation_space, spaces.Graph) or graph_space:
        warnings.warn(
            "Graph observation space is not supported by Stable-Baselines3. "
            "Note: The checks for returned values are skipped."
        )
        should_skip = True

    _check_non_zero_start(action_space, "action")

    if not _is_numpy_array_space(action_space):
        warnings.warn(
            "The action space is not based off a numpy array. Typically this means it's either a Dict or Tuple space. "
            "This type of action space is currently not supported by Stable Baselines 3. You should try to flatten the "
            "action using a wrapper."
        )
    return should_skip


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
    assert isinstance(vec_env, VecEnv), (
        "Your environment must inherit from stable_baselines3.common.vec_env.VecEnv"
    )

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