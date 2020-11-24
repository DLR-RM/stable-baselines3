import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.preprocessing import (
    is_image_space,
    has_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment. Designed for image observations.

    Dimension to stack over is either first (channels-first) or
    last (channels-last), which is detected automatically using
    ``common.preprocessing.is_image_space_channels_first`` if
    observation is an image space.

    :param venv: the vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
    """

    def __init__(
        self, venv: VecEnv, n_stack: int, channels_order: Optional[str] = None
    ):
        self.venv = venv
        self.n_stack = n_stack

        wrapped_obs_space = venv.observation_space

        if isinstance(wrapped_obs_space, spaces.Box):
            (
                self.channels_first,
                self.stack_dimension,
                self.stackedobs,
                observation_space,
            ) = self.compute_stacking(channels_order, wrapped_obs_space)

        elif isinstance(wrapped_obs_space, spaces.Dict):
            self.channels_first = {}
            self.stack_dimension = {}
            self.stackedobs = {}
            space_dict = {}
            for (key, subspace) in wrapped_obs_space.spaces.items():
                assert isinstance(
                    subspace, spaces.Box
                ), "VecFrameStack with gym.spaces.Dict only works with nested gym.spaces.Box"
                (
                    self.channels_first[key],
                    self.stack_dimension[key],
                    self.stackedobs[key],
                    space_dict[key],
                ) = self.compute_stacking(channels_order, subspace)
            observation_space = spaces.Dict(spaces=space_dict)
        else:
            raise Exception(
                "VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces"
            )

        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def compute_stacking(self, channels_order, obs_space):
        channels_first = False
        if channels_order is None:
            # Detect channel location automatically for images
            if is_image_space(obs_space):
                channels_first = is_image_space_channels_first(obs_space)
            else:
                # Default behavior for non-image space, stack on the last axis
                channels_first = False
        else:
            assert channels_order in {
                "last",
                "first",
            }, "`channels_order` must be one of following: 'last', 'first'"

            channels_first = channels_order == "first"

        # This includes the vec-env dimension (first)
        stack_dimension = 1 if channels_first else -1
        repeat_axis = 0 if channels_first else -1
        low = np.repeat(obs_space.low, self.n_stack, axis=repeat_axis)
        high = np.repeat(obs_space.high, self.n_stack, axis=repeat_axis)
        stackedobs = np.zeros((self.venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)
        return channels_first, stack_dimension, stackedobs, observation_space

    def step_wait(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:

        observations, rewards, dones, infos = self.venv.step_wait()

        if isinstance(self.venv.observation_space, spaces.Box):
            stack_ax_size = observations.shape[self.stack_dimension]
            self.stackedobs = np.roll(
                self.stackedobs, shift=-stack_ax_size, axis=self.stack_dimension
            )
            for i, done in enumerate(dones):
                if done:
                    if "terminal_observation" in infos[i]:
                        old_terminal = infos[i]["terminal_observation"]
                        if self.channels_first:
                            new_terminal = np.concatenate(
                                (
                                    self.stackedobs[i, :-stack_ax_size, ...],
                                    old_terminal,
                                ),
                                axis=self.stack_dimension,
                            )
                        else:
                            new_terminal = np.concatenate(
                                (
                                    self.stackedobs[i, ..., :-stack_ax_size],
                                    old_terminal,
                                ),
                                axis=self.stack_dimension,
                            )
                        infos[i]["terminal_observation"] = new_terminal
                    else:
                        warnings.warn(
                            "VecFrameStack wrapping a VecEnv without terminal_observation info"
                        )
                    self.stackedobs[i] = 0
            if self.channels_first:
                self.stackedobs[
                    :, -observations.shape[self.stack_dimension] :, ...
                ] = observations
            else:
                self.stackedobs[
                    ..., -observations.shape[self.stack_dimension] :
                ] = observations
        elif isinstance(self.venv.observation_space, spaces.Dict):
            for key in self.stackedobs.keys():
                stack_ax_size = observations[key].shape[self.stack_dimension[key]]
                self.stackedobs[key] = np.roll(
                    self.stackedobs[key],
                    shift=-stack_ax_size,
                    axis=self.stack_dimension[key],
                )

                for i, done in enumerate(dones):
                    if done:
                        if "terminal_observation" in infos[i]:
                            old_terminal = infos[i]["terminal_observation"][key]
                            if self.channels_first[key]:
                                # new_terminal = np.concatenate(
                                #     (self.stackedobs[key][i, :-stack_ax_size, ...], old_terminal), axis=self.stack_dimension[key]
                                # )
                                # ValueError: all the input array dimensions for the concatenation axis must match exactly,
                                # but along dimension 0, the array at index 0 has size 6 and the array at index 1 has size 2
                                new_terminal = np.vstack(
                                    (
                                        self.stackedobs[key][i, :-stack_ax_size, ...],
                                        old_terminal,
                                    )
                                )
                            else:
                                new_terminal = np.concatenate(
                                    (
                                        self.stackedobs[key][i, ..., :-stack_ax_size],
                                        old_terminal,
                                    ),
                                    axis=self.stack_dimension[key],
                                )
                            infos[i]["terminal_observation"][key] = new_terminal
                        else:
                            warnings.warn(
                                "VecFrameStack wrapping a VecEnv without terminal_observation info"
                            )
                        self.stackedobs[key][i] = 0
                if self.channels_first:
                    self.stackedobs[key][
                        :, -observations[key].shape[self.stack_dimension[key]] :, ...
                    ] = observations[key]
                else:
                    self.stackedobs[key][
                        ..., -observations[key].shape[self.stack_dimension] :
                    ] = observations[key]
        else:
            raise Exception(
                f"Unhandled observation type {type(self.venv.observation_space)}"
            )

        return self.stackedobs, rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        """
        observation = self.venv.reset()  # pytype:disable=annotation-type-mismatch

        if isinstance(self.venv.observation_space, spaces.Box):
            self.stackedobs[...] = 0
            if self.channels_first:
                self.stackedobs[
                    :, -observation.shape[self.stack_dimension] :, ...
                ] = observation
            else:
                self.stackedobs[
                    ..., -observation.shape[self.stack_dimension] :
                ] = observation

        elif isinstance(self.venv.observation_space, spaces.Dict):
            for key, obs in observation.items():
                self.stackedobs[key][...] = 0
                if self.channels_first[key]:
                    self.stackedobs[key][
                        :, -obs.shape[self.stack_dimension[key]] :, ...
                    ] = obs
                else:
                    self.stackedobs[key][
                        ..., -obs.shape[self.stack_dimension[key]] :
                    ] = obs

        return self.stackedobs

    def close(self) -> None:
        self.venv.close()
