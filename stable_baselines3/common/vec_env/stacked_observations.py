import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first


class StackedObservations:
    """
    Frame stacking wrapper for data.

    Dimension to stack over is either first (channels-first) or last (channels-last), which is detected automatically using
    ``common.preprocessing.is_image_space_channels_first`` if observation is an image space.

    :param num_envs: Number of environments
    :param n_stack: Number of frames to stack
    :param observation_space: Environment observation space
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last".
        For Dict space, channels_order can also be a dictionary.
    """

    def __init__(
        self,
        num_envs: int,
        n_stack: int,
        observation_space: Union[spaces.Box, spaces.Dict],
        channels_order: Optional[Union[str, Dict[str, str]]] = None,
    ):
        self.n_stack = n_stack
        if isinstance(observation_space, spaces.Dict):
            if not isinstance(channels_order, dict):
                channels_order = {key: channels_order for key in observation_space.spaces.keys()}
            self.sub_stacked_observations = {
                key: StackedObservations(num_envs, n_stack, subspace, channels_order[key])
                for key, subspace in observation_space.spaces.items()
            }
        elif isinstance(observation_space, spaces.Box):
            assert channels_order is None or isinstance(channels_order, str), "TODO: write error message"
            self.channels_first, self.stack_dimension, self.stacked_shape, self.repeat_axis = self.compute_stacking(
                n_stack, observation_space, channels_order
            )
            low = np.repeat(observation_space.low, self.n_stack, axis=self.repeat_axis)
            high = np.repeat(observation_space.high, self.n_stack, axis=self.repeat_axis)
            self.stacked_observation_space = spaces.Box(low=low, high=high, dtype=observation_space.dtype)
            self.stacked_obs = np.zeros((num_envs,) + self.stacked_shape, dtype=observation_space.dtype)

    @staticmethod
    def compute_stacking(
        n_stack: int, observation_space: spaces.Box, channels_order: Optional[str] = None
    ) -> Tuple[bool, int, Tuple[int], int]:
        """
        Calculates the parameters in order to stack observations

        :param n_stack: Number of observations to stack
        :param observation_space: Observation space
        :param channels_order: Order of the channels
        :return: Tuple of channels_first, stack_dimension, stackedobs, repeat_axis
        """

        if channels_order is None:
            # Detect channel location automatically for images
            if is_image_space(observation_space):
                channels_first = is_image_space_channels_first(observation_space)
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
        stacked_shape = list(observation_space.shape)
        stacked_shape[repeat_axis] *= n_stack
        return channels_first, stack_dimension, tuple(stacked_shape), repeat_axis

    def reset(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Reset the stacked_obs, add the reset observation to the stack, and return the stack.

        :param observation: Reset observation
        :return: The stacked reset observation
        """
        if isinstance(observation, dict):
            return {key: self.sub_stacked_observations[key].reset(obs) for key, obs in observation.items()}

        self.stacked_obs[...] = 0
        if self.channels_first:
            self.stacked_obs[:, -observation.shape[self.stack_dimension] :, ...] = observation
        else:
            self.stacked_obs[..., -observation.shape[self.stack_dimension] :] = observation
        return self.stacked_obs

    def update(
        self,
        observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Add the observations to the stack and use the dones to update the infos.

        :param observations: Observations
        :param dones: Dones
        :param infos: Infos
        :return: Tuple of the stacked observations and the updated infos
        """
        if isinstance(observations, dict):
            # From [{}, {terminal_obs: {key1: ..., key2: ...}}]
            # to {key1: [{}, {terminal_obs: ...}], key2: [{}, {terminal_obs: ...}]}
            sub_infos = {
                key: [
                    {"terminal_observation": info["terminal_observation"][key]} if "terminal_observation" in info else {}
                    for info in infos
                ]
                for key in observations.keys()
            }

            stacked_obs = {}
            stacked_infos = {}
            for key, obs in observations.items():
                stacked_obs[key], stacked_infos[key] = self.sub_stacked_observations[key].update(obs, dones, sub_infos[key])

            # From {key1: [{}, {terminal_obs: ...}], key2: [{}, {terminal_obs: ...}]}
            # to [{}, {terminal_obs: {key1: ..., key2: ...}}]
            for key in infos.keys():
                for env_idx in range(len(infos)):
                    if "terminal_observation" in infos[env_idx]:
                        infos[env_idx]["terminal_observation"][key] = stacked_infos[key][env_idx]["terminal_observation"]
            return stacked_obs, infos

        shift = -observations.shape[self.stack_dimension]
        self.stacked_obs = np.roll(self.stacked_obs, shift, axis=self.stack_dimension)
        for env_idx, done in enumerate(dones):
            if done:
                if "terminal_observation" in infos[env_idx]:
                    old_terminal = infos[env_idx]["terminal_observation"]
                    if self.channels_first:
                        # self.stack_dimension - 1, as there is not batch dim
                        axis = 0
                        new_terminal = np.concatenate((self.stacked_obs[env_idx, :shift, ...], old_terminal), axis)
                    else:
                        axis = self.stack_dimension
                        new_terminal = np.concatenate((self.stacked_obs[env_idx, ..., :shift], old_terminal), axis)
                    infos[env_idx]["terminal_observation"] = new_terminal
                else:
                    warnings.warn("VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stacked_obs[env_idx] = 0
        if self.channels_first:
            self.stacked_obs[:, -observations.shape[self.stack_dimension] :, ...] = observations
        else:
            self.stacked_obs[..., -observations.shape[self.stack_dimension] :] = observations
        return self.stacked_obs, infos
