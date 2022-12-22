import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first


class StackedObservations:
    """
    Frame stacking wrapper for data.

    Dimension to stack over is either first (channels-first) or
    last (channels-last), which is detected automatically using
    ``common.preprocessing.is_image_space_channels_first`` if
    observation is an image space.

    :param num_envs: number of environments
    :param n_stack: Number of frames to stack
    :param observation_space: Environment observation space.
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
    """

    def __init__(
        self,
        num_envs: int,
        n_stack: int,
        observation_space: spaces.Space,
        channels_order: Optional[str] = None,
    ):

        self.n_stack = n_stack
        (
            self.channels_first,
            self.stack_dimension,
            self.stackedobs,
            self.repeat_axis,
        ) = self.compute_stacking(num_envs, n_stack, observation_space, channels_order)
        super().__init__()

    @staticmethod
    def compute_stacking(
        num_envs: int,
        n_stack: int,
        observation_space: spaces.Box,
        channels_order: Optional[str] = None,
    ) -> Tuple[bool, int, np.ndarray, int]:
        """
        Calculates the parameters in order to stack observations

        :param num_envs: Number of environments in the stack
        :param n_stack: The number of observations to stack
        :param observation_space: The observation space
        :param channels_order: The order of the channels
        :return: tuple of channels_first, stack_dimension, stackedobs, repeat_axis
        """
        channels_first = False
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
        low = np.repeat(observation_space.low, n_stack, axis=repeat_axis)
        stackedobs = np.zeros((num_envs,) + low.shape, low.dtype)
        return channels_first, stack_dimension, stackedobs, repeat_axis

    def stack_observation_space(self, observation_space: spaces.Box) -> spaces.Box:
        """
        Given an observation space, returns a new observation space with stacked observations

        :return: New observation space with stacked dimensions
        """
        low = np.repeat(observation_space.low, self.n_stack, axis=self.repeat_axis)
        high = np.repeat(observation_space.high, self.n_stack, axis=self.repeat_axis)
        return spaces.Box(low=low, high=high, dtype=observation_space.dtype)

    def reset(self, observation: np.ndarray) -> np.ndarray:
        """
        Resets the stackedobs, adds the reset observation to the stack, and returns the stack

        :param observation: Reset observation
        :return: The stacked reset observation
        """
        self.stackedobs[...] = 0
        if self.channels_first:
            self.stackedobs[:, -observation.shape[self.stack_dimension] :, ...] = observation
        else:
            self.stackedobs[..., -observation.shape[self.stack_dimension] :] = observation
        return self.stackedobs

    def update(
        self,
        observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Adds the observations to the stack and uses the dones to update the infos.

        :param observations: numpy array of observations
        :param dones: numpy array of done info
        :param infos: numpy array of info dicts
        :return: tuple of the stacked observations and the updated infos
        """
        stack_ax_size = observations.shape[self.stack_dimension]
        self.stackedobs = np.roll(self.stackedobs, shift=-stack_ax_size, axis=self.stack_dimension)
        for i, done in enumerate(dones):
            if done:
                if "terminal_observation" in infos[i]:
                    old_terminal = infos[i]["terminal_observation"]
                    if self.channels_first:
                        new_terminal = np.concatenate(
                            (self.stackedobs[i, :-stack_ax_size, ...], old_terminal),
                            axis=0,  # self.stack_dimension - 1, as there is not batch dim
                        )
                    else:
                        new_terminal = np.concatenate(
                            (self.stackedobs[i, ..., :-stack_ax_size], old_terminal),
                            axis=self.stack_dimension,
                        )
                    infos[i]["terminal_observation"] = new_terminal
                else:
                    warnings.warn("VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stackedobs[i] = 0
        if self.channels_first:
            self.stackedobs[:, -observations.shape[self.stack_dimension] :, ...] = observations
        else:
            self.stackedobs[..., -observations.shape[self.stack_dimension] :] = observations
        return self.stackedobs, infos


class StackedDictObservations(StackedObservations):
    """
    Frame stacking wrapper for dictionary data.

    Dimension to stack over is either first (channels-first) or
    last (channels-last), which is detected automatically using
    ``common.preprocessing.is_image_space_channels_first`` if
    observation is an image space.

    :param num_envs: number of environments
    :param n_stack: Number of frames to stack
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
    """

    def __init__(
        self,
        num_envs: int,
        n_stack: int,
        observation_space: spaces.Dict,
        channels_order: Optional[Union[str, Dict[str, str]]] = None,
    ):
        self.n_stack = n_stack
        self.channels_first = {}
        self.stack_dimension = {}
        self.stackedobs = {}
        self.repeat_axis = {}

        for key, subspace in observation_space.spaces.items():
            assert isinstance(subspace, spaces.Box), "StackedDictObservations only works with nested gym.spaces.Box"
            if isinstance(channels_order, str) or channels_order is None:
                subspace_channel_order = channels_order
            else:
                subspace_channel_order = channels_order[key]
            (
                self.channels_first[key],
                self.stack_dimension[key],
                self.stackedobs[key],
                self.repeat_axis[key],
            ) = self.compute_stacking(num_envs, n_stack, subspace, subspace_channel_order)

    def stack_observation_space(self, observation_space: spaces.Dict) -> spaces.Dict:
        """
        Returns the stacked version of a Dict observation space

        :param observation_space: Dict observation space to stack
        :return: stacked observation space
        """
        spaces_dict = {}
        for key, subspace in observation_space.spaces.items():
            low = np.repeat(subspace.low, self.n_stack, axis=self.repeat_axis[key])
            high = np.repeat(subspace.high, self.n_stack, axis=self.repeat_axis[key])
            spaces_dict[key] = spaces.Box(low=low, high=high, dtype=subspace.dtype)
        return spaces.Dict(spaces=spaces_dict)

    def reset(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # pytype: disable=signature-mismatch
        """
        Resets the stacked observations, adds the reset observation to the stack, and returns the stack

        :param observation: Reset observation
        :return: Stacked reset observations
        """
        for key, obs in observation.items():
            self.stackedobs[key][...] = 0
            if self.channels_first[key]:
                self.stackedobs[key][:, -obs.shape[self.stack_dimension[key]] :, ...] = obs
            else:
                self.stackedobs[key][..., -obs.shape[self.stack_dimension[key]] :] = obs
        return self.stackedobs

    def update(
        self,
        observations: Dict[str, np.ndarray],
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:  # pytype: disable=signature-mismatch
        """
        Adds the observations to the stack and uses the dones to update the infos.

        :param observations: Dict of numpy arrays of observations
        :param dones: numpy array of dones
        :param infos: dict of infos
        :return: tuple of the stacked observations and the updated infos
        """
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
                        warnings.warn("VecFrameStack wrapping a VecEnv without terminal_observation info")
                    self.stackedobs[key][i] = 0
            if self.channels_first[key]:
                self.stackedobs[key][:, -stack_ax_size:, ...] = observations[key]
            else:
                self.stackedobs[key][..., -stack_ax_size:] = observations[key]
        return self.stackedobs, infos
