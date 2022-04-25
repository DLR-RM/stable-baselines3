from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import numpy as np
from gym import GoalEnv, spaces
from gym.envs.registration import EnvSpec

from stable_baselines3.common.type_aliases import GymStepReturn


class BitFlippingEnv(GoalEnv):
    """
    Simple bit flipping env, useful to test HER.
    The goal is to flip all the bits to get a vector of ones.
    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.

    :param n_bits: Number of bits to flip
    :param continuous: Whether to use the continuous actions version or not,
        by default, it uses the discrete one
    :param max_steps: Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: Whether to use the discrete observation
        version or not, by default, it uses the ``MultiBinary`` one
    :param image_obs_space: Use image as input instead of the ``MultiBinary`` one.
    :param channel_first: Whether to use channel-first or last image.
    """

    spec = EnvSpec("BitFlippingEnv-v0")

    def __init__(
        self,
        n_bits: int = 10,
        continuous: bool = False,
        max_steps: Optional[int] = None,
        discrete_obs_space: bool = False,
        image_obs_space: bool = False,
        channel_first: bool = True,
    ):
        super().__init__()
        # Shape of the observation when using image space
        self.image_shape = (1, 36, 36) if channel_first else (36, 36, 1)
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        if discrete_obs_space:
            # In the discrete case, the agent act on the binary
            # representation of the observation
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Discrete(2**n_bits),
                    "achieved_goal": spaces.Discrete(2**n_bits),
                    "desired_goal": spaces.Discrete(2**n_bits),
                }
            )
        elif image_obs_space:
            # When using image as input,
            # one image contains the bits 0 -> 0, 1 -> 255
            # and the rest is filled with zeros
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                    "achieved_goal": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                    "desired_goal": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.MultiBinary(n_bits),
                    "achieved_goal": spaces.MultiBinary(n_bits),
                    "desired_goal": spaces.MultiBinary(n_bits),
                }
            )

        self.obs_space = spaces.MultiBinary(n_bits)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)
        self.continuous = continuous
        self.discrete_obs_space = discrete_obs_space
        self.image_obs_space = image_obs_space
        self.state = None
        self.desired_goal = np.ones((n_bits,))
        if max_steps is None:
            max_steps = n_bits
        self.max_steps = max_steps
        self.current_step = 0

    def seed(self, seed: int) -> None:
        self.obs_space.seed(seed)

    def convert_if_needed(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        Convert to discrete space if needed.

        :param state:
        :return:
        """
        if self.discrete_obs_space:
            # The internal state is the binary representation of the
            # observed one
            return int(sum(state[i] * 2**i for i in range(len(state))))

        if self.image_obs_space:
            size = np.prod(self.image_shape)
            image = np.concatenate((state * 255, np.zeros(size - len(state), dtype=np.uint8)))
            return image.reshape(self.image_shape).astype(np.uint8)
        return state

    def convert_to_bit_vector(self, state: Union[int, np.ndarray], batch_size: int) -> np.ndarray:
        """
        Convert to bit vector if needed.

        :param state:
        :param batch_size:
        :return:
        """
        # Convert back to bit vector
        if isinstance(state, int):
            state = np.array(state).reshape(batch_size, -1)
            # Convert to binary representation
            state = ((state[:, :] & (1 << np.arange(len(self.state)))) > 0).astype(int)
        elif self.image_obs_space:
            state = state.reshape(batch_size, -1)[:, : len(self.state)] / 255
        else:
            state = np.array(state).reshape(batch_size, -1)

        return state

    def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
        """
        Helper to create the observation.

        :return: The current observation.
        """
        return OrderedDict(
            [
                ("observation", self.convert_if_needed(self.state.copy())),
                ("achieved_goal", self.convert_if_needed(self.state.copy())),
                ("desired_goal", self.convert_if_needed(self.desired_goal.copy())),
            ]
        )

    def reset(self) -> Dict[str, Union[int, np.ndarray]]:
        self.current_step = 0
        self.state = self.obs_space.sample()
        return self._get_obs()

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None))
        done = reward == 0
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {"is_success": done}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    def compute_reward(
        self, achieved_goal: Union[int, np.ndarray], desired_goal: Union[int, np.ndarray], _info: Optional[Dict[str, Any]]
    ) -> np.float32:
        # As we are using a vectorized version, we need to keep track of the `batch_size`
        if isinstance(achieved_goal, int):
            batch_size = 1
        elif self.image_obs_space:
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 3 else 1
        else:
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1

        desired_goal = self.convert_to_bit_vector(desired_goal, batch_size)
        achieved_goal = self.convert_to_bit_vector(achieved_goal, batch_size)

        # Deceptive reward: it is positive only when the goal is achieved
        # Here we are using a vectorized version
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0).astype(np.float32)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self.state.copy()
        print(self.state)

    def close(self) -> None:
        pass
