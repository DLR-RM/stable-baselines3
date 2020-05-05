from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from gym import GoalEnv, spaces

from stable_baselines3.common.type_aliases import GymStepReturn


class BitFlippingEnv(GoalEnv):
    """
    Simple bit flipping env, useful to test HER.
    The goal is to flip all the bits to get a vector of ones.
    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.

    :param n_bits: (int) Number of bits to flip
    :param continuous: (bool) Whether to use the continuous actions version or not,
        by default, it uses the discrete one
    :param max_steps: (Optional[int]) Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: (bool) Whether to use the discrete observation
        version or not, by default, it uses the MultiBinary one
    """
    def __init__(self, n_bits: int = 10,
                 continuous: bool = False,
                 max_steps: Optional[int] = None,
                 discrete_obs_space: bool = False):
        super(BitFlippingEnv, self).__init__()
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        if discrete_obs_space:
            # In the discrete case, the agent act on the binary
            # representation of the observation
            self.observation_space = spaces.Dict({
                'observation': spaces.Discrete(2 ** n_bits - 1),
                'achieved_goal': spaces.Discrete(2 ** n_bits - 1),
                'desired_goal': spaces.Discrete(2 ** n_bits - 1)
            })
        else:
            self.observation_space = spaces.Dict({
                'observation': spaces.MultiBinary(n_bits),
                'achieved_goal': spaces.MultiBinary(n_bits),
                'desired_goal': spaces.MultiBinary(n_bits)
            })

        self.obs_space = spaces.MultiBinary(n_bits)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)
        self.continuous = continuous
        self.discrete_obs_space = discrete_obs_space
        self.state = None
        self.desired_goal = np.ones((n_bits,))
        if max_steps is None:
            max_steps = n_bits
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def convert_if_needed(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        Convert to discrete space if needed.

        :param state: (np.ndarray)
        :return: (np.ndarray or int)
        """
        if self.discrete_obs_space:
            # The internal state is the binary representation of the
            # observed one
            return int(sum([state[i] * 2**i for i in range(len(state))]))
        return state

    def _get_obs(self) -> OrderedDict:
        """
        Helper to create the observation.

        :return: (OrderedDict<int or ndarray>)
        """
        return OrderedDict([
            ('observation', self.convert_if_needed(self.state.copy())),
            ('achieved_goal', self.convert_if_needed(self.state.copy())),
            ('desired_goal', self.convert_if_needed(self.desired_goal.copy()))
        ])

    def reset(self) -> OrderedDict:
        self.current_step = 0
        self.state = self.obs_space.sample()
        return self._get_obs()

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
        done = reward == 0
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {'is_success': done}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       _info) -> float:
        # Deceptive reward: it is positive only when the goal is achieved
        if self.discrete_obs_space:
            return 0.0 if achieved_goal == desired_goal else -1.0
        return 0.0 if (achieved_goal == desired_goal).all() else -1.0

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        if mode == 'rgb_array':
            return self.state.copy()
        print(self.state)

    def close(self) -> None:
        pass
