from typing import Dict, Union

import gym
import numpy as np

from stable_baselines3.common.type_aliases import GymStepReturn


class SimpleMultiObsEnv(gym.Env):
    """
    Base class for GridWorld-based MultiObs Environments 4x4  grid world.

    .. code-block:: text

        ____________
       | 0  1  2   3|
       | 4|¯5¯¯6¯| 7|
       | 8|_9_10_|11|
       |12 13  14 15|
       ¯¯¯¯¯¯¯¯¯¯¯¯¯¯

    start is 0
    states 5, 6, 9, and 10 are blocked
    goal is 15
    actions are = [left, down, right, up]

    simple linear state env of 15 states but encoded with a vector and an image observation:
    each column is represented by a random vector and each row is
    represented by a random image, both sampled once at creation time.

    :param num_col: Number of columns in the grid
    :param num_row: Number of rows in the grid
    :param random_start: If true, agent starts in random position
    :param channel_last: If true, the image will be channel last, else it will be channel first
    """

    def __init__(
        self,
        num_col: int = 4,
        num_row: int = 4,
        random_start: bool = True,
        discrete_actions: bool = True,
        channel_last: bool = True,
    ):
        super(SimpleMultiObsEnv, self).__init__()

        self.vector_size = 5
        if channel_last:
            self.img_size = [64, 64, 1]
        else:
            self.img_size = [1, 64, 64]

        self.random_start = random_start
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.action_space = gym.spaces.Discrete(4)
        else:
            self.action_space = gym.spaces.Box(0, 1, (4,))

        self.observation_space = gym.spaces.Dict(
            spaces={
                "vec": gym.spaces.Box(0, 1, (self.vector_size,)),
                "img": gym.spaces.Box(0, 255, self.img_size, dtype=np.uint8),
            }
        )
        self.count = 0
        # Timeout
        self.max_count = 100
        self.log = ""
        self.state = 0
        self.action2str = ["left", "down", "right", "up"]
        self.init_possible_transitions()

        self.num_col = num_col
        self.state_mapping = []
        self.init_state_mapping(num_col, num_row)

        self.max_state = len(self.state_mapping) - 1

    def init_state_mapping(self, num_col: int, num_row: int) -> None:
        """
        Initializes the state_mapping array which holds the observation values for each state

        :param num_col: Number of columns.
        :param num_row: Number of rows.
        """
        # Each column is represented by a random vector
        col_vecs = np.random.random((num_col, self.vector_size))
        # Each row is represented by a random image
        row_imgs = np.random.randint(0, 255, (num_row, 64, 64), dtype=np.int32)

        for i in range(num_col):
            for j in range(num_row):
                self.state_mapping.append({"vec": col_vecs[i], "img": row_imgs[j].reshape(self.img_size)})

    def get_state_mapping(self) -> Dict[str, np.ndarray]:
        """
        Uses the state to get the observation mapping.

        :return: observation dict {'vec': ..., 'img': ...}
        """
        return self.state_mapping[self.state]

    def init_possible_transitions(self) -> None:
        """
        Initializes the transitions of the environment
        The environment exploits the cardinal directions of the grid by noting that
        they correspond to simple addition and subtraction from the cell id within the grid

        - up => means moving up a row => means subtracting the length of a column
        - down => means moving down a row => means adding the length of a column
        - left => means moving left by one => means subtracting 1
        - right => means moving right by one => means adding 1

        Thus one only needs to specify in which states each action is possible
        in order to define the transitions of the environment
        """
        self.left_possible = [1, 2, 3, 13, 14, 15]
        self.down_possible = [0, 4, 8, 3, 7, 11]
        self.right_possible = [0, 1, 2, 12, 13, 14]
        self.up_possible = [4, 8, 12, 7, 11, 15]

    def step(self, action: Union[int, float, np.ndarray]) -> GymStepReturn:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action:
        :return: tuple (observation, reward, done, info).
        """
        if not self.discrete_actions:
            action = np.argmax(action)
        else:
            action = int(action)

        self.count += 1

        prev_state = self.state

        reward = -0.1
        # define state transition
        if self.state in self.left_possible and action == 0:  # left
            self.state -= 1
        elif self.state in self.down_possible and action == 1:  # down
            self.state += self.num_col
        elif self.state in self.right_possible and action == 2:  # right
            self.state += 1
        elif self.state in self.up_possible and action == 3:  # up
            self.state -= self.num_col

        got_to_end = self.state == self.max_state
        reward = 1 if got_to_end else reward
        done = self.count > self.max_count or got_to_end

        self.log = f"Went {self.action2str[action]} in state {prev_state}, got to state {self.state}"

        return self.get_state_mapping(), reward, done, {"got_to_end": got_to_end}

    def render(self, mode: str = "human") -> None:
        """
        Prints the log of the environment.

        :param mode:
        """
        print(self.log)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the environment state and step count and returns reset observation.

        :return: observation dict {'vec': ..., 'img': ...}
        """
        self.count = 0
        if not self.random_start:
            self.state = 0
        else:
            self.state = np.random.randint(0, self.max_state)
        return self.state_mapping[self.state]
