from typing import Dict, Tuple, Union

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

    simple linear state env of 15 states but encoded with a vector and an image observation

    :param num_col: Number of columns in the grid
    :param num_row: Number of rows in the grid
    :param random_start: If true, agent starts in random position
    :param noise: Noise added to the observations
    """

    def __init__(
        self,
        num_col: int = 4,
        num_row: int = 4,
        random_start: bool = True,
        noise: float = 0.0,
        discrete_actions: bool = True,
    ):
        super(SimpleMultiObsEnv, self).__init__()

        self.vector_size = 5
        self.img_size = [1, 20, 20]

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
        self.max_count = 100
        self.log = ""
        self.state = 0
        self.action2str = ["left", "down", "right", "up"]
        self.noise = noise
        self.init_possible_transitions()

        self.init_state_mapping(num_col, num_row)

        self.max_state = len(self.state_mapping) - 1

    def random_upsample_img(
        self,
        value_range: Tuple[int, int] = (0, 255),
        initial_size: Tuple[int, int] = (4, 4),
        up_size: Tuple[int, int] = (20, 20),
    ) -> np.ndarray:
        """
        Generated a random image and upsample it.

        :param value_range: The range of values for the img
        :param initial_size: The initial size of the image to generate
        :param up_size: The size of the upsample
        :return: upsampled img
        """
        im = np.random.randint(value_range[0], value_range[1], initial_size, dtype=np.int32)
        return np.array(
            [
                [
                    [
                        im[int(initial_size[0] * row / up_size[0])][int(initial_size[1] * col / up_size[1])]
                        for col in range(up_size[0])
                    ]
                    for row in range(up_size[1])
                ]
            ]
        ).astype(np.int32)

    def init_state_mapping(self, num_col: int, num_row: int) -> None:
        """
        Initializes the state_mapping array which holds the observation values for each state

        :param num_col: Number of columns.
        :param num_row: Number of rows.
        """
        self.num_col = num_col
        self.state_mapping = []

        col_vecs = [np.random.random(self.vector_size) for i in range(num_col)]
        row_imgs = [self.random_upsample_img() for j in range(num_row)]
        for i in range(num_col):
            for j in range(num_row):
                self.state_mapping.append({"vec": col_vecs[i], "img": row_imgs[j]})

    def get_state_mapping(self) -> Dict[str, np.ndarray]:
        """
        Uses the state to get the observation mapping and applies noise if there is any.

        :return: observation dict {'vec': ..., 'img': ...}
        """
        state_dict = self.state_mapping[self.state]
        if self.noise > 0:
            state_dict["vec"] += np.random.random(self.vector_size) * self.noise
            img_noise = int(255 * self.noise)
            state_dict["img"] += np.random.randint(-img_noise, img_noise, tuple(self.img_size), dtype=np.int32)
            state_dict["img"] = np.clip(state_dict["img"], 0, 255)
        return state_dict

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


class NineRoomMultiObsEnv(SimpleMultiObsEnv):
    """
    Extension of the SimpleMultiObsEnv to a 9 room  grid world

    .. code-block:: text

        ____________________________________
       | 0  1  2  |  3   4   5 | 6   7   8  |
       | 9  10 11   12  13  14   15  16  17 |
       | 18 19 20 | 21  22  23 | 24  25  26 |
       |¯¯¯¯   ¯¯¯|¯¯¯¯    ¯¯¯¯|¯¯¯¯    ¯¯¯¯|
       | 27 28 29 | 30  31  32 | 33  34  35 |
       | 36 37 38   39  40  41   42  43  44 |
       | 45 46 47 | 48  49  50 | 51  52  53 |
       |¯¯¯    ¯¯¯|¯¯¯¯    ¯¯¯¯|¯¯¯¯    ¯¯¯¯|
       | 54 55 56 | 57  58  59 | 60  61  62 |
       | 63 64 65   66  67  68   69  70  71 |
       | 72 73 74 | 75  76  77 | 78  79  80 |
       ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯

    :param random_start: If true, agent starts in random position
    :param noise: Noise added to the observations
    """

    def __init__(self, random_start: bool = True, noise: float = 0.0):
        super(NineRoomMultiObsEnv, self).__init__(9, 9, random_start=random_start, noise=noise)

    def init_possible_transitions(self) -> None:
        """
        Initializes the state_mapping array which holds the observation values for each state
        """
        self.left_possible = (
            [1, 2, 4, 5, 7, 8]
            + list(range(10, 18))
            + [19, 20, 22, 23, 25, 26]
            + [28, 29, 31, 32, 34, 35]
            + list(range(37, 45))
            + [46, 47, 49, 50, 52, 53]
            + [55, 56, 58, 59, 61, 62]
            + list(range(64, 72))
            + [73, 74, 76, 77, 79, 80]
        )

        self.down_possible = list(range(18)) + [19, 22, 25] + list(range(27, 45)) + [46, 49, 52] + list(range(54, 72))

        self.right_possible = (
            [0, 1, 3, 4, 6, 7]
            + list(range(9, 17))
            + [18, 19, 21, 22, 24, 25]
            + [27, 28, 30, 31, 33, 34]
            + list(range(36, 44))
            + [45, 46, 48, 49, 51, 52]
            + [54, 55, 57, 58, 60, 61]
            + list(range(63, 71))
            + [72, 73, 75, 76, 78, 79]
        )

        self.up_possible = list(range(9, 27)) + [28, 31, 34] + list(range(36, 54)) + [55, 58, 61] + list(range(63, 81))
