import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


class BoxObsVecEnv:
    """Custom Environment that produces box observations"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, obs_space: spaces.Box):
        self.num_envs = 4
        self.action_space = spaces.Discrete(2)
        self.observation_space = obs_space

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs = np.zeros((self.num_envs,) + self.observation_space.shape)
        terminal_obs = np.zeros(self.observation_space.shape)
        return (
            obs,
            np.zeros((self.num_envs,)),
            np.ones((self.num_envs,), dtype=bool),
            [{"terminal_observation": terminal_obs} for _ in range(self.num_envs)],
        )

    def reset(self):
        return np.zeros((self.num_envs,) + self.observation_space.shape)

    def render(self, mode="human", close=False):
        pass


class DictObsVecEnv:
    """Custom Environment that produces observation in a dictionary like the procgen env"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.num_envs = 4
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(86, 86, 3), dtype=np.uint8),
                "scalar": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs = {
            "rgb": np.zeros((self.num_envs, 86, 86, 3)),
            "scalar": np.zeros((self.num_envs,)),
        }
        terminal_obs = {"rgb": np.zeros((86, 86, 3)), "scalar": np.zeros((1,))}
        return (
            obs,
            np.zeros((self.num_envs,)),
            np.ones((self.num_envs,), dtype=bool),
            [{"terminal_observation": terminal_obs} for _ in range(self.num_envs)],
        )

    def reset(self):
        return {
            "rgb": np.zeros((self.num_envs, 86, 86, 3)),
            "scalar": np.zeros((self.num_envs,)),
        }

    def render(self, mode="human", close=False):
        pass


def test_scalar_frame_stack():
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    env = BoxObsVecEnv(obs_space)
    env = VecFrameStack(env, n_stack=8)
    obs, _, _, _ = env.step(env.action_space.sample())

    assert len(obs) == 4
    assert obs[0].shape == (8,)


def test_dict_frame_stack():
    env = DictObsVecEnv()
    env = VecFrameStack(env, n_stack=8)
    obs, _, _, _ = env.step(env.action_space.sample())

    assert obs["rgb"].shape == (4, 86, 86, 8 * 3)
    assert obs["scalar"].shape == (4, 8)


def test_regular_frame_stack():
    obs_space_0 = spaces.Box(low=0.0, high=1.0, shape=(86, 86, 3), dtype=np.float32)
    obs_space_1 = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

    env_0 = BoxObsVecEnv(obs_space_0)
    env_0 = VecFrameStack(env_0, n_stack=8)
    obs_0, _, _, _ = env_0.step(env_0.action_space.sample())
    assert len(obs_0) == 4
    assert obs_0[0].shape == (86, 86, 3 * 8)

    env_1 = BoxObsVecEnv(obs_space_1)
    env_1 = VecFrameStack(env_1, n_stack=8)
    obs_1, _, _, _ = env_1.step(env_1.action_space.sample())
    assert len(obs_1) == 4
    assert obs_1[0].shape == (8 * 6,)
