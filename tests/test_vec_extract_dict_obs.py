import numpy as np
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor


class DictObsVecEnv:
    """Custom Environment that produces observation in a dictionary like the procgen env"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.num_envs = 4
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({"rgb": spaces.Box(low=0.0, high=255.0, shape=(86, 86), dtype=np.float32)})

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        return (
            {"rgb": np.zeros((self.num_envs, 86, 86))},
            np.zeros((self.num_envs,)),
            np.zeros((self.num_envs,), dtype=bool),
            [{} for _ in range(self.num_envs)],
        )

    def reset(self):
        return {"rgb": np.zeros((self.num_envs, 86, 86))}

    def render(self, mode="human", close=False):
        pass


def test_extract_dict_obs():
    """Test VecExtractDictObs"""

    env = DictObsVecEnv()
    env = VecExtractDictObs(env, "rgb")
    assert env.reset().shape == (4, 86, 86)


def test_vec_with_ppo():
    """
    Test the `VecExtractDictObs` with PPO
    """
    env = DictObsVecEnv()
    env = VecExtractDictObs(env, "rgb")
    monitor_env = VecMonitor(env)
    model = PPO("MlpPolicy", monitor_env, verbose=1, n_steps=64, device="cpu")
    model.learn(total_timesteps=250)
