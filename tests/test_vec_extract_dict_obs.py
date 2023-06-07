import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecExtractDictObs, VecMonitor


class DictObsVecEnv(VecEnv):
    """Custom Environment that produces observation in a dictionary like the procgen env"""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.num_envs = 4
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({"rgb": spaces.Box(low=0.0, high=255.0, shape=(86, 86), dtype=np.float32)})
        self.n_steps = 0
        self.max_steps = 5
        self.render_mode = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.n_steps += 1
        done = self.n_steps >= self.max_steps
        if done:
            infos = [
                {"terminal_observation": {"rgb": np.zeros((86, 86), dtype=np.float32)}, "TimeLimit.truncated": True}
                for _ in range(self.num_envs)
            ]
        else:
            infos = []
        return (
            {"rgb": np.zeros((self.num_envs, 86, 86), dtype=np.float32)},
            np.zeros((self.num_envs,), dtype=np.float32),
            np.ones((self.num_envs,), dtype=bool) * done,
            infos,
        )

    def reset(self):
        self.n_steps = 0
        return {"rgb": np.zeros((self.num_envs, 86, 86), dtype=np.float32)}

    def render(self, mode=""):
        pass

    def get_attr(self, attr_name, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        return [getattr(self, attr_name) for _ in indices]

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        return [False for _ in indices]

    def env_method(self):
        raise NotImplementedError  # not used in the test

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError  # not used in the test


def test_extract_dict_obs():
    """Test VecExtractDictObs"""

    env = DictObsVecEnv()
    env = VecExtractDictObs(env, "rgb")
    assert env.reset().shape == (4, 86, 86)

    for _ in range(10):
        obs, _, dones, infos = env.step([env.action_space.sample() for _ in range(env.num_envs)])
        assert obs.shape == (4, 86, 86)
        for idx, info in enumerate(infos):
            if "terminal_observation" in info:
                assert dones[idx]
                assert info["terminal_observation"].shape == (86, 86)
            else:
                assert not dones[idx]


def test_vec_with_ppo():
    """
    Test the `VecExtractDictObs` with PPO
    """
    env = DictObsVecEnv()
    env = VecExtractDictObs(env, "rgb")
    monitor_env = VecMonitor(env)
    model = PPO("MlpPolicy", monitor_env, verbose=1, n_steps=64, device="cpu")
    model.learn(total_timesteps=250)
