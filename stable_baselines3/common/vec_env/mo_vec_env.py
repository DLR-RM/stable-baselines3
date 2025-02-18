import numpy as np

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


class MoDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns, n_objectives=2, *args, **kwargs):
        self.n_objectives = n_objectives
        super().__init__(env_fns, *args, **kwargs)
        self.buf_rews = np.zeros((self.num_envs, n_objectives), dtype=np.float32)


class MoVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, n_objectives=2, *args, **kwargs):
        self.n_objectives = n_objectives
        super().__init__(env_fns, *args, **kwargs)
        self.buf_rews = np.zeros((self.num_envs, n_objectives), dtype=np.float32)
