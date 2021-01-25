import time

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param venv: The vectorized environment
    """

    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, "f")
        self.eplens = np.zeros(self.num_envs, "i")
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {"r": ret, "l": eplen, "t": round(time.time() - self.tstart, 6)}
                info["episode"] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos
