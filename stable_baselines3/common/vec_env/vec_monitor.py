import time

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments, it is used to record the episode reward, length, time and other data.

    Some environments like [`openai/procgen`](https://github.com/openai/procgen) or [`gym3`](https://github.com/openai/gym3) directly
    initialize the vectorized environments, without giving us a chance to use the `Monitor` wrapper. So this class simply does the job
    of the `Monitor` wrapper on a vectorized level.

    As an example, the following two ways of initializing vectorized envs should be equivalent

    ```python
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gym
    def make_env(gym_id):
        def thunk():
            env = gym.make(gym_id, render_mode='rgb_array')
            env = Monitor(env)
            return env
        return thunk
    envs = DummyVecEnv([make_env('procgen-starpilot-v0')])
    ```

    ```python
    from procgen import ProcgenEnv
    from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor
    venv = ProcgenEnv(num_envs=1, env_name='starpilot')
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv)
    ```
    See [here](https://github.com/openai/train-procgen/blob/1a2ae2194a61f76a733a39339530401c024c3ad8/train_procgen/train.py#L36-L43) for a full example.

    :param venv: The vectorized environment
    """

    def __init__(self, venv: VecEnv):
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
