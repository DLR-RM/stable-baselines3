import warnings

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecCheckNan(VecEnvWrapper):
    """
    NaN and inf checking wrapper for vectorized environment, will raise a warning by default,
    allowing you to know from what the NaN of inf originated from.

    :param venv: the vectorized environment to wrap
    :param raise_exception: Whether to raise a ValueError, instead of a UserWarning
    :param warn_once: Whether to only warn once.
    :param check_inf: Whether to check for +inf or -inf as well
    """

    def __init__(self, venv: VecEnv, raise_exception: bool = False, warn_once: bool = True, check_inf: bool = True) -> None:
        super().__init__(venv)
        self.raise_exception = raise_exception
        self.warn_once = warn_once
        self.check_inf = check_inf

        self._user_warned = False

        self._actions: np.ndarray
        self._observations: VecEnvObs

    def step_async(self, actions: np.ndarray) -> None:
        self._check_val(event="step_async", actions=actions)
        self._actions = actions
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        self._check_val(event="step_wait", observations=observations, rewards=rewards, dones=dones)
        self._observations = observations
        return observations, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        observations = self.venv.reset()
        self._check_val(event="reset", observations=observations)
        self._observations = observations
        return observations

    def _check_val(self, event: str, **kwargs) -> None:
        # if warn and warn once and have warned once: then stop checking
        if not self.raise_exception and self.warn_once and self._user_warned:
            return

        found = []
        for name, val in kwargs.items():
            has_nan = np.any(np.isnan(val))
            has_inf = self.check_inf and np.any(np.isinf(val))
            if has_inf:
                found.append((name, "inf"))
            if has_nan:
                found.append((name, "nan"))

        if found:
            self._user_warned = True
            msg = ""
            for i, (name, type_val) in enumerate(found):
                msg += f"found {type_val} in {name}"
                if i != len(found) - 1:
                    msg += ", "

            msg += ".\r\nOriginated from the "

            if event == "reset":
                msg += "environment observation (at reset)"
            elif event == "step_wait":
                msg += f"environment, Last given value was: \r\n\taction={self._actions}"
            elif event == "step_async":
                msg += f"RL model, Last given value was: \r\n\tobservations={self._observations}"
            else:
                raise ValueError("Internal error.")

            if self.raise_exception:
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning)
