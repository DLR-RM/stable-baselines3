import os

import gym
import numpy as np
import torch as th

from stable_baselines3 import SAC


def test_load_compiled():
    env = gym.make("Pendulum-v1")

    model = SAC("MlpPolicy", env, verbose=1)
    model.policy = th.compile(model.policy)  # Compile the model
    model.save("sac_pendulum")

    del model  # remove to demonstrate saving and loading

    bugged = False
    try:
        model = SAC.load("sac_pendulum")
    except Exception:
        bugged = True
    finally:
        os.remove("sac_pendulum.zip")

    assert not bugged, "Bugged sadly"


if __name__ == "__main__":
    test_load_compiled()
