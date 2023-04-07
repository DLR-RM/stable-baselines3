import os

import gym
import torch as th

from stable_baselines3 import SAC


def test_load_compiled():
    env = gym.make("Pendulum-v1")

    model = SAC("MlpPolicy", env, verbose=1)
    model.policy = th.compile(model.policy)  # Compile the model
    model.save("sac_pendulum")

    del model  # remove to demonstrate saving and loading

    try:
        SAC.load("sac_pendulum")
    finally:
        os.remove("sac_pendulum.zip")


if __name__ == "__main__":
    test_load_compiled()
