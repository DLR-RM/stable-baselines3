import os

import pytest
import numpy as np

from torchy_baselines import A2C, CEMRL, PPO, SAC, TD3
from torchy_baselines.common.noise import NormalActionNoise
from torchy_baselines.common.vec_env import DummyVecEnv
from torchy_baselines.common.identity_env import IdentityEnvBox

MODEL_LIST = [
    PPO
]


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_save_load(model_class):
    """
    Test if 'save' and 'load' saves and loads model correctly

    :param model_class: (BaseRLModel) A RL model
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(10)])

    # create model
    model = model_class('MlpPolicy', env, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)

    # test action probability for given (obs, action) pair
    env = model.get_env()
    obs = env.reset()
    observations = np.array([obs for _ in range(10)])
    observations = np.squeeze(observations)

    #actions = np.array([env.action_space.sample() for _ in range(10)])

    # Get dictionary of current parameters
    params = model.get_parameters()

    # Modify all parameters to be random values
    random_params = dict((param_name,np.random.random(size=param.shape)) for param_name, param in params.items())
    # Update model parameters with the new zeroed values
    model.load_parameters(random_params)
    # Get new action probas
    #...

    # Check
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save.zip")
    model = model.load("test_save")
    os.remove("test_save.zip")
