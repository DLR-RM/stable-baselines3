import os
import pytest
from copy import deepcopy

import torch as th

from torchy_baselines import A2C, CEMRL, PPO, SAC, TD3
from torchy_baselines.common.vec_env import DummyVecEnv
from torchy_baselines.common.identity_env import IdentityEnvBox

MODEL_LIST = [
    PPO
]


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_save_load(model_class):
    """
    Test if 'save' and 'load' saves and loads model correctly
    and if 'load_parameters' and 'get_policy_parameters' work correctly
    
    ''warning does not test function of optimizer parameter load

    :param model_class: (BaseRLModel) A RL model
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(10)])

    # create model
    model = model_class('MlpPolicy', env, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)

    # Get dictionary of current parameters
    params = deepcopy(model.get_policy_parameters())
    opt_params = deepcopy(model.get_opt_parameters())

    # Modify all parameters to be random values
    random_params = dict((param_name, th.rand_like(param)) for param_name, param in params.items())

    # Update model parameters with the new random values
    model.load_parameters(random_params, opt_params)

    # Get items that are the same in params and new_params
    new_params = model.get_policy_parameters()
    shared_items = {k: params[k] for k in params if k in new_params and th.all(th.eq(params[k], new_params[k]))}

    # Check that the there are at least some parameters new random parameters
    #for k in params.key():
    #    assert not th.allclose(params[k], new_params[k])
    assert not len(shared_items) == len(new_params), "Selected actions did not change " \
                                                     "after changing model parameters."

    params = new_params

    # Check
    model.learn(total_timesteps=1000, eval_freq=500)
    model.save("test_save.zip")
    del model
    model = model_class.load("test_save")

    #check if params are still the same after load
    new_params = model.get_policy_parameters()
    shared_items = {k: params[k] for k in params if k in new_params and th.all(th.eq(params[k], new_params[k]))}
    # Check that at least some actions are chosen different now
    assert len(shared_items) == len(new_params), "Parameters not the same after save and load."
    os.remove("test_save.zip")
