import os
import pytest
from copy import deepcopy
import numpy as np

import torch as th

from torchy_baselines import A2C, CEMRL, PPO, SAC, TD3
from torchy_baselines.common.vec_env import DummyVecEnv
from torchy_baselines.common.identity_env import IdentityEnvBox

MODEL_LIST = [
    CEMRL,
    PPO,
    A2C,
    TD3,
    SAC,
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
    model.learn(total_timesteps=500, eval_freq=250)

    env.reset()
    observations = np.array([env.step(env.action_space.sample())[0] for _ in range(10)])
    observations = np.squeeze(observations)

    # Get dictionary of current parameters
    params = deepcopy(model.get_policy_parameters())
    opt_params = deepcopy(model.get_opt_parameters())

    # Modify all parameters to be random values
    random_params = dict((param_name, th.rand_like(param)) for param_name, param in params.items())

    # Update model parameters with the new random values
    model.load_parameters(random_params, opt_params)

    new_params = model.get_policy_parameters()
    # Check that all params are different now
    for k in params:
        assert not th.allclose(params[k], new_params[k]), "Parameters did not change as expected."

    params = new_params

    # get selected actions
    selected_actions = [model.predict(observation, deterministic=True) for observation in observations]

    # Check
    model.save("test_save.zip")
    del model
    model = model_class.load("test_save")

    # check if params are still the same after load
    new_params = model.get_policy_parameters()

    # Check that all params are the same as before save load procedure now
    for k in params:
        assert th.allclose(params[k], new_params[k]), "Model parameters not the same after save and load."

    # check if optimizer params are still the same after load
    new_opt_params = model.get_opt_parameters()
    # check if keys are the same
    assert opt_params.keys() == new_opt_params.keys()
    # check if values are the same: only tested for Adam and RMSProp so far
    # comparing states not implemented so far. hashes of state_entries are not the same for equal tensors
    # comparing every sub_entry does not work because of bool value of Tensor with more than one value is ambiguous
    # so far only comparing param_lists
    for optimizer, opt_state in opt_params.items():
        for param_group_idx, param_group in enumerate(opt_state['param_groups']):
            for param_key, param_value in param_group.items():
                if param_key == 'params':  # don't know how to handle params correctly, therefore only check if we have the same amount
                    assert len(param_value) == len(
                        new_opt_params[optimizer]['param_groups'][param_group_idx][param_key])
                else:
                    assert param_value == new_opt_params[optimizer]['param_groups'][param_group_idx][param_key]

    # check if model still selects the same actions
    new_selected_actions = [model.predict(observation, deterministic=True) for observation in observations]
    for i in range(len(selected_actions)):
        assert selected_actions[i] == new_selected_actions[i]

    # check if learn still works
    model.learn(total_timesteps=1000, eval_freq=500)

    # clear file from os
    os.remove("test_save.zip")
