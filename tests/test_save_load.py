import numpy as np
import os
import pytest
import torch as th
from copy import deepcopy

from torchy_baselines import A2C, CEMRL, PPO, SAC, TD3
from torchy_baselines.common.identity_env import IdentityEnvBox
from torchy_baselines.common.vec_env import DummyVecEnv

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
    observations = observations.reshape(10, -1)

    # Get dictionary of current parameters
    params = deepcopy(model.policy.state_dict())

    # Modify all parameters to be random values
    random_params = dict((param_name, th.rand_like(param)) for param_name, param in params.items())

    # Update model parameters with the new random values
    model.policy.load_state_dict(random_params)

    new_params = model.policy.state_dict()
    # Check that all params are different now
    for k in params:
        assert not th.allclose(params[k], new_params[k]), "Parameters did not change as expected."

    params = new_params

    # get selected actions
    selected_actions = model.predict(observations, deterministic=True)

    # Check
    model.save("test_save.zip")
    del model
    model = model_class.load("test_save", env=env)

    # check if params are still the same after load
    new_params = model.policy.state_dict()

    # Check that all params are the same as before save load procedure now
    for key in params:
        assert th.allclose(params[key], new_params[key]), "Model parameters not the same after save and load."

    # check if model still selects the same actions
    new_selected_actions = model.predict(observations, deterministic=True)
    assert np.allclose(selected_actions, new_selected_actions, 1e-4)

    # check if learn still works
    model.learn(total_timesteps=1000, eval_freq=500)

    # clear file from os
    os.remove("test_save.zip")


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_set_env(model_class):
    """
    Test if set_env function does work correct
    :param model_class: (BaseRLModel) A RL model
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(10)])
    env2 = DummyVecEnv([lambda: IdentityEnvBox(10)])
    env3 = IdentityEnvBox(10)

    # create model
    model = model_class('MlpPolicy', env, policy_kwargs=dict(net_arch=[16]), create_eval_env=True)
    # learn
    model.learn(total_timesteps=1000, eval_freq=500)

    # change env
    model.set_env(env2)
    # learn again
    model.learn(total_timesteps=1000, eval_freq=500)

    # change env test wrapping
    model.set_env(env3)
    # learn again
    model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_exclude_include_saved_params(model_class):
    """
    Test if exclude and include parameters of save() work

    :param model_class: (BaseRLModel) A RL model
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(10)])

    # create model, set verbose as 2, which is not standard
    model = model_class('MlpPolicy', env, policy_kwargs=dict(net_arch=[16]), verbose=2, create_eval_env=True)

    # Check if exclude works
    model.save("test_save.zip", exclude=["verbose"])
    del model
    model = model_class.load("test_save")
    # check if verbose was not saved
    assert model.verbose != 2

    # set verbose as something different then standard settings
    model.verbose = 2
    # Check if include works
    model.save("test_save.zip", exclude=["verbose"], include=["verbose"])
    del model
    model = model_class.load("test_save")
    assert model.verbose == 2

    # clear file from os
    os.remove("test_save.zip")


@pytest.mark.parametrize("model_class", [SAC, TD3])
def test_save_load_replay_buffer(model_class):
    log_folder = 'logs'
    replay_path = os.path.join(log_folder, 'replay_buffer.pkl')
    os.makedirs(log_folder, exist_ok=True)
    model = model_class('MlpPolicy', 'Pendulum-v0', buffer_size=1000)
    model.learn(500)
    old_replay_buffer = deepcopy(model.replay_buffer)
    model.save_replay_buffer(log_folder)
    model.replay_buffer = None
    model.load_replay_buffer(replay_path)

    assert np.allclose(old_replay_buffer.observations, model.replay_buffer.observations)
    assert np.allclose(old_replay_buffer.actions, model.replay_buffer.actions)
    assert np.allclose(old_replay_buffer.next_observations, model.replay_buffer.next_observations)
    assert np.allclose(old_replay_buffer.rewards, model.replay_buffer.rewards)
    assert np.allclose(old_replay_buffer.dones, model.replay_buffer.dones)

    # test extending replay buffer
    model.replay_buffer.extend(old_replay_buffer.observations, old_replay_buffer.next_observations,
                               old_replay_buffer.actions, old_replay_buffer.rewards, old_replay_buffer.dones)

    # clear file from os
    os.remove(replay_path)
