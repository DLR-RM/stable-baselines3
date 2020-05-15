import os
from copy import deepcopy

import pytest
import numpy as np
import torch as th

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.identity_env import IdentityEnvBox
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.identity_env import FakeImageEnv


MODEL_LIST = [
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
    model = model_class('MlpPolicy', env, policy_kwargs=dict(net_arch=[16]), verbose=1)
    model.learn(total_timesteps=500, eval_freq=250)

    env.reset()
    observations = np.concatenate([env.step(env.action_space.sample())[0] for _ in range(10)], axis=0)

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
    selected_actions, _ = model.predict(observations, deterministic=True)

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
    new_selected_actions, _ = model.predict(observations, deterministic=True)
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
    model = model_class('MlpPolicy', env, policy_kwargs=dict(net_arch=[16]))
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
    model = model_class('MlpPolicy', env, policy_kwargs=dict(net_arch=[16]), verbose=2)

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


@pytest.mark.parametrize("model_class", MODEL_LIST)
@pytest.mark.parametrize("policy_str", ['MlpPolicy', 'CnnPolicy'])
def test_save_load_policy(model_class, policy_str):
    """
    Test saving and loading policy only.

    :param model_class: (BaseRLModel) A RL model
    :param policy_str: (str) Name of the policy.
    """
    kwargs = {}
    if policy_str == 'MlpPolicy':
        env = IdentityEnvBox(10)
    else:
        if model_class in [SAC, TD3]:
            # Avoid memory error when using replay buffer
            # Reduce the size of the features
            kwargs = dict(buffer_size=250)
        env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=2,
                           discrete=False)

    env = DummyVecEnv([lambda: env])

    # create model
    model = model_class(policy_str, env, policy_kwargs=dict(net_arch=[16]),
                        verbose=1, **kwargs)
    model.learn(total_timesteps=500, eval_freq=250)

    env.reset()
    observations = np.concatenate([env.step(env.action_space.sample())[0] for _ in range(10)], axis=0)

    policy = model.policy
    policy_class = policy.__class__
    actor, actor_class = None, None
    if model_class in [SAC, TD3]:
        actor = policy.actor
        actor_class = actor.__class__

    # Get dictionary of current parameters
    params = deepcopy(policy.state_dict())

    # Modify all parameters to be random values
    random_params = dict((param_name, th.rand_like(param)) for param_name, param in params.items())

    # Update model parameters with the new random values
    policy.load_state_dict(random_params)

    new_params = policy.state_dict()
    # Check that all params are different now
    for k in params:
        assert not th.allclose(params[k], new_params[k]), "Parameters did not change as expected."

    params = new_params

    # get selected actions
    selected_actions, _ = policy.predict(observations, deterministic=True)
    # Should also work with the actor only
    if actor is not None:
        selected_actions_actor, _ = actor.predict(observations, deterministic=True)

    # Save and load policy
    policy.save("./logs/policy.pkl")
    # Save and load actor
    if actor is not None:
        actor.save("./logs/actor.pkl")

    del policy, actor

    policy = policy_class.load("./logs/policy.pkl")
    if actor_class is not None:
        actor = actor_class.load("./logs/actor.pkl")

    # check if params are still the same after load
    new_params = policy.state_dict()

    # Check that all params are the same as before save load procedure now
    for key in params:
        assert th.allclose(params[key], new_params[key]), "Policy parameters not the same after save and load."

    # check if model still selects the same actions
    new_selected_actions, _ = policy.predict(observations, deterministic=True)
    assert np.allclose(selected_actions, new_selected_actions, 1e-4)

    if actor_class is not None:
        new_selected_actions_actor, _ = actor.predict(observations, deterministic=True)
        assert np.allclose(selected_actions_actor, new_selected_actions_actor, 1e-4)
        assert np.allclose(selected_actions_actor, new_selected_actions, 1e-4)

    # clear file from os
    os.remove("./logs/policy.pkl")
    if actor_class is not None:
        os.remove("./logs/actor.pkl")
