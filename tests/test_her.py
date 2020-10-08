import os
from copy import deepcopy

import numpy as np
import pytest
import torch as th

from stable_baselines3 import DDPG, DQN, HER, SAC, TD3
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


@pytest.mark.parametrize("model_class", [SAC, TD3, DDPG, DQN])
@pytest.mark.parametrize("online_sampling", [True, False])
def test_her(model_class, online_sampling):
    """
    Test Hindsight Experience Replay.
    """
    n_bits = 4
    env = BitFlippingEnv(n_bits=n_bits, continuous=not (model_class == DQN))

    model = HER(
        "MlpPolicy",
        env,
        model_class,
        goal_selection_strategy="future",
        online_sampling=online_sampling,
        gradient_steps=1,
        train_freq=1,
        n_episodes_rollout=-1,
        max_episode_length=n_bits,
        policy_kwargs=dict(net_arch=[64]),
    )

    model.learn(total_timesteps=500)


@pytest.mark.parametrize(
    "goal_selection_strategy",
    [
        "final",
        "episode",
        "future",
        "random",
        GoalSelectionStrategy.FUTURE,
        GoalSelectionStrategy.RANDOM,
        GoalSelectionStrategy.EPISODE,
        GoalSelectionStrategy.FINAL,
    ],
)
@pytest.mark.parametrize("online_sampling", [True, False])
def test_goal_selection_strategy(goal_selection_strategy, online_sampling):
    """
    Test different goal strategies.
    """
    env = BitFlippingEnv(continuous=True)

    model = HER(
        "MlpPolicy",
        env,
        SAC,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        gradient_steps=1,
        train_freq=1,
        n_episodes_rollout=-1,
        max_episode_length=10,
        policy_kwargs=dict(net_arch=[64]),
    )
    model.learn(total_timesteps=200)


@pytest.mark.parametrize("model_class", [SAC, TD3, DDPG, DQN])
@pytest.mark.parametrize("use_sde", [False, True])
def test_save_load(tmp_path, model_class, use_sde):
    """
    Test if 'save' and 'load' saves and loads model correctly
    """
    if use_sde and model_class != SAC:
        pytest.skip("Only SAC has gSDE support")

    n_bits = 4
    env = BitFlippingEnv(n_bits=n_bits, continuous=not (model_class == DQN))

    kwargs = dict(use_sde=True) if use_sde else {}

    # create model
    model = HER(
        "MlpPolicy",
        env,
        model_class,
        n_sampled_goal=5,
        goal_selection_strategy="future",
        online_sampling=True,
        verbose=0,
        tau=0.05,
        batch_size=128,
        learning_rate=0.001,
        policy_kwargs=dict(net_arch=[64]),
        buffer_size=int(1e6),
        gamma=0.98,
        gradient_steps=1,
        train_freq=1,
        n_episodes_rollout=-1,
        max_episode_length=n_bits,
        **kwargs
    )

    model.learn(total_timesteps=500)

    env.reset()

    observations_list = []
    for _ in range(10):
        obs = env.step(env.action_space.sample())[0]
        observation = ObsDictWrapper.convert_dict(obs)
        observations_list.append(observation)
    observations = np.array(observations_list)

    # Get dictionary of current parameters
    params = deepcopy(model.model.policy.state_dict())

    # Modify all parameters to be random values
    random_params = dict((param_name, th.rand_like(param)) for param_name, param in params.items())

    # Update model parameters with the new random values
    model.model.policy.load_state_dict(random_params)

    new_params = model.model.policy.state_dict()
    # Check that all params are different now
    for k in params:
        assert not th.allclose(params[k], new_params[k]), "Parameters did not change as expected."

    params = new_params

    # get selected actions
    selected_actions, _ = model.model.predict(observations, deterministic=True)

    # Check
    model.save(tmp_path / "test_save.zip")
    del model
    model = HER.load(str(tmp_path / "test_save.zip"), env=env)

    # check if params are still the same after load
    new_params = model.model.policy.state_dict()

    # Check that all params are the same as before save load procedure now
    for key in params:
        assert th.allclose(params[key], new_params[key]), "Model parameters not the same after save and load."

    # check if model still selects the same actions
    new_selected_actions, _ = model.model.predict(observations, deterministic=True)
    assert np.allclose(selected_actions, new_selected_actions, 1e-4)

    # check if learn still works
    model.learn(total_timesteps=1000, eval_freq=500)

    # clear file from os
    os.remove(tmp_path / "test_save.zip")


@pytest.mark.parametrize("online_sampling", [False, True])
@pytest.mark.parametrize("n_bits", [15])
def test_dqn_her(online_sampling, n_bits):
    """
    Test HER with DQN for BitFlippingEnv.
    """
    env = BitFlippingEnv(n_bits=n_bits, continuous=False)

    model = HER(
        "MlpPolicy",
        env,
        DQN,
        n_sampled_goal=5,
        goal_selection_strategy="future",
        online_sampling=online_sampling,
        verbose=1,
        learning_rate=0.0005,
        max_episode_length=n_bits,
        train_freq=1,
        learning_starts=100,
        exploration_final_eps=0.02,
        target_update_interval=500,
        seed=0,
        batch_size=32,
    )

    model.learn(total_timesteps=10000)

    assert np.mean(model.ep_success_buffer) > 0.0
