import os
from copy import deepcopy

import numpy as np
import pytest
import torch as th

from stable_baselines3 import DDPG, DQN, SAC, TD3
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her import HER


@pytest.mark.parametrize("model_class, policy", [(SAC, "MlpPolicy"), (TD3, "MlpPolicy"), (DDPG, "MlpPolicy")])
@pytest.mark.parametrize("online_sampling", [True, False])
def test_her(model_class, policy, online_sampling):
    """
    Test Hindsight Experience Replay.
    """
    n_bits = 4
    env = BitFlippingEnv(n_bits=n_bits, continuous=True)
    env = DummyVecEnv([lambda: env])

    # Create action noise
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        np.zeros(
            n_actions,
        ),
        0.2 * np.ones((n_actions,)),
    )

    model = HER(
        policy,
        env,
        model_class,
        n_sampled_goal=5,
        goal_selection_strategy="future",
        online_sampling=online_sampling,
        action_noise=action_noise,
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
    )

    model.learn(total_timesteps=500, callback=None)

    # Evaluate the agent
    n_eval_episodes = 5
    n_episodes = 0
    episode_rewards = []
    episode_reward = 0.0

    eval_env = BitFlippingEnv(n_bits=4, continuous=True)

    observation = eval_env.reset()

    while n_episodes < n_eval_episodes:

        obs = np.concatenate([observation["observation"], observation["desired_goal"]])

        with th.no_grad():
            obs_ = th.FloatTensor(np.array(obs).reshape(1, -1)).to(model.model.device)
            action = model.model.policy.predict(obs_)[0][0]

        observation, reward, done, _ = eval_env.step(action)

        # Render the env
        # eval_env.render()

        episode_reward += reward

        if done:
            n_episodes += 1
            observation = eval_env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

    eval_env.close()


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
    env = DummyVecEnv([lambda: env])

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
    )
    model.learn(total_timesteps=200, callback=None)


@pytest.mark.parametrize("model_class, policy", [(SAC, "MlpPolicy"), (TD3, "MlpPolicy"), (DDPG, "MlpPolicy")])
@pytest.mark.parametrize("use_sde", [False, True])
def test_save_load(tmp_path, model_class, policy, use_sde):
    """
    Test if 'save' and 'load' saves and loads model correctly
    """
    if use_sde and model_class != SAC:
        pytest.skip("Only SAC has gSDE support")

    n_bits = 4
    env = BitFlippingEnv(n_bits=n_bits, continuous=True)
    env = DummyVecEnv([lambda: env])

    # Create action noise
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        np.zeros(
            n_actions,
        ),
        0.2 * np.ones((n_actions,)),
    )

    kwargs = dict(use_sde=True) if use_sde else {}

    # create model
    model = HER(
        policy,
        env,
        model_class,
        n_sampled_goal=5,
        goal_selection_strategy="future",
        online_sampling=True,
        action_noise=action_noise,
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

    model.learn(total_timesteps=500, callback=None)

    env.reset()

    observations_list = []
    for _ in range(10):
        obs = env.step([env.action_space.sample()])[0]
        observation = ObsDictWrapper.convert_dict(obs)
        observations_list.append(observation)

    observations = np.concatenate(observations_list, axis=0)

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

    # offline
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

    model.learn(total_timesteps=20000)
