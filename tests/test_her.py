import os
import pathlib
import warnings
from copy import deepcopy

import gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import DDPG, DQN, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import get_time_limit


def test_import_error():
    with pytest.raises(ImportError) as excinfo:
        from stable_baselines3 import HER

        HER("MlpPolicy")
    assert "documentation" in str(excinfo.value)


@pytest.mark.parametrize("model_class", [SAC, TD3, DDPG, DQN])
@pytest.mark.parametrize("online_sampling", [True, False])
@pytest.mark.parametrize("image_obs_space", [True, False])
def test_her(model_class, online_sampling, image_obs_space):
    """
    Test Hindsight Experience Replay.
    """
    n_bits = 4
    env = BitFlippingEnv(
        n_bits=n_bits,
        continuous=not (model_class == DQN),
        image_obs_space=image_obs_space,
    )

    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=2,
            goal_selection_strategy="future",
            online_sampling=online_sampling,
            max_episode_length=n_bits,
        ),
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=100,
        buffer_size=int(2e4),
    )

    model.learn(total_timesteps=150)
    evaluate_policy(model, Monitor(env))


@pytest.mark.parametrize(
    "goal_selection_strategy",
    [
        "final",
        "episode",
        "future",
        GoalSelectionStrategy.FINAL,
        GoalSelectionStrategy.EPISODE,
        GoalSelectionStrategy.FUTURE,
    ],
)
@pytest.mark.parametrize("online_sampling", [True, False])
def test_goal_selection_strategy(goal_selection_strategy, online_sampling):
    """
    Test different goal strategies.
    """
    env = BitFlippingEnv(continuous=True)

    normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
            max_episode_length=10,
            n_sampled_goal=2,
        ),
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=100,
        buffer_size=int(1e5),
        action_noise=normal_action_noise,
    )
    assert model.action_noise is not None
    model.learn(total_timesteps=150)


@pytest.mark.parametrize("model_class", [SAC, TD3, DDPG, DQN])
@pytest.mark.parametrize("use_sde", [False, True])
@pytest.mark.parametrize("online_sampling", [False, True])
def test_save_load(tmp_path, model_class, use_sde, online_sampling):
    """
    Test if 'save' and 'load' saves and loads model correctly
    """
    if use_sde and model_class != SAC:
        pytest.skip("Only SAC has gSDE support")

    n_bits = 4
    env = BitFlippingEnv(n_bits=n_bits, continuous=not (model_class == DQN))

    kwargs = dict(use_sde=True) if use_sde else {}

    # create model
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=2,
            goal_selection_strategy="future",
            online_sampling=online_sampling,
            max_episode_length=n_bits,
        ),
        verbose=0,
        tau=0.05,
        batch_size=128,
        learning_rate=0.001,
        policy_kwargs=dict(net_arch=[64]),
        buffer_size=int(1e5),
        gamma=0.98,
        gradient_steps=1,
        train_freq=4,
        learning_starts=100,
        **kwargs
    )

    model.learn(total_timesteps=150)

    obs = env.reset()

    observations = {key: [] for key in obs.keys()}
    for _ in range(10):
        obs = env.step(env.action_space.sample())[0]
        for key in obs.keys():
            observations[key].append(obs[key])
    observations = {key: np.array(obs) for key, obs in observations.items()}

    # Get dictionary of current parameters
    params = deepcopy(model.policy.state_dict())

    # Modify all parameters to be random values
    random_params = {param_name: th.rand_like(param) for param_name, param in params.items()}

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
    model.save(tmp_path / "test_save.zip")
    del model

    # test custom_objects
    # Load with custom objects
    custom_objects = dict(learning_rate=2e-5, dummy=1.0)
    model_ = model_class.load(str(tmp_path / "test_save.zip"), env=env, custom_objects=custom_objects, verbose=2)
    assert model_.verbose == 2
    # Check that the custom object was taken into account
    assert model_.learning_rate == custom_objects["learning_rate"]
    # Check that only parameters that are here already are replaced
    assert not hasattr(model_, "dummy")

    model = model_class.load(str(tmp_path / "test_save.zip"), env=env)

    # check if params are still the same after load
    new_params = model.policy.state_dict()

    # Check that all params are the same as before save load procedure now
    for key in params:
        assert th.allclose(params[key], new_params[key]), "Model parameters not the same after save and load."

    # check if model still selects the same actions
    new_selected_actions, _ = model.predict(observations, deterministic=True)
    assert np.allclose(selected_actions, new_selected_actions, 1e-4)

    # check if learn still works
    model.learn(total_timesteps=150)

    # Test that the change of parameters works
    model = model_class.load(str(tmp_path / "test_save.zip"), env=env, verbose=3, learning_rate=2.0)
    assert model.learning_rate == 2.0
    assert model.verbose == 3

    # clear file from os
    os.remove(tmp_path / "test_save.zip")


@pytest.mark.parametrize("online_sampling", [False, True])
@pytest.mark.parametrize("truncate_last_trajectory", [False, True])
def test_save_load_replay_buffer(tmp_path, recwarn, online_sampling, truncate_last_trajectory):
    """
    Test if 'save_replay_buffer' and 'load_replay_buffer' works correctly
    """
    # remove gym warnings
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", category=UserWarning, module="gym")

    path = pathlib.Path(tmp_path / "replay_buffer.pkl")
    path.parent.mkdir(exist_ok=True, parents=True)  # to not raise a warning
    env = BitFlippingEnv(n_bits=4, continuous=True)
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=2,
            goal_selection_strategy="future",
            online_sampling=online_sampling,
            max_episode_length=4,
        ),
        gradient_steps=1,
        train_freq=4,
        buffer_size=int(2e4),
        policy_kwargs=dict(net_arch=[64]),
        seed=1,
    )
    model.learn(200)
    if online_sampling:
        old_replay_buffer = deepcopy(model.replay_buffer)
    else:
        old_replay_buffer = deepcopy(model.replay_buffer.replay_buffer)
    model.save_replay_buffer(path)
    del model.replay_buffer

    with pytest.raises(AttributeError):
        model.replay_buffer

    # Check that there is no warning
    assert len(recwarn) == 0

    model.load_replay_buffer(path, truncate_last_traj=truncate_last_trajectory)

    if truncate_last_trajectory:
        assert len(recwarn) == 1
        warning = recwarn.pop(UserWarning)
        assert "The last trajectory in the replay buffer will be truncated" in str(warning.message)
    else:
        assert len(recwarn) == 0

    if online_sampling:
        n_episodes_stored = model.replay_buffer.n_episodes_stored
        assert np.allclose(
            old_replay_buffer._buffer["observation"][:n_episodes_stored],
            model.replay_buffer._buffer["observation"][:n_episodes_stored],
        )
        assert np.allclose(
            old_replay_buffer._buffer["next_obs"][:n_episodes_stored],
            model.replay_buffer._buffer["next_obs"][:n_episodes_stored],
        )
        assert np.allclose(
            old_replay_buffer._buffer["action"][:n_episodes_stored],
            model.replay_buffer._buffer["action"][:n_episodes_stored],
        )
        assert np.allclose(
            old_replay_buffer._buffer["reward"][:n_episodes_stored],
            model.replay_buffer._buffer["reward"][:n_episodes_stored],
        )
        # we might change the last done of the last trajectory so we don't compare it
        assert np.allclose(
            old_replay_buffer._buffer["done"][: n_episodes_stored - 1],
            model.replay_buffer._buffer["done"][: n_episodes_stored - 1],
        )
    else:
        replay_buffer = model.replay_buffer.replay_buffer
        assert np.allclose(old_replay_buffer.observations["observation"], replay_buffer.observations["observation"])
        assert np.allclose(old_replay_buffer.observations["desired_goal"], replay_buffer.observations["desired_goal"])
        assert np.allclose(old_replay_buffer.actions, replay_buffer.actions)
        assert np.allclose(old_replay_buffer.rewards, replay_buffer.rewards)
        assert np.allclose(old_replay_buffer.dones, replay_buffer.dones)

    # test if continuing training works properly
    reset_num_timesteps = False if truncate_last_trajectory is False else True
    model.learn(200, reset_num_timesteps=reset_num_timesteps)


def test_full_replay_buffer():
    """
    Test if HER works correctly with a full replay buffer when using online sampling.
    It should not sample the current episode which is not finished.
    """
    n_bits = 4
    env = BitFlippingEnv(n_bits=n_bits, continuous=True)

    # use small buffer size to get the buffer full
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=2,
            goal_selection_strategy="future",
            online_sampling=True,
            max_episode_length=n_bits,
        ),
        gradient_steps=1,
        train_freq=4,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=1,
        buffer_size=20,
        verbose=1,
        seed=757,
    )

    model.learn(total_timesteps=100)


def test_get_max_episode_length():
    dict_env = DummyVecEnv([lambda: BitFlippingEnv()])

    # Cannot infer max epsiode length
    with pytest.raises(ValueError):
        get_time_limit(dict_env, current_max_episode_length=None)

    default_length = 10
    assert get_time_limit(dict_env, current_max_episode_length=default_length) == default_length

    env = gym.make("CartPole-v1")
    vec_env = DummyVecEnv([lambda: env])

    assert get_time_limit(vec_env, current_max_episode_length=None) == 500
    # Overwrite max_episode_steps
    assert get_time_limit(vec_env, current_max_episode_length=default_length) == default_length

    # Set max_episode_steps to None
    env.spec.max_episode_steps = None
    vec_env = DummyVecEnv([lambda: env])
    with pytest.raises(ValueError):
        get_time_limit(vec_env, current_max_episode_length=None)

    # Initialize HER and specify max_episode_length, should not raise an issue
    DQN("MultiInputPolicy", dict_env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(max_episode_length=5))

    with pytest.raises(ValueError):
        DQN("MultiInputPolicy", dict_env, replay_buffer_class=HerReplayBuffer)

    # Wrapped in a timelimit, should be fine
    # Note: it requires env.spec to be defined
    env = DummyVecEnv([lambda: gym.wrappers.TimeLimit(BitFlippingEnv(), 10)])
    DQN("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(max_episode_length=5))


@pytest.mark.parametrize("online_sampling", [False, True])
@pytest.mark.parametrize("n_bits", [10])
def test_performance_her(online_sampling, n_bits):
    """
    That DQN+HER can solve BitFlippingEnv.
    It should not work when n_sampled_goal=0 (DQN alone).
    """
    env = BitFlippingEnv(n_bits=n_bits, continuous=False)

    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=5,
            goal_selection_strategy="future",
            online_sampling=online_sampling,
            max_episode_length=n_bits,
        ),
        verbose=1,
        learning_rate=5e-4,
        train_freq=1,
        learning_starts=100,
        exploration_final_eps=0.02,
        target_update_interval=500,
        seed=0,
        batch_size=32,
        buffer_size=int(1e5),
    )

    model.learn(total_timesteps=5000, log_interval=50)

    # 90% training success
    assert np.mean(model.ep_success_buffer) > 0.90
