import os
import shutil

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import BitFlippingEnv, IdentityEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def select_env(model_class) -> str:
    if model_class is DQN:
        return "CartPole-v1"
    else:
        return "Pendulum-v1"


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, TD3, DQN, DDPG])
def test_callbacks(tmp_path, model_class):
    log_folder = tmp_path / "logs/callbacks/"

    # DQN only support discrete actions
    env_id = select_env(model_class)
    # Create RL model
    # Small network for fast test
    model = model_class("MlpPolicy", env_id, policy_kwargs=dict(net_arch=[32]))

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_folder)

    eval_env = gym.make(env_id)
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)

    # Stop training if there is no model improvement after 2 evaluations
    callback_no_model_improvement = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=1, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        callback_after_eval=callback_no_model_improvement,
        best_model_save_path=log_folder,
        log_path=log_folder,
        eval_freq=100,
        warn=False,
    )
    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=log_folder, name_prefix="event")

    event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

    # Stop training if max number of episodes is reached
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=100, verbose=1)

    callback = CallbackList([checkpoint_callback, eval_callback, event_callback, callback_max_episodes])
    model.learn(500, callback=callback)

    # Check access to local variables
    assert model.env.observation_space.contains(callback.locals["new_obs"][0])
    # Check that the child callback was called
    assert checkpoint_callback.locals["new_obs"] is callback.locals["new_obs"]
    assert event_callback.locals["new_obs"] is callback.locals["new_obs"]
    assert checkpoint_on_event.locals["new_obs"] is callback.locals["new_obs"]
    # Check that internal callback counters match models' counters
    assert event_callback.num_timesteps == model.num_timesteps
    assert event_callback.n_calls == model.num_timesteps

    model.learn(500, callback=None)
    # Transform callback into a callback list automatically and use progress bar
    model.learn(500, callback=[checkpoint_callback, eval_callback], progress_bar=True)
    # Automatic wrapping, old way of doing callbacks
    model.learn(500, callback=lambda _locals, _globals: True)

    # Testing models that support multiple envs
    if model_class in [A2C, PPO]:
        max_episodes = 1
        n_envs = 2
        # Pendulum-v1 has a timelimit of 200 timesteps
        max_episode_length = 200
        envs = make_vec_env(env_id, n_envs=n_envs, seed=0)

        model = model_class("MlpPolicy", envs, policy_kwargs=dict(net_arch=[32]))

        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
        callback = CallbackList([callback_max_episodes])
        model.learn(1000, callback=callback)

        # Check that the actual number of episodes and timesteps per env matches the expected one
        episodes_per_env = callback_max_episodes.n_episodes // n_envs
        assert episodes_per_env == max_episodes
        timesteps_per_env = model.num_timesteps // n_envs
        assert timesteps_per_env == max_episode_length

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)


def test_eval_callback_vec_env():
    # tests that eval callback does not crash when given a vector
    n_eval_envs = 3
    train_env = IdentityEnv()
    eval_env = DummyVecEnv([lambda: IdentityEnv()] * n_eval_envs)
    model = A2C("MlpPolicy", train_env, seed=0)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=100,
        warn=False,
    )
    model.learn(300, callback=eval_callback)
    assert eval_callback.last_mean_reward == 100.0


class AlwaysFailCallback(BaseCallback):
    def __init__(self, *args, callback_false_value, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_false_value = callback_false_value

    def _on_step(self) -> bool:
        return self.callback_false_value


@pytest.mark.parametrize(
    "model_class,model_kwargs",
    [
        (A2C, dict(n_steps=1, stats_window_size=1)),
        (
            SAC,
            dict(
                learning_starts=1,
                buffer_size=1,
                batch_size=1,
            ),
        ),
    ],
)
@pytest.mark.parametrize("callback_false_value", [False, np.bool_(0), th.tensor(0, dtype=th.bool)])
def test_callbacks_can_cancel_runs(model_class, model_kwargs, callback_false_value):
    assert not callback_false_value  # Sanity check to ensure parametrized values are valid
    env_id = select_env(model_class)
    model = model_class("MlpPolicy", env_id, **model_kwargs, policy_kwargs=dict(net_arch=[2]))
    alwaysfailcallback = AlwaysFailCallback(callback_false_value=callback_false_value)
    model.learn(10, callback=alwaysfailcallback)

    assert alwaysfailcallback.n_calls == 1


def test_eval_success_logging(tmp_path):
    n_bits = 2
    n_envs = 2
    env = BitFlippingEnv(n_bits=n_bits)
    eval_env = DummyVecEnv([lambda: BitFlippingEnv(n_bits=n_bits)] * n_envs)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=250,
        log_path=tmp_path,
        warn=False,
    )
    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        learning_starts=100,
        seed=0,
    )
    model.learn(500, callback=eval_callback)
    assert len(eval_callback._is_success_buffer) > 0
    # More than 50% success rate
    assert np.mean(eval_callback._is_success_buffer) > 0.5


def test_eval_callback_logs_are_written_with_the_correct_timestep(tmp_path):
    # Skip if no tensorboard installed
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    env_id = select_env(DQN)
    model = DQN(
        "MlpPolicy",
        env_id,
        policy_kwargs=dict(net_arch=[32]),
        tensorboard_log=tmp_path,
        verbose=1,
        seed=1,
    )

    eval_env = gym.make(env_id)
    eval_freq = 101
    eval_callback = EvalCallback(eval_env, eval_freq=eval_freq, warn=False)
    model.learn(500, callback=eval_callback)

    acc = EventAccumulator(str(tmp_path / "DQN_1"))
    acc.Reload()
    for event in acc.scalars.Items("eval/mean_reward"):
        assert event.step % eval_freq == 0


def test_eval_friendly_error():
    # tests that eval callback does not crash when given a vector
    train_env = VecNormalize(DummyVecEnv([lambda: gym.make("CartPole-v1")]))
    eval_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    _ = train_env.reset()
    original_obs = train_env.get_original_obs()
    model = A2C("MlpPolicy", train_env, n_steps=50, seed=0)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=100,
        warn=False,
    )
    model.learn(100, callback=eval_callback)

    # Check synchronization
    assert np.allclose(train_env.normalize_obs(original_obs), eval_env.normalize_obs(original_obs))

    wrong_eval_env = gym.make("CartPole-v1")
    eval_callback = EvalCallback(
        wrong_eval_env,
        eval_freq=100,
        warn=False,
    )

    with pytest.warns(Warning):
        with pytest.raises(AssertionError):
            model.learn(100, callback=eval_callback)


def test_checkpoint_additional_info(tmp_path):
    # tests if the replay buffer and the VecNormalize stats are saved with every checkpoint
    dummy_vec_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env = VecNormalize(dummy_vec_env)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_callback = CheckpointCallback(
        save_freq=200,
        save_path=checkpoint_dir,
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=2,
    )

    model = DQN("MlpPolicy", env, learning_starts=100, buffer_size=500, seed=0)
    model.learn(200, callback=checkpoint_callback)

    assert os.path.exists(checkpoint_dir / "rl_model_200_steps.zip")
    assert os.path.exists(checkpoint_dir / "rl_model_replay_buffer_200_steps.pkl")
    assert os.path.exists(checkpoint_dir / "rl_model_vecnormalize_200_steps.pkl")
    # Check that checkpoints can be properly loaded
    model = DQN.load(checkpoint_dir / "rl_model_200_steps.zip")
    model.load_replay_buffer(checkpoint_dir / "rl_model_replay_buffer_200_steps.pkl")
    VecNormalize.load(checkpoint_dir / "rl_model_vecnormalize_200_steps.pkl", dummy_vec_env)


def test_eval_callback_chaining(tmp_path):
    class DummyCallback(BaseCallback):
        def _on_step(self):
            # Check that the parent callback is an EvalCallback
            assert isinstance(self.parent, EvalCallback)
            assert hasattr(self.parent, "best_mean_reward")
            return True

    stop_on_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)

    eval_callback = EvalCallback(
        gym.make("Pendulum-v1"),
        best_model_save_path=tmp_path,
        log_path=tmp_path,
        eval_freq=32,
        deterministic=True,
        render=False,
        callback_on_new_best=CallbackList([DummyCallback(), stop_on_threshold_callback]),
        callback_after_eval=CallbackList([DummyCallback()]),
        warn=False,
    )

    model = PPO("MlpPolicy", "Pendulum-v1", n_steps=64, n_epochs=1)
    model.learn(64, callback=eval_callback)
