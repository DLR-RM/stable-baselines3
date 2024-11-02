import os
import shutil

import ale_py
import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

import stable_baselines3 as sb3
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.env_util import is_wrapped, make_atari_env, make_vec_env, unwrap_wrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, VectorizedActionNoise
from stable_baselines3.common.utils import (
    check_shape_equal,
    get_parameters_by_name,
    get_system_info,
    is_vectorized_observation,
    polyak_update,
    zip_strict,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

gym.register_envs(ale_py)


@pytest.mark.parametrize("env_id", ["CartPole-v1", lambda: gym.make("CartPole-v1")])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("vec_env_cls", [None, SubprocVecEnv])
@pytest.mark.parametrize("wrapper_class", [None, gym.wrappers.RecordEpisodeStatistics])
def test_make_vec_env(env_id, n_envs, vec_env_cls, wrapper_class):
    env = make_vec_env(env_id, n_envs, vec_env_cls=vec_env_cls, wrapper_class=wrapper_class, monitor_dir=None, seed=0)

    assert env.num_envs == n_envs

    if vec_env_cls is None:
        assert isinstance(env, DummyVecEnv)
        if wrapper_class is not None:
            assert isinstance(env.envs[0], wrapper_class)
        else:
            assert isinstance(env.envs[0], Monitor)
    else:
        assert isinstance(env, SubprocVecEnv)
    # Kill subprocesses
    env.close()


def test_make_vec_env_func_checker():
    """The functions in ``env_fns'' must return distinct instances since we need distinct environments."""
    env = gym.make("CartPole-v1")

    with pytest.raises(ValueError):
        make_vec_env(lambda: env, n_envs=2)

    env.close()


# Use Asterix as it does not requires fire reset
@pytest.mark.parametrize("env_id", ["BreakoutNoFrameskip-v4", "AsterixNoFrameskip-v4"])
@pytest.mark.parametrize("noop_max", [0, 10])
@pytest.mark.parametrize("action_repeat_probability", [0.0, 0.25])
@pytest.mark.parametrize("frame_skip", [1, 4])
@pytest.mark.parametrize("screen_size", [60])
@pytest.mark.parametrize("terminal_on_life_loss", [True, False])
@pytest.mark.parametrize("clip_reward", [True])
def test_make_atari_env(
    env_id, noop_max, action_repeat_probability, frame_skip, screen_size, terminal_on_life_loss, clip_reward
):
    n_envs = 2
    wrapper_kwargs = {
        "noop_max": noop_max,
        "action_repeat_probability": action_repeat_probability,
        "frame_skip": frame_skip,
        "screen_size": screen_size,
        "terminal_on_life_loss": terminal_on_life_loss,
        "clip_reward": clip_reward,
    }
    venv = make_atari_env(
        env_id,
        n_envs=2,
        wrapper_kwargs=wrapper_kwargs,
        monitor_dir=None,
        seed=0,
    )

    assert venv.num_envs == n_envs

    needs_fire_reset = env_id == "BreakoutNoFrameskip-v4"
    expected_frame_number_low = frame_skip * 2 if needs_fire_reset else 0  # FIRE - UP on reset
    expected_frame_number_high = expected_frame_number_low + noop_max
    expected_shape = (n_envs, screen_size, screen_size, 1)

    obs = venv.reset()
    frame_numbers = [env.unwrapped.ale.getEpisodeFrameNumber() for env in venv.envs]
    for frame_number in frame_numbers:
        assert expected_frame_number_low <= frame_number <= expected_frame_number_high
    assert obs.shape == expected_shape

    new_obs, reward, _, _ = venv.step([venv.action_space.sample() for _ in range(n_envs)])

    new_frame_numbers = [env.unwrapped.ale.getEpisodeFrameNumber() for env in venv.envs]
    for frame_number, new_frame_number in zip(frame_numbers, new_frame_numbers):
        assert new_frame_number - frame_number == frame_skip
    assert new_obs.shape == expected_shape
    if clip_reward:
        assert np.max(np.abs(reward)) < 1.0


def test_vec_env_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, env_kwargs={"goal_velocity": 0.11})
    assert env.get_attr("goal_velocity")[0] == 0.11


def test_vec_env_wrapper_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, wrapper_class=MaxAndSkipEnv, wrapper_kwargs={"skip": 3})
    assert env.get_attr("_skip")[0] == 3


def test_vec_env_monitor_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": False})
    assert env.get_attr("allow_early_resets")[0] is False

    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": False})
    assert env.get_attr("allow_early_resets")[0] is False

    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": True})
    assert env.get_attr("allow_early_resets")[0] is True

    env = make_atari_env(
        "BreakoutNoFrameskip-v4",
        n_envs=1,
        seed=0,
        monitor_kwargs={"allow_early_resets": True},
    )
    assert env.get_attr("allow_early_resets")[0] is True


def test_env_auto_monitor_wrap():
    env = gym.make("Pendulum-v1")
    model = A2C("MlpPolicy", env)
    assert model.env.env_is_wrapped(Monitor)[0] is True

    env = Monitor(env)
    model = A2C("MlpPolicy", env)
    assert model.env.env_is_wrapped(Monitor)[0] is True

    model = A2C("MlpPolicy", "Pendulum-v1")
    assert model.env.env_is_wrapped(Monitor)[0] is True


def test_custom_vec_env(tmp_path):
    """
    Stand alone test for a special case (passing a custom VecEnv class) to avoid doubling the number of tests.
    """
    monitor_dir = tmp_path / "test_make_vec_env/"
    env = make_vec_env(
        "CartPole-v1",
        n_envs=1,
        monitor_dir=monitor_dir,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": None},
    )

    assert env.num_envs == 1
    assert isinstance(env, SubprocVecEnv)
    assert os.path.isdir(monitor_dir)
    # Kill subprocess
    env.close()
    # Cleanup folder
    shutil.rmtree(monitor_dir)

    # This should fail because DummyVecEnv does not have any keyword argument
    with pytest.raises(TypeError):
        make_vec_env("CartPole-v1", n_envs=1, vec_env_kwargs={"dummy": False})


@pytest.mark.parametrize("direct_policy", [False, True])
def test_evaluate_policy(direct_policy):
    model = A2C("MlpPolicy", "Pendulum-v1", seed=0)
    n_steps_per_episode, n_eval_episodes = 200, 2

    def dummy_callback(locals_, _globals):
        locals_["model"].n_callback_calls += 1
        assert "observations" in locals_
        assert "new_observations" in locals_
        assert locals_["new_observations"] is not locals_["observations"]
        assert not np.allclose(locals_["new_observations"], locals_["observations"])

    assert model.policy is not None
    policy = model.policy if direct_policy else model

    policy.n_callback_calls = 0  # type: ignore[assignment, attr-defined]
    _, episode_lengths = evaluate_policy(
        policy,  # type: ignore[arg-type]
        model.get_env(),  # type: ignore[arg-type]
        n_eval_episodes,
        deterministic=True,
        render=False,
        callback=dummy_callback,
        reward_threshold=None,
        return_episode_rewards=True,
    )

    n_steps = sum(episode_lengths)  # type: ignore[arg-type]
    assert n_steps == n_steps_per_episode * n_eval_episodes
    assert n_steps == policy.n_callback_calls  # type: ignore[attr-defined]

    # Reaching a mean reward of zero is impossible with the Pendulum env
    with pytest.raises(AssertionError):
        evaluate_policy(policy, model.get_env(), n_eval_episodes, reward_threshold=0.0)  # type: ignore[arg-type]

    episode_rewards, _ = evaluate_policy(
        policy,  # type: ignore[arg-type]
        model.get_env(),  # type: ignore[arg-type]
        n_eval_episodes,
        return_episode_rewards=True,
    )
    assert len(episode_rewards) == n_eval_episodes  # type: ignore[arg-type]

    # Test that warning is given about no monitor
    eval_env = gym.make("Pendulum-v1")
    with pytest.warns(UserWarning):
        _ = evaluate_policy(policy, eval_env, n_eval_episodes)  # type: ignore[arg-type]


class ZeroRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward * 0


class AlwaysDoneWrapper(gym.Wrapper):
    # Pretends that environment only has single step for each
    # episode.
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None
        self.needs_reset = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.needs_reset = terminated or truncated
        self.last_obs = obs
        return obs, reward, True, truncated, info

    def reset(self, **kwargs):
        info = {}
        if self.needs_reset:
            obs, info = self.env.reset(**kwargs)
            self.last_obs = obs
            self.needs_reset = False
        return self.last_obs, info


@pytest.mark.parametrize("n_envs", [1, 2, 5, 7])
def test_evaluate_vector_env(n_envs):
    # Tests that the number of episodes evaluated is correct
    n_eval_episodes = 6

    env = make_vec_env("CartPole-v1", n_envs)
    model = A2C("MlpPolicy", "CartPole-v1", seed=0)

    class CountCallback:
        def __init__(self):
            self.count = 0

        def __call__(self, locals_, globals_):
            if locals_["done"]:
                self.count += 1

    count_callback = CountCallback()

    evaluate_policy(model, env, n_eval_episodes, callback=count_callback)

    assert count_callback.count == n_eval_episodes


@pytest.mark.parametrize("vec_env_class", [None, DummyVecEnv, SubprocVecEnv])
def test_evaluate_policy_monitors(vec_env_class):
    # Make numpy warnings throw exception
    np.seterr(all="raise")
    # Test that results are correct with monitor environments.
    # Also test VecEnvs
    n_eval_episodes = 3
    n_envs = 2
    env_id = "CartPole-v1"
    model = A2C("MlpPolicy", env_id, seed=0)

    def make_eval_env(with_monitor, wrapper_class=gym.Wrapper):
        # Make eval environment with or without monitor in root,
        # and additionally wrapped with another wrapper (after Monitor).
        env = None
        if vec_env_class is None:
            # No vecenv, traditional env
            env = gym.make(env_id)
            if with_monitor:
                env = Monitor(env)
            env = wrapper_class(env)
        else:
            if with_monitor:
                env = vec_env_class([lambda: wrapper_class(Monitor(gym.make(env_id)))] * n_envs)
            else:
                env = vec_env_class([lambda: wrapper_class(gym.make(env_id))] * n_envs)
        return env

    # Test that evaluation with VecEnvs works as expected
    eval_env = make_eval_env(with_monitor=True)
    _ = evaluate_policy(model, eval_env, n_eval_episodes)
    eval_env.close()

    # Warning without Monitor
    eval_env = make_eval_env(with_monitor=False)
    with pytest.warns(UserWarning):
        _ = evaluate_policy(model, eval_env, n_eval_episodes)
    eval_env.close()

    # Test that we gather correct reward with Monitor wrapper
    # Sanity check that we get zero-reward without Monitor
    eval_env = make_eval_env(with_monitor=False, wrapper_class=ZeroRewardWrapper)
    average_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes, warn=False)
    assert average_reward == 0.0, "ZeroRewardWrapper wrapper for testing did not work"
    eval_env.close()

    # Should get non-zero-rewards with Monitor (true reward)
    eval_env = make_eval_env(with_monitor=True, wrapper_class=ZeroRewardWrapper)
    average_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes)
    assert average_reward > 0.0, "evaluate_policy did not get reward from Monitor"
    eval_env.close()

    # Test that we also track correct episode dones, not the wrapped ones.
    # Sanity check that we get only one step per episode.
    eval_env = make_eval_env(with_monitor=False, wrapper_class=AlwaysDoneWrapper)
    episode_rewards, episode_lengths = evaluate_policy(
        model, eval_env, n_eval_episodes, return_episode_rewards=True, warn=False
    )
    assert all(map(lambda length: length == 1, episode_lengths)), "AlwaysDoneWrapper did not fix episode lengths to one"
    eval_env.close()

    # Should get longer episodes with with Monitor (true episodes)
    eval_env = make_eval_env(with_monitor=True, wrapper_class=AlwaysDoneWrapper)
    episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes, return_episode_rewards=True)
    assert all(map(lambda length: length > 1, episode_lengths)), "evaluate_policy did not get episode lengths from Monitor"
    eval_env.close()


def test_vec_noise():
    num_envs = 4
    num_actions = 10
    mu = np.zeros(num_actions)
    sigma = np.ones(num_actions) * 0.4
    base = OrnsteinUhlenbeckActionNoise(mu, sigma)
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(base, -1)
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(base, None)
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(base, "whatever")

    vec = VectorizedActionNoise(base, num_envs)
    assert vec.n_envs == num_envs
    assert vec().shape == (num_envs, num_actions)
    assert not (vec() == base()).all()
    with pytest.raises(ValueError):
        vec = VectorizedActionNoise(None, num_envs)
    with pytest.raises(TypeError):
        vec = VectorizedActionNoise(12, num_envs)
    with pytest.raises(AssertionError):
        vec.noises = []
    with pytest.raises(TypeError):
        vec.noises = None
    with pytest.raises(ValueError):
        vec.noises = [None] * vec.n_envs
    with pytest.raises(AssertionError):
        vec.noises = [base] * (num_envs - 1)
    assert all(isinstance(noise, type(base)) for noise in vec.noises)
    assert len(vec.noises) == num_envs


def test_get_parameters_by_name():
    model = th.nn.Sequential(th.nn.Linear(5, 5), th.nn.BatchNorm1d(5))
    # Initialize stats
    model(th.ones(3, 5))
    included_names = ["weight", "bias", "running_"]
    # 2 x weight, 2 x bias, 1 x running_mean, 1 x running_var; Ignore num_batches_tracked.
    parameters = get_parameters_by_name(model, included_names)
    assert len(parameters) == 6
    assert th.allclose(parameters[4], model[1].running_mean)
    assert th.allclose(parameters[5], model[1].running_var)
    parameters = get_parameters_by_name(model, ["running_"])
    assert len(parameters) == 2
    assert th.allclose(parameters[0], model[1].running_mean)
    assert th.allclose(parameters[1], model[1].running_var)


def test_polyak():
    param1, param2 = th.nn.Parameter(th.ones((5, 5))), th.nn.Parameter(th.zeros((5, 5)))
    target1, target2 = th.nn.Parameter(th.ones((5, 5))), th.nn.Parameter(th.zeros((5, 5)))
    tau = 0.1
    polyak_update([param1], [param2], tau)
    with th.no_grad():
        for param, target_param in zip([target1], [target2]):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    assert th.allclose(param1, target1)
    assert th.allclose(param2, target2)


def test_zip_strict():
    # Iterables with different lengths
    list_a = [0, 1]
    list_b = [1, 2, 3]
    # zip does not raise any error
    for _, _ in zip(list_a, list_b):
        pass

    # zip_strict does raise an error
    with pytest.raises(ValueError):
        for _, _ in zip_strict(list_a, list_b):
            pass

    # same length, should not raise an error
    for _, _ in zip_strict(list_a, list_b[: len(list_a)]):
        pass


def test_is_wrapped():
    """Test that is_wrapped correctly detects wraps"""
    env = gym.make("Pendulum-v1")
    env = gym.Wrapper(env)
    assert not is_wrapped(env, Monitor)
    monitor_env = Monitor(env)
    assert is_wrapped(monitor_env, Monitor)
    env = gym.Wrapper(monitor_env)
    assert is_wrapped(env, Monitor)
    # Test that unwrap works as expected
    assert unwrap_wrapper(env, Monitor) == monitor_env


def test_get_system_info():
    info, info_str = get_system_info(print_info=True)
    assert info["Stable-Baselines3"] == str(sb3.__version__)
    assert "Python" in info_str
    assert "PyTorch" in info_str
    assert "GPU Enabled" in info_str
    assert "Numpy" in info_str
    assert "Gym" in info_str


def test_is_vectorized_observation():
    # with pytest.raises("ValueError"):
    #     pass
    # All vectorized
    box_space = spaces.Box(-1, 1, shape=(2,))
    box_obs = np.ones((1, *box_space.shape))
    assert is_vectorized_observation(box_obs, box_space)

    discrete_space = spaces.Discrete(2)
    discrete_obs = np.ones((3,), dtype=np.int8)
    assert is_vectorized_observation(discrete_obs, discrete_space)

    multidiscrete_space = spaces.MultiDiscrete([2, 3])
    multidiscrete_obs = np.ones((1, 2), dtype=np.int8)
    assert is_vectorized_observation(multidiscrete_obs, multidiscrete_space)

    multibinary_space = spaces.MultiBinary(3)
    multibinary_obs = np.ones((1, 3), dtype=np.int8)
    assert is_vectorized_observation(multibinary_obs, multibinary_space)

    dict_space = spaces.Dict({"box": box_space, "discrete": discrete_space})
    dict_obs = {"box": box_obs, "discrete": discrete_obs}
    assert is_vectorized_observation(dict_obs, dict_space)

    # All not vectorized
    box_obs = np.ones(box_space.shape)
    assert not is_vectorized_observation(box_obs, box_space)

    discrete_obs = np.ones((), dtype=np.int8)
    assert not is_vectorized_observation(discrete_obs, discrete_space)

    multidiscrete_obs = np.ones((2,), dtype=np.int8)
    assert not is_vectorized_observation(multidiscrete_obs, multidiscrete_space)

    multibinary_obs = np.ones((3,), dtype=np.int8)
    assert not is_vectorized_observation(multibinary_obs, multibinary_space)

    dict_obs = {"box": box_obs, "discrete": discrete_obs}
    assert not is_vectorized_observation(dict_obs, dict_space)

    # A mix of vectorized and non-vectorized things
    with pytest.raises(ValueError):
        discrete_obs = np.ones((1,), dtype=np.int8)
        dict_obs = {"box": box_obs, "discrete": discrete_obs}
        is_vectorized_observation(dict_obs, dict_space)

    # Vectorized with the wrong shape
    with pytest.raises(ValueError):
        discrete_obs = np.ones((1,), dtype=np.int8)
        box_obs = np.ones((1, 2, *box_space.shape))
        dict_obs = {"box": box_obs, "discrete": discrete_obs}
        is_vectorized_observation(dict_obs, dict_space)

    # Weird shape: error
    with pytest.raises(ValueError):
        discrete_obs = np.ones((1, *box_space.shape), dtype=np.int8)
        is_vectorized_observation(discrete_obs, discrete_space)

    # wrong shape
    with pytest.raises(ValueError):
        multidiscrete_obs = np.ones((2, 1), dtype=np.int8)
        is_vectorized_observation(multidiscrete_obs, multidiscrete_space)

    # wrong shape
    with pytest.raises(ValueError):
        multibinary_obs = np.ones((2, 1), dtype=np.int8)
        is_vectorized_observation(multidiscrete_obs, multibinary_space)

    # Almost good shape: one dimension too much for Discrete obs
    with pytest.raises(ValueError):
        box_obs = np.ones((1, *box_space.shape))
        discrete_obs = np.ones((1, 1), dtype=np.int8)
        dict_obs = {"box": box_obs, "discrete": discrete_obs}
        is_vectorized_observation(dict_obs, dict_space)


def test_policy_is_vectorized_obs():
    """
    Additional tests to check `policy.is_vectorized()`
    which handle transposing image to channel-first if needed.

    We check for basic cases, the rest is handled
    by is_vectorized_observation() helper.
    """
    policy = sb3.DQN("MlpPolicy", "CartPole-v1").policy

    box_space = spaces.Box(-1, 1, shape=(2,))
    box_obs = np.ones((1, *box_space.shape))
    policy.observation_space = box_space
    assert policy.is_vectorized_observation(box_obs)
    assert not policy.is_vectorized_observation(np.ones(box_space.shape))

    discrete_space = spaces.Discrete(2)
    discrete_obs = np.ones((3,), dtype=np.int8)
    policy.observation_space = discrete_space
    assert not policy.is_vectorized_observation(np.ones((), dtype=np.int8))

    dict_space = spaces.Dict({"box": box_space, "discrete": discrete_space})
    dict_obs = {"box": box_obs, "discrete": discrete_obs}
    policy.observation_space = dict_space
    assert policy.is_vectorized_observation(dict_obs)
    dict_obs = {"box": np.ones(box_space.shape), "discrete": np.ones((), dtype=np.int8)}
    assert not policy.is_vectorized_observation(dict_obs)

    # Image space are channel-first (done automatically in SB3 using VecTranspose)
    # but observation passed is channel last
    image_space = spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8)

    image_channel_first = image_space.sample()
    image_channel_last = np.transpose(image_channel_first, (1, 2, 0))
    policy.observation_space = image_space
    assert not policy.is_vectorized_observation(image_channel_first)
    assert not policy.is_vectorized_observation(image_channel_last)
    assert policy.is_vectorized_observation(image_channel_first[np.newaxis])
    assert policy.is_vectorized_observation(image_channel_last[np.newaxis])

    # Same with dict obs
    dict_space = spaces.Dict({"image": image_space})
    policy.observation_space = dict_space
    assert not policy.is_vectorized_observation({"image": image_channel_first})
    assert not policy.is_vectorized_observation({"image": image_channel_last})
    assert policy.is_vectorized_observation({"image": image_channel_first[np.newaxis]})
    assert policy.is_vectorized_observation({"image": image_channel_last[np.newaxis]})


def test_check_shape_equal():
    space1 = spaces.Box(low=0, high=1, shape=(2, 2))
    space2 = spaces.Box(low=-1, high=1, shape=(2, 2))
    check_shape_equal(space1, space2)

    space1 = spaces.Box(low=0, high=1, shape=(2, 2))
    space2 = spaces.Box(low=-1, high=2, shape=(3, 3))
    with pytest.raises(AssertionError):
        check_shape_equal(space1, space2)

    space1 = spaces.Dict({"key1": spaces.Box(low=0, high=1, shape=(2, 2)), "key2": spaces.Box(low=0, high=1, shape=(2, 2))})
    space2 = spaces.Dict({"key1": spaces.Box(low=-1, high=2, shape=(2, 2)), "key2": spaces.Box(low=-1, high=2, shape=(2, 2))})
    check_shape_equal(space1, space2)

    space1 = spaces.Dict({"key1": spaces.Box(low=0, high=1, shape=(2, 2)), "key2": spaces.Box(low=0, high=1, shape=(2, 2))})
    space2 = spaces.Dict({"key1": spaces.Box(low=-1, high=2, shape=(3, 3)), "key2": spaces.Box(low=-1, high=2, shape=(2, 2))})
    with pytest.raises(AssertionError):
        check_shape_equal(space1, space2)
