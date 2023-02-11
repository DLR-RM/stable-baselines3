import csv
import json
import os
import uuid
import warnings

import gymnasium as gym
import pandas
import pytest

from stable_baselines3 import PPO
from stable_baselines3.common.envs.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor, get_monitor_files, load_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


def test_vec_monitor(tmp_path):
    """
    Test the `VecMonitor` wrapper
    """
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env.seed(0)
    monitor_file = os.path.join(str(tmp_path), f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env = VecMonitor(env, monitor_file)
    monitor_env.reset()
    total_steps = 1000
    ep_len, ep_reward = 0, 0
    for _ in range(total_steps):
        _, rewards, dones, infos = monitor_env.step([monitor_env.action_space.sample()])
        ep_len += 1
        ep_reward += rewards[0]
        if dones[0]:
            assert ep_reward == infos[0]["episode"]["r"]
            assert ep_len == infos[0]["episode"]["l"]
            ep_len, ep_reward = 0, 0

    monitor_env.close()

    with open(monitor_file) as file_handler:
        first_line = file_handler.readline()
        assert first_line.startswith("#")
        metadata = json.loads(first_line[1:])
        assert set(metadata.keys()) == {"t_start", "env_id"}, "Incorrect keys in monitor metadata"

        last_logline = pandas.read_csv(file_handler, index_col=None)
        assert set(last_logline.keys()) == {"l", "t", "r"}, "Incorrect keys in monitor logline"
    os.remove(monitor_file)


def test_vec_monitor_info_keywords(tmp_path):
    """
    Test loggig `info_keywords` in the `VecMonitor` wrapper
    """
    monitor_file = os.path.join(str(tmp_path), f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")

    env = DummyVecEnv([lambda: BitFlippingEnv()])

    monitor_env = VecMonitor(env, info_keywords=("is_success",), filename=monitor_file)

    monitor_env.reset()
    total_steps = 1000
    for _ in range(total_steps):
        _, _, dones, infos = monitor_env.step([monitor_env.action_space.sample()])
        if dones[0]:
            assert "is_success" in infos[0]["episode"]

    monitor_env.close()

    with open(monitor_file) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0 or i == 1:
                continue
            else:
                assert len(line) == 4, "Incorrect keys in monitor logline"
                assert line[3] in ["False", "True"], "Incorrect value in monitor logline"

    os.remove(monitor_file)


def test_vec_monitor_load_results(tmp_path):
    """
    test load_results on log files produced by the monitor wrapper
    """
    tmp_path = str(tmp_path)
    env1 = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env1.seed(0)
    monitor_file1 = os.path.join(str(tmp_path), f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env1 = VecMonitor(env1, monitor_file1)

    monitor_files = get_monitor_files(tmp_path)
    assert len(monitor_files) == 1
    assert monitor_file1 in monitor_files

    monitor_env1.reset()
    episode_count1 = 0
    for _ in range(1000):
        _, _, dones, _ = monitor_env1.step([monitor_env1.action_space.sample()])
        if dones[0]:
            episode_count1 += 1
            monitor_env1.reset()

    results_size1 = len(load_results(os.path.join(tmp_path)).index)
    assert results_size1 == episode_count1

    env2 = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env2.seed(0)
    monitor_file2 = os.path.join(str(tmp_path), f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env2 = VecMonitor(env2, monitor_file2)
    monitor_files = get_monitor_files(tmp_path)
    assert len(monitor_files) == 2
    assert monitor_file1 in monitor_files
    assert monitor_file2 in monitor_files

    monitor_env2.reset()
    episode_count2 = 0
    for _ in range(1000):
        _, _, dones, _ = monitor_env2.step([monitor_env2.action_space.sample()])
        if dones[0]:
            episode_count2 += 1
            monitor_env2.reset()

    results_size2 = len(load_results(os.path.join(tmp_path)).index)

    assert results_size2 == (results_size1 + episode_count2)

    os.remove(monitor_file1)
    os.remove(monitor_file2)


def test_vec_monitor_ppo(recwarn):
    """
    Test the `VecMonitor` with PPO
    """
    warnings.filterwarnings(action="ignore", category=DeprecationWarning, module=r".*passive_env_checker")
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env.seed(seed=0)
    monitor_env = VecMonitor(env)
    model = PPO("MlpPolicy", monitor_env, verbose=1, n_steps=64, device="cpu")
    model.learn(total_timesteps=250)

    # No warnings because using `VecMonitor`
    evaluate_policy(model, monitor_env)
    assert len(recwarn) == 0, f"{[str(warning) for warning in recwarn]}"


def test_vec_monitor_warn():
    env = DummyVecEnv([lambda: Monitor(gym.make("CartPole-v1"))])
    # We should warn the user when the env is already wrapped with a Monitor wrapper
    with pytest.warns(UserWarning):
        VecMonitor(env)

    with pytest.warns(UserWarning):
        VecMonitor(VecNormalize(env))
