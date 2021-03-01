import json
import os
import uuid

import gym
import pandas

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import get_monitor_files, load_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def test_vec_monitor(tmp_path):
    """
    Test the `VecMonitor` wrapper
    """
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env.seed(0)
    monitor_file = os.path.join(str(tmp_path), "stable_baselines-test-{}.monitor.csv".format(uuid.uuid4()))
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

    with open(monitor_file, "rt") as file_handler:
        first_line = file_handler.readline()
        assert first_line.startswith("#")
        metadata = json.loads(first_line[1:])
        assert set(metadata.keys()) == {"t_start"}, "Incorrect keys in monitor metadata"

        last_logline = pandas.read_csv(file_handler, index_col=None)
        assert set(last_logline.keys()) == {"l", "t", "r"}, "Incorrect keys in monitor logline"
    os.remove(monitor_file)


def test_vec_monitor_load_results(tmp_path):
    """
    test load_results on log files produced by the monitor wrapper
    """
    tmp_path = str(tmp_path)
    env1 = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env1.seed(0)
    monitor_file1 = os.path.join(str(tmp_path), "stable_baselines-test-{}.monitor.csv".format(uuid.uuid4()))
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
    monitor_file2 = os.path.join(str(tmp_path), "stable_baselines-test-{}.monitor.csv".format(uuid.uuid4()))
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


def test_vec_monitor_with_PPO():
    """
    Test the `VecMonitor` with PPO
    """
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env.seed(0)
    monitor_env = VecMonitor(env)
    model = PPO("MlpPolicy", monitor_env, verbose=3, n_steps=16, device="cpu")
    model.learn(total_timesteps=250)
