import json
import os
import uuid

import gym
import pandas

from stable_baselines3.common.monitor import Monitor, get_monitor_files, load_results


def test_monitor(tmp_path):
    """
    Test the monitor wrapper
    """
    env = gym.make("CartPole-v1")
    env.seed(0)
    monitor_file = os.path.join(str(tmp_path), f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env = Monitor(env, monitor_file)
    monitor_env.reset()
    total_steps = 1000
    ep_rewards = []
    ep_lengths = []
    ep_len, ep_reward = 0, 0
    for _ in range(total_steps):
        _, reward, done, _ = monitor_env.step(monitor_env.action_space.sample())
        ep_len += 1
        ep_reward += reward
        if done:
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            monitor_env.reset()
            ep_len, ep_reward = 0, 0

    monitor_env.close()
    assert monitor_env.get_total_steps() == total_steps
    assert sum(ep_lengths) == sum(monitor_env.get_episode_lengths())
    assert sum(monitor_env.get_episode_rewards()) == sum(ep_rewards)
    _ = monitor_env.get_episode_times()

    with open(monitor_file) as file_handler:
        first_line = file_handler.readline()
        assert first_line.startswith("#")
        metadata = json.loads(first_line[1:])
        assert metadata["env_id"] == "CartPole-v1"
        assert set(metadata.keys()) == {"env_id", "t_start"}, "Incorrect keys in monitor metadata"

        last_logline = pandas.read_csv(file_handler, index_col=None)
        assert set(last_logline.keys()) == {"l", "t", "r"}, "Incorrect keys in monitor logline"
    os.remove(monitor_file)

    # Check missing filename directories are created
    monitor_dir = os.path.join(str(tmp_path), "missing-dir")
    monitor_file = os.path.join(monitor_dir, f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    assert os.path.exists(monitor_dir) is False
    _ = Monitor(env, monitor_file)
    assert os.path.exists(monitor_dir) is True
    os.remove(monitor_file)
    os.rmdir(monitor_dir)


def test_monitor_load_results(tmp_path):
    """
    test load_results on log files produced by the monitor wrapper
    """
    tmp_path = str(tmp_path)
    env1 = gym.make("CartPole-v1")
    env1.seed(0)
    monitor_file1 = os.path.join(tmp_path, f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env1 = Monitor(env1, monitor_file1)

    monitor_files = get_monitor_files(tmp_path)
    assert len(monitor_files) == 1
    assert monitor_file1 in monitor_files

    monitor_env1.reset()
    episode_count1 = 0
    for _ in range(1000):
        _, _, done, _ = monitor_env1.step(monitor_env1.action_space.sample())
        if done:
            episode_count1 += 1
            monitor_env1.reset()

    results_size1 = len(load_results(os.path.join(tmp_path)).index)
    assert results_size1 == episode_count1

    env2 = gym.make("CartPole-v1")
    env2.seed(0)
    monitor_file2 = os.path.join(tmp_path, f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env2 = Monitor(env2, monitor_file2)
    monitor_files = get_monitor_files(tmp_path)
    assert len(monitor_files) == 2
    assert monitor_file1 in monitor_files
    assert monitor_file2 in monitor_files

    episode_count2 = 0
    for _ in range(2):
        # Test appending to existing file
        monitor_env2 = Monitor(env2, monitor_file2, override_existing=False)
        monitor_env2.reset()
        for _ in range(1000):
            _, _, done, _ = monitor_env2.step(monitor_env2.action_space.sample())
            if done:
                episode_count2 += 1
                monitor_env2.reset()

    results_size2 = len(load_results(os.path.join(tmp_path)).index)

    assert results_size2 == (results_size1 + episode_count2)

    os.remove(monitor_file1)
    os.remove(monitor_file2)
