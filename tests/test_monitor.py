import uuid
import json
import os

import pandas
import gym

from torchy_baselines.common.monitor import Monitor, get_monitor_files, load_results


def test_monitor():
    """
    test the monitor wrapper
    """
    env = gym.make("CartPole-v1")
    env.seed(0)
    monitor_file = "/tmp/stable_baselines-test-{}.monitor.csv".format(uuid.uuid4())
    monitor_env = Monitor(env, monitor_file)
    monitor_env.reset()
    for _ in range(1000):
        _, _, done, _ = monitor_env.step(0)
        if done:
            monitor_env.reset()

    file_handler = open(monitor_file, 'rt')

    first_line = file_handler.readline()
    assert first_line.startswith('#')
    metadata = json.loads(first_line[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 't_start'}, "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(file_handler, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    file_handler.close()
    os.remove(monitor_file)

def test_monitor_load_results(tmp_path):
    """
    test load_results on log files produced by the monitor wrapper
    """
    tmp_path = str(tmp_path)
    env1 = gym.make("CartPole-v1")
    env1.seed(0)
    monitor_file1 = os.path.join(tmp_path, "stable_baselines-test-{}.monitor.csv".format(uuid.uuid4()))
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
    monitor_file2 = os.path.join(tmp_path, "stable_baselines-test-{}.monitor.csv".format(uuid.uuid4()))
    monitor_env2 = Monitor(env2, monitor_file2)
    monitor_files = get_monitor_files(tmp_path)
    assert len(monitor_files) == 2
    assert monitor_file1 in monitor_files
    assert monitor_file2 in monitor_files

    monitor_env2.reset()
    episode_count2 = 0
    for _ in range(1000):
        _, _, done, _ = monitor_env2.step(monitor_env2.action_space.sample())
        if done:
            episode_count2 += 1
            monitor_env2.reset()

    results_size2 = len(load_results(os.path.join(tmp_path)).index)

    assert results_size2 == (results_size1 + episode_count2)

    os.remove(monitor_file1)
    os.remove(monitor_file2)
