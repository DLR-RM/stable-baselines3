import numpy as np
import pytest
from gym import spaces

from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.buffers import NstepReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise


def test_nsteps():
    buffer = NstepReplayBuffer(5, spaces.Discrete(5), spaces.Discrete(5), n_step=5, gamma=1)
    buffer.add(0, 1, 10, 1, 0)
    buffer.add(1, 2, 11, 1, 0)
    buffer.add(2, 3, 12, 1, 0)
    buffer.add(3, 4, 13, 1, 0)
    buffer.add(4, 5, 14, 1, 0)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1, 2, 3, 4]))
    assert obs.shape == (4, 1)
    assert act.shape == (4, 1)
    assert next_obs.shape == (4, 1)
    assert dones.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert np.allclose(dones, np.zeros_like(dones))
    assert np.allclose(next_obs, np.array([[5], [5], [5], [5]]))
    assert np.allclose(rewards, np.array([4, 3, 2, 1]).reshape(4, 1))
    assert np.allclose(act, np.array([11, 12, 13, 14]).reshape(4, 1))

    # shouldn't be able to get batch with indice 0 because the pointer is at 0
    with pytest.raises(AssertionError):
        buffer._get_samples(np.array([0, 1, 2, 3]))

    buffer = NstepReplayBuffer(5, spaces.Discrete(5), spaces.Discrete(5), n_step=5, gamma=0.9)
    buffer.add(0, 1, 10, 1, 0)
    buffer.add(1, 2, 11, 1, 0)
    buffer.add(2, 3, 12, 1, 0)
    buffer.add(3, 4, 13, 1, 0)
    buffer.add(4, 5, 14, 1, 0)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1, 2, 3, 4]))
    assert obs.shape == (4, 1)
    assert act.shape == (4, 1)
    assert next_obs.shape == (4, 1)
    assert dones.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert np.allclose(dones, np.zeros_like(dones))
    assert np.allclose(next_obs, np.array([[5], [5], [5], [5]]))
    assert np.allclose(rewards, np.array([1 + 0.9 + 0.9 ** 2 + 0.9 ** 3, 1 + 0.9 + 0.9 ** 2, 1 + 0.9, 1]).reshape(4, 1))
    assert np.allclose(act, np.array([11, 12, 13, 14]).reshape(4, 1))

    buffer = NstepReplayBuffer(10, spaces.Discrete(5), spaces.Discrete(5), n_step=5, gamma=0.9)
    buffer.add(0, 1, 10, 1, 0)
    buffer.add(1, 2, 11, 1, 0)
    buffer.add(2, 3, 12, 1, 0)
    buffer.add(3, 4, 13, 1, 0)
    buffer.add(4, 5, 14, 1, 0)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1, 2, 3, 4]))
    assert obs.shape == (4, 1)
    assert act.shape == (4, 1)
    assert next_obs.shape == (4, 1)
    assert dones.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert np.allclose(dones, np.zeros_like(dones))
    assert np.allclose(next_obs, np.array([[5], [5], [5], [5]]))
    assert np.allclose(rewards, np.array([1 + 0.9 + 0.9 ** 2 + 0.9 ** 3, 1 + 0.9 + 0.9 ** 2, 1 + 0.9, 1]).reshape(4, 1))
    assert np.allclose(act, np.array([11, 12, 13, 14]).reshape(4, 1))

    # shouldn't be able to get batch with indice 5 because the pointer is at 5
    with pytest.raises(AssertionError):
        buffer._get_samples(np.array([5]))

    buffer = NstepReplayBuffer(10, spaces.Discrete(5), spaces.Discrete(5), n_step=5, gamma=0.9)
    buffer.add(0, 1, 10, 1, 1)
    buffer.add(1, 2, 11, 1, 1)
    buffer.add(2, 3, 12, 1, 1)
    buffer.add(3, 4, 13, 1, 1)
    buffer.add(4, 5, 14, 1, 1)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1, 2, 3, 4]))
    assert obs.shape == (4, 1)
    assert act.shape == (4, 1)
    assert next_obs.shape == (4, 1)
    assert dones.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert np.allclose(dones, np.ones_like(dones))
    assert np.allclose(rewards, np.array([1, 1, 1, 1]).reshape(4, 1))
    assert np.allclose(act, np.array([11, 12, 13, 14]).reshape(4, 1))
    assert np.allclose(next_obs, np.array([[2], [3], [4], [5]]))

    buffer = NstepReplayBuffer(5, spaces.Discrete(5), spaces.Discrete(5), n_step=2, gamma=0.99)
    buffer.add(0, 1, 10, 2, 0)
    buffer.add(1, 2, 11, 1, 0)
    buffer.add(2, 3, 12, 2, 0)
    buffer.add(3, 4, 13, 2, 1)
    buffer.add(4, 5, 14, 2, 0)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1, 1, 2, 3, 4]))
    assert obs.shape == (5, 1)
    assert act.shape == (5, 1)
    assert next_obs.shape == (5, 1)
    assert dones.shape == (5, 1)
    assert rewards.shape == (5, 1)
    assert np.allclose(dones, np.array([0, 0, 1, 1, 0]).reshape(5, 1))
    assert np.allclose(rewards, np.array([2.98, 2.98, 3.98, 2, 2]).reshape(5, 1))
    assert np.allclose(act, np.array([11, 11, 12, 13, 14]).reshape(5, 1))
    assert np.allclose(next_obs, np.array([[3], [3], [4], [4], [5]]))


@pytest.mark.parametrize("algo", [DQN, TD3])
def test_with_algo(algo):
    # Integration test
    kwargs = {"policy_kwargs": dict(net_arch=[64]), "_init_setup_model": False}
    if algo in [TD3, SAC]:
        env_id = "Pendulum-v0"
        kwargs.update({"action_noise": NormalActionNoise(0.0, 0.1), "learning_starts": 100})
    else:
        env_id = "CartPole-v1"
        if algo == DQN:
            kwargs.update({"learning_starts": 100})
    agent = algo("MlpPolicy", env_id, **kwargs)
    agent.replay_buffer_cls = NstepReplayBuffer
    agent.replay_buffer_kwargs.update({"gamma": agent.gamma, "n_step": 10})
    agent._setup_model()
    agent.learn(500)
