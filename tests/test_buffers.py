import pytest

import numpy as np
from gym import spaces

from stable_baselines3.common.buffers import NstepReplayBuffer, ReplayBuffer

def test_nsteps():
    buffer = NstepReplayBuffer(
        5,
        spaces.Discrete(5),
        spaces.Discrete(5),
        n_step=5,
        gamma=1
    )
    buffer.add(0, 1, 10, 1, 0)
    buffer.add(1, 2, 11, 1, 0)
    buffer.add(2, 3, 12, 1, 0)
    buffer.add(3, 4, 13, 1, 0)
    buffer.add(4, 5, 14, 1, 0)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1,2,3,4]))
    assert obs.shape == (4, 1)
    assert act.shape == (4, 1)
    assert next_obs.shape == (4, 1)
    assert dones.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert np.allclose(dones, np.zeros_like(dones))
    assert np.allclose(next_obs, np.array([[5],[5],[5],[5]]))
    assert np.allclose(rewards, np.array([4, 3, 2, 1]).reshape(4, 1))
    assert np.allclose(act, np.array([11, 12, 13, 14]).reshape(4, 1))

    # shouldn't be able to get batch with indice 0 because the pointer is at 0
    with pytest.raises(AssertionError):
        buffer._get_samples(np.array([0, 1, 2, 3]))

    buffer = NstepReplayBuffer(
        5,
        spaces.Discrete(5),
        spaces.Discrete(5),
        n_step=5,
        gamma=0.9
    )
    buffer.add(0, 1, 10, 1, 0)
    buffer.add(1, 2, 11, 1, 0)
    buffer.add(2, 3, 12, 1, 0)
    buffer.add(3, 4, 13, 1, 0)
    buffer.add(4, 5, 14, 1, 0)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1,2,3,4]))
    assert obs.shape == (4, 1)
    assert act.shape == (4, 1)
    assert next_obs.shape == (4, 1)
    assert dones.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert np.allclose(dones, np.zeros_like(dones))
    assert np.allclose(next_obs, np.array([[5],[5],[5],[5]]))
    assert np.allclose(rewards, np.array([1+0.9+0.9**2+0.9**3, 1+0.9+0.9**2, 1+0.9, 1]).reshape(4, 1))
    assert np.allclose(act, np.array([11, 12, 13, 14]).reshape(4, 1))

    buffer = NstepReplayBuffer(
        10,
        spaces.Discrete(5),
        spaces.Discrete(5),
        n_step=5,
        gamma=0.9
    )
    buffer.add(0, 1, 10, 1, 0)
    buffer.add(1, 2, 11, 1, 0)
    buffer.add(2, 3, 12, 1, 0)
    buffer.add(3, 4, 13, 1, 0)
    buffer.add(4, 5, 14, 1, 0)
    obs, act, next_obs, dones, rewards = buffer._get_samples(np.array([1,2,3,4]))
    assert obs.shape == (4, 1)
    assert act.shape == (4, 1)
    assert next_obs.shape == (4, 1)
    assert dones.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert np.allclose(dones, np.zeros_like(dones))
    assert np.allclose(next_obs, np.array([[5],[5],[5],[5]]))
    assert np.allclose(rewards, np.array([1+0.9+0.9**2+0.9**3, 1+0.9+0.9**2, 1+0.9, 1]).reshape(4, 1))
    assert np.allclose(act, np.array([11, 12, 13, 14]).reshape(4, 1))

    # shouldn't be able to get batch with indice 5 because the pointer is at 5
    with pytest.raises(AssertionError):
        buffer._get_samples(np.array([5]))
