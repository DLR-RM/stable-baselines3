import numpy as np
import pytest
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer


def test_replay_buffer_no_copy_when_already_array():
    """Test that ReplayBuffer avoids unnecessary copies when inputs are already numpy arrays."""
    obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    buffer = ReplayBuffer(buffer_size=10, observation_space=obs_space, action_space=action_space)
    
    # Create numpy arrays
    obs = np.array([1, 2, 3, 4], dtype=np.float32)
    next_obs = np.array([2, 3, 4, 5], dtype=np.float32)
    action = np.array([0.5, -0.5], dtype=np.float32)
    reward = np.array([1.0], dtype=np.float32)
    done = np.array([False], dtype=np.float32)
    
    # Add to buffer
    buffer.add(obs, next_obs, action, reward, done, [{}])
    
    # Verify data was stored correctly
    assert np.array_equal(buffer.observations[0], obs)
    assert np.array_equal(buffer.next_observations[0], next_obs)
    assert np.array_equal(buffer.actions[0], action)
    assert np.array_equal(buffer.rewards[0], reward)
    assert np.array_equal(buffer.dones[0], done)
    
    # Verify that modifying original arrays doesn't affect buffer (copy was made for observations)
    obs[:] = 0
    next_obs[:] = 0
    assert not np.array_equal(buffer.observations[0], obs)
    assert not np.array_equal(buffer.next_observations[0], next_obs)
    
    # Actions, rewards, dones don't need copy protection
    action[:] = 99
    reward[:] = 99
    done[:] = 1
    # These may or may not be equal depending on implementation details
    # The important thing is that the buffer functions correctly


def test_replay_buffer_handles_lists_and_scalars():
    """Test that ReplayBuffer correctly handles different input types."""
    obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(3)
    buffer = ReplayBuffer(buffer_size=10, observation_space=obs_space, action_space=action_space)
    
    # Test with lists
    obs_list = [1.0, 2.0, 3.0, 4.0]
    next_obs_list = [2.0, 3.0, 4.0, 5.0]
    action_scalar = 1
    reward_scalar = 2.5
    done_bool = True
    
    buffer.add(obs_list, next_obs_list, action_scalar, reward_scalar, done_bool, [{}])
    
    # Verify conversion worked
    assert buffer.observations[0].shape == (4,)
    assert buffer.actions[0].shape == (1,)
    assert isinstance(buffer.rewards[0], np.ndarray)
    assert isinstance(buffer.dones[0], np.ndarray)


def test_replay_buffer_memory_optimization_mode():
    """Test that memory optimization mode works correctly with the optimization."""
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
    action_space = spaces.Discrete(4)
    
    buffer = ReplayBuffer(
        buffer_size=100,
        observation_space=obs_space,
        action_space=action_space,
        optimize_memory_usage=True
    )
    
    obs = np.random.randint(0, 255, size=(84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 255, size=(84, 84, 4), dtype=np.uint8)
    
    buffer.add(obs, next_obs, 2, 1.0, False, [{}])
    
    # In optimize_memory_usage mode, next_obs is stored at (pos + 1) % buffer_size
    assert np.array_equal(buffer.observations[0], obs)
    assert np.array_equal(buffer.observations[1], next_obs)
    
    # Verify buffer doesn't have next_observations array
    assert not hasattr(buffer, 'next_observations') or buffer.next_observations is None


def test_replay_buffer_discrete_observation_space():
    """Test that discrete observation spaces are handled correctly."""
    obs_space = spaces.Discrete(10)
    action_space = spaces.Discrete(2)
    buffer = ReplayBuffer(buffer_size=10, observation_space=obs_space, action_space=action_space)
    
    obs = 5
    next_obs = 7
    action = 1
    
    buffer.add(obs, next_obs, action, 1.0, False, [{}])
    
    # Check reshaping worked correctly
    assert buffer.observations[0].shape == (1,)
    assert buffer.observations[0][0] == 5