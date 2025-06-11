import gymnasium as gym
import numpy as np
import pytest

from stable_baselines3 import DQN, SAC, TD3
from stable_baselines3.common.buffers import NStepReplayBuffer, ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env


@pytest.mark.parametrize("model_class", [SAC, DQN, TD3])
def test_run(model_class):
    env_id = "CartPole-v1" if model_class == DQN else "Pendulum-v1"
    env = make_vec_env(env_id, n_envs=2)
    gamma = 0.989

    model = model_class(
        "MlpPolicy",
        env,
        train_freq=4,
        n_steps=3,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=100,
        buffer_size=int(2e4),
        gamma=gamma,
    )
    assert isinstance(model.replay_buffer, NStepReplayBuffer)
    assert model.replay_buffer.n_steps == 3
    assert model.replay_buffer.gamma == gamma

    model.learn(total_timesteps=150)


def create_buffer(buffer_size=10, n_steps=3, gamma=0.99, n_envs=1):
    obs_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
    return NStepReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        n_envs=n_envs,
        n_steps=n_steps,
        gamma=gamma,
    )


def create_normal_buffer(buffer_size=10, n_envs=1):
    obs_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
    return ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        n_envs=n_envs,
    )


def fill_buffer(buffer, length, done_at=None, truncated_at=None):
    """
    Fill the buffer with:
    - reward = 1.0
    - observation = index
    - optional `done` at index `done_at`
    - optional truncation at index `truncated_at`
    """
    for i in range(length):
        obs = np.full((1, 4), i, dtype=np.float32)
        next_obs = np.full((1, 4), i + 1, dtype=np.float32)
        action = np.zeros((1, 2), dtype=np.float32)
        reward = np.array([1.0])
        done = np.array([1.0 if i == done_at else 0.0])
        truncated = i == truncated_at
        infos = [{"TimeLimit.truncated": truncated}]
        buffer.add(obs, next_obs, action, reward, done, infos)


def compute_expected_nstep_reward(gamma, n_steps, stop_idx=None):
    """
    Compute the expected n-step reward for the test env (reward=1 for each step),
    optionally stopping early due to termination/truncation.
    """
    returns = np.zeros(n_steps)
    rewards = np.ones(n_steps)
    last_sum = 0.0
    for step in reversed(range(n_steps)):
        next_non_terminal = step != stop_idx
        last_sum = rewards[step] + gamma * next_non_terminal * last_sum
        returns[step] = last_sum
    return returns[0]


@pytest.mark.parametrize("done_at", [1, 2])
@pytest.mark.parametrize("n_steps", [3, 5])
@pytest.mark.parametrize("base_idx", [0, 2])
def test_nstep_early_termination(done_at, n_steps, base_idx):
    gamma = 0.98
    buffer = create_buffer(n_steps=n_steps, gamma=gamma)
    fill_buffer(buffer, length=10, done_at=done_at)

    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.item()
    expected = compute_expected_nstep_reward(gamma=gamma, n_steps=n_steps, stop_idx=done_at - base_idx)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert batch.dones.item() == float(base_idx <= done_at)


@pytest.mark.parametrize("truncated_at", [1, 2])
@pytest.mark.parametrize("n_steps", [2, 5])
@pytest.mark.parametrize("base_idx", [0, 1])
def test_nstep_early_truncation(truncated_at, n_steps, base_idx):
    buffer = create_buffer(n_steps=n_steps)
    fill_buffer(buffer, length=10, truncated_at=truncated_at)

    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.item()

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=truncated_at - base_idx)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert batch.dones.item() == 0.0


@pytest.mark.parametrize("n_steps", [3, 5])
def test_nstep_no_terminations(n_steps):
    buffer = create_buffer(n_steps=n_steps)
    fill_buffer(buffer, length=10)  # no done or truncation
    gamma = 0.99

    base_idx = 3
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.item()
    # Discount factor for bootstrapping with target Q-Value
    np.testing.assert_allclose(batch.discounts.item(), gamma**n_steps)
    expected = compute_expected_nstep_reward(gamma=gamma, n_steps=n_steps)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert batch.dones.item() == 0.0

    # Check that self.pos-1 truncation is set when buffer is full
    # Note: buffer size is 10, here we are erasing past transitions
    fill_buffer(buffer, length=2)
    # We create a tmp truncation to not sample across episodes
    base_idx = 0
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.item()
    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=buffer.pos - 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert batch.dones.item() == 0.0
    # Discount factor for bootstrapping with target Q-Value
    # (not equal to gamma ** n_steps because of truncation at n_steps=2)
    np.testing.assert_allclose(batch.discounts.item(), gamma**2)

    # Set done=1 manually, the tmp truncation should not be set (it would set batch.done=False)
    buffer.dones[buffer.pos - 1, :] = True
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.item()
    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=buffer.pos - 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert batch.dones.item() == 1.0


def test_match_normal_buffer():
    buffer = create_buffer(n_steps=1)
    ref_buffer = create_normal_buffer()
    # no done or truncation
    fill_buffer(buffer, length=10)
    fill_buffer(ref_buffer, length=10)

    base_idx = 3
    batch1 = buffer._get_samples(np.array([base_idx]))
    actual1 = batch1.rewards.item()

    batch2 = ref_buffer._get_samples(np.array([base_idx]))

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=1)
    np.testing.assert_allclose(actual1, expected, rtol=1e-6)
    assert batch1.dones.item() == 0.0

    np.testing.assert_allclose(batch1.rewards.numpy(), batch2.rewards.numpy(), rtol=1e-6)
