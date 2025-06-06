import gymnasium as gym
import numpy as np
import pytest

from stable_baselines3 import DQN, SAC
from stable_baselines3.common.buffers import NStepReplayBuffer, ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env


@pytest.mark.parametrize("model_class", [SAC, DQN])
def test_run(model_class):
    env_id = "CartPole-v1" if model_class == DQN else "Pendulum-v1"
    env = make_vec_env(env_id, n_envs=2)

    # FIXME: need to set the discount factor manually
    n_steps = 2
    gamma = 0.99
    discount = gamma**n_steps

    model = model_class(
        "MlpPolicy",
        env,
        replay_buffer_class=NStepReplayBuffer,
        replay_buffer_kwargs=dict(
            n_steps=n_steps,
            gamma=gamma,
        ),
        train_freq=4,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=100,
        buffer_size=int(2e4),
        gamma=discount,
    )

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
    Compute the expected n-step reward starting from zero idx,
    optionally stopping early due to termination/truncation.
    """
    rewards = [1.0 * (gamma**i) for i in range(n_steps)]
    if stop_idx is not None:
        rewards = rewards[: stop_idx + 1]
    return sum(rewards)


def compute_expected_nstep_reward2(gamma, n_steps, stop_idx=None):
    """
    Compute the expected n-step reward,
    optionally stopping early due to termination/truncation.
    Alternative implementation that can handle different rewards.
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
def test_nstep_early_termination(done_at, n_steps):
    buffer = create_buffer(n_steps=n_steps)
    fill_buffer(buffer, length=5, done_at=done_at)

    base_idx = 0
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.numpy().item()

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=done_at - base_idx)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    assert batch.dones.numpy().item() == 1.0


@pytest.mark.parametrize("truncated_at", [1, 2])
def test_nstep_early_truncation(truncated_at):
    buffer = create_buffer()
    fill_buffer(buffer, length=5, truncated_at=truncated_at)

    base_idx = 0
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.numpy().item()

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=3, stop_idx=truncated_at - base_idx)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    assert batch.dones.numpy().item() == 0.0


@pytest.mark.parametrize("n_steps", [3, 5])
def test_nstep_no_termination_or_truncation(n_steps):
    buffer = create_buffer(n_steps=n_steps)
    fill_buffer(buffer, length=10)  # no done or truncation

    base_idx = 3
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.numpy().item()

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    assert batch.dones.numpy().item() == 0.0

    # Check that self.pos-1 truncation is set when buffer is full
    # Note: buffer size is 10, here we are erasing past transitions
    fill_buffer(buffer, length=2)
    # We create a tmp truncation to not sample across episodes
    base_idx = 0
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.numpy().item()
    # Note: compute_expected_nstep assumes base_idx=1
    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=buffer.pos - 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    assert batch.dones.numpy().item() == 0.0

    # Set done=1 manually, the tmp truncation should not be set (it would set batch.done=False)
    buffer.dones[buffer.pos - 1, :] = True
    batch = buffer._get_samples(np.array([base_idx]))
    actual = batch.rewards.numpy().item()
    # Note: compute_expected_nstep assumes base_idx=0
    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=buffer.pos - 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    assert batch.dones.numpy().item() == 1.0


def test_match_normal_buffer():
    buffer = create_buffer(n_steps=1)
    ref_buffer = create_normal_buffer()
    # no done or truncation
    fill_buffer(buffer, length=10)
    fill_buffer(ref_buffer, length=10)

    base_idx = 3
    batch1 = buffer._get_samples(np.array([base_idx]))
    actual1 = batch1.rewards.numpy().item()

    batch2 = ref_buffer._get_samples(np.array([base_idx]))

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=1)
    np.testing.assert_allclose(actual1, expected, rtol=1e-4)
    assert batch1.dones.numpy().item() == 0.0

    np.testing.assert_allclose(batch1.rewards.numpy(), batch2.rewards.numpy(), rtol=1e-4)
