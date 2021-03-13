import gym
import numpy as np
import pytest
from gym import spaces

from stable_baselines3 import SAC, TD3, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
    sync_envs_normalization,
    unwrap_vec_normalize,
)

ENV_ID = "Pendulum-v0"


class DummyDictEnv(gym.GoalEnv):
    """
    Dummy gym goal env for testing purposes
    """

    def __init__(self):
        super(DummyDictEnv, self).__init__()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32),
                "achieved_goal": spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        obs = self.observation_space.sample()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        done = np.random.rand() > 0.8
        return obs, reward, done, {}

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info) -> np.float32:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0).astype(np.float32)


def allclose(obs_1, obs_2):
    """
    Generalized np.allclose() to work with dict spaces.
    """
    if isinstance(obs_1, dict):
        all_close = True
        for key in obs_1.keys():
            if not np.allclose(obs_1[key], obs_2[key]):
                all_close = False
                break
        return all_close
    return np.allclose(obs_1, obs_2)


def make_env():
    return Monitor(gym.make(ENV_ID))


def make_dict_env():
    return Monitor(DummyDictEnv())


def check_rms_equal(rmsa, rmsb):
    if isinstance(rmsa, dict):
        for key in rmsa.keys():
            assert np.all(rmsa[key].mean == rmsb[key].mean)
            assert np.all(rmsa[key].var == rmsb[key].var)
            assert np.all(rmsa[key].count == rmsb[key].count)
    else:
        assert np.all(rmsa.mean == rmsb.mean)
        assert np.all(rmsa.var == rmsb.var)
        assert np.all(rmsa.count == rmsb.count)


def check_vec_norm_equal(norma, normb):
    assert norma.observation_space == normb.observation_space
    assert norma.action_space == normb.action_space
    assert norma.num_envs == normb.num_envs

    check_rms_equal(norma.obs_rms, normb.obs_rms)
    check_rms_equal(norma.ret_rms, normb.ret_rms)
    assert norma.clip_obs == normb.clip_obs
    assert norma.clip_reward == normb.clip_reward
    assert norma.norm_obs == normb.norm_obs
    assert norma.norm_reward == normb.norm_reward

    assert np.all(norma.ret == normb.ret)
    assert norma.gamma == normb.gamma
    assert norma.epsilon == normb.epsilon
    assert norma.training == normb.training


def _make_warmstart_cartpole():
    """Warm-start VecNormalize by stepping through CartPole"""
    venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    venv = VecNormalize(venv)
    venv.reset()
    venv.get_original_obs()

    for _ in range(100):
        actions = [venv.action_space.sample()]
        venv.step(actions)
    return venv


def _make_warmstart_dict_env():
    """Warm-start VecNormalize by stepping through BitFlippingEnv"""
    venv = DummyVecEnv([make_dict_env])
    venv = VecNormalize(venv)
    venv.reset()
    venv.get_original_obs()

    for _ in range(100):
        actions = [venv.action_space.sample()]
        venv.step(actions)
    return venv


def test_runningmeanstd():
    """Test RunningMeanStd object"""
    for (x_1, x_2, x_3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2)),
    ]:
        rms = RunningMeanStd(epsilon=0.0, shape=x_1.shape[1:])

        x_cat = np.concatenate([x_1, x_2, x_3], axis=0)
        moments_1 = [x_cat.mean(axis=0), x_cat.var(axis=0)]
        rms.update(x_1)
        rms.update(x_2)
        rms.update(x_3)
        moments_2 = [rms.mean, rms.var]

        assert np.allclose(moments_1, moments_2)


@pytest.mark.parametrize("make_env", [make_env, make_dict_env])
def test_vec_env(tmp_path, make_env):
    """Test VecNormalize Object"""
    clip_obs = 0.5
    clip_reward = 5.0

    orig_venv = DummyVecEnv([make_env])
    norm_venv = VecNormalize(orig_venv, norm_obs=True, norm_reward=True, clip_obs=clip_obs, clip_reward=clip_reward)
    _, done = norm_venv.reset(), [False]
    while not done[0]:
        actions = [norm_venv.action_space.sample()]
        obs, rew, done, _ = norm_venv.step(actions)
        if isinstance(obs, dict):
            for key in obs.keys():
                assert np.max(np.abs(obs[key])) <= clip_obs
        else:
            assert np.max(np.abs(obs)) <= clip_obs
        assert np.max(np.abs(rew)) <= clip_reward

    path = tmp_path / "vec_normalize"
    norm_venv.save(path)
    deserialized = VecNormalize.load(path, venv=orig_venv)
    check_vec_norm_equal(norm_venv, deserialized)


def test_get_original():
    venv = _make_warmstart_cartpole()
    for _ in range(3):
        actions = [venv.action_space.sample()]
        obs, rewards, _, _ = venv.step(actions)
        obs = obs[0]
        orig_obs = venv.get_original_obs()[0]
        rewards = rewards[0]
        orig_rewards = venv.get_original_reward()[0]

        assert np.all(orig_rewards == 1)
        assert orig_obs.shape == obs.shape
        assert orig_rewards.dtype == rewards.dtype
        assert not np.array_equal(orig_obs, obs)
        assert not np.array_equal(orig_rewards, rewards)
        np.testing.assert_allclose(venv.normalize_obs(orig_obs), obs)
        np.testing.assert_allclose(venv.normalize_reward(orig_rewards), rewards)


def test_get_original_dict():
    venv = _make_warmstart_dict_env()
    for _ in range(3):
        actions = [venv.action_space.sample()]
        obs, rewards, _, _ = venv.step(actions)
        # obs = obs[0]
        orig_obs = venv.get_original_obs()
        rewards = rewards[0]
        orig_rewards = venv.get_original_reward()[0]

        for key in orig_obs.keys():
            assert orig_obs[key].shape == obs[key].shape
        assert orig_rewards.dtype == rewards.dtype

        assert not allclose(orig_obs, obs)
        assert not np.array_equal(orig_rewards, rewards)
        assert allclose(venv.normalize_obs(orig_obs), obs)
        np.testing.assert_allclose(venv.normalize_reward(orig_rewards), rewards)


def test_normalize_external():
    venv = _make_warmstart_cartpole()

    rewards = np.array([1, 1])
    norm_rewards = venv.normalize_reward(rewards)
    assert norm_rewards.shape == rewards.shape
    # Episode return is almost always >= 1 in CartPole. So reward should shrink.
    assert np.all(norm_rewards < 1)


@pytest.mark.parametrize("model_class", [SAC, TD3, HerReplayBuffer])
@pytest.mark.parametrize("online_sampling", [False, True])
def test_offpolicy_normalization(model_class, online_sampling):

    if online_sampling and model_class != HerReplayBuffer:
        pytest.skip()

    make_env_ = make_dict_env if model_class == HerReplayBuffer else make_env
    env = DummyVecEnv([make_env_])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env_])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    if model_class == HerReplayBuffer:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_starts=100,
            policy_kwargs=dict(net_arch=[64]),
            replay_buffer_kwargs=dict(
                max_episode_length=100,
                online_sampling=online_sampling,
                n_sampled_goal=2,
            ),
            replay_buffer_class=HerReplayBuffer,
            seed=2,
        )
    else:
        model = model_class("MlpPolicy", env, verbose=1, learning_starts=100, policy_kwargs=dict(net_arch=[64]))

    model.learn(total_timesteps=150, eval_env=eval_env, eval_freq=75)
    # Check getter
    assert isinstance(model.get_vec_normalize_env(), VecNormalize)


@pytest.mark.parametrize("make_env", [make_env, make_dict_env])
def test_sync_vec_normalize(make_env):
    env = DummyVecEnv([make_env])

    assert unwrap_vec_normalize(env) is None

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=100.0)

    assert isinstance(unwrap_vec_normalize(env), VecNormalize)

    if not isinstance(env.observation_space, spaces.Dict):
        env = VecFrameStack(env, 1)
        assert isinstance(unwrap_vec_normalize(env), VecNormalize)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=100.0)

    if not isinstance(env.observation_space, spaces.Dict):
        eval_env = VecFrameStack(eval_env, 1)

    env.seed(0)
    env.action_space.seed(0)

    env.reset()
    # Initialize running mean
    latest_reward = None
    for _ in range(100):
        _, latest_reward, _, _ = env.step([env.action_space.sample()])

    # Check that unnormalized reward is same as original reward
    original_latest_reward = env.get_original_reward()
    assert np.allclose(original_latest_reward, env.unnormalize_reward(latest_reward))

    obs = env.reset()
    dummy_rewards = np.random.rand(10)
    original_obs = env.get_original_obs()
    # Check that unnormalization works
    assert allclose(original_obs, env.unnormalize_obs(obs))
    # Normalization must be different (between different environments)
    assert not allclose(obs, eval_env.normalize_obs(original_obs))

    # Test syncing of parameters
    sync_envs_normalization(env, eval_env)
    # Now they must be synced
    assert allclose(obs, eval_env.normalize_obs(original_obs))
    assert allclose(env.normalize_reward(dummy_rewards), eval_env.normalize_reward(dummy_rewards))
