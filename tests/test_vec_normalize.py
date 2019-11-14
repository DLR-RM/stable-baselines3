import gym
import pytest
import numpy as np

from torchy_baselines.common.running_mean_std import RunningMeanStd
from torchy_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from torchy_baselines.common.vec_env.vec_normalize import VecNormalize
from torchy_baselines import CEMRL, SAC, TD3

ENV_ID = 'Pendulum-v0'


def test_runningmeanstd():
    """Test RunningMeanStd object"""
    for (x_1, x_2, x_3) in [
         (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
         (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2))]:
        rms = RunningMeanStd(epsilon=0.0, shape=x_1.shape[1:])

        x_cat = np.concatenate([x_1, x_2, x_3], axis=0)
        moments_1 = [x_cat.mean(axis=0), x_cat.var(axis=0)]
        rms.update(x_1)
        rms.update(x_2)
        rms.update(x_3)
        moments_2 = [rms.mean, rms.var]

        assert np.allclose(moments_1, moments_2)


def test_vec_env():
    """Test VecNormalize Object"""

    def make_env():
        return gym.make(ENV_ID)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    _, done = env.reset(), [False]
    obs = None
    while not done[0]:
        actions = [env.action_space.sample()]
        obs, _, done, _ = env.step(actions)
    assert np.max(obs) <= 10


@pytest.mark.parametrize("model_class", [SAC, TD3, CEMRL])
def test_offpolicy_normalization(model_class):
    env = DummyVecEnv([lambda: gym.make(ENV_ID)])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10., clip_reward=10.)

    eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., clip_reward=10.)

    model = model_class('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000, eval_env=eval_env, eval_freq=1000)
