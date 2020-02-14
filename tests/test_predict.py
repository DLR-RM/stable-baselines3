import gym
import pytest

from torchy_baselines import A2C, CEMRL, PPO, SAC, TD3
from torchy_baselines.common.vec_env import DummyVecEnv

MODEL_LIST = [
    CEMRL,
    PPO,
    A2C,
    TD3,
    SAC,
]

@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_auto_wrap(model_class):
    # test auto wrapping of env into a VecEnv
    env = gym.make('Pendulum-v0')
    eval_env = gym.make('Pendulum-v0')
    model = model_class('MlpPolicy', env)
    model.learn(100, eval_env=eval_env)


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_predict(model_class):
    # test detection of different shapes by the predict method
    model = model_class('MlpPolicy', 'Pendulum-v0')
    env = gym.make('Pendulum-v0')
    vec_env = DummyVecEnv([lambda: gym.make('Pendulum-v0'), lambda: gym.make('Pendulum-v0')])

    obs = env.reset()
    action = model.predict(obs)
    assert action.shape == env.action_space.shape
    assert env.action_space.contains(action)

    vec_env_obs = vec_env.reset()
    action = model.predict(vec_env_obs)
    assert action.shape[0] == vec_env_obs.shape[0]


@pytest.mark.parametrize("model_class", [A2C, PPO])
def test_predict_discrete(model_class):
    # test detection of different shapes by the predict method
    model = model_class('MlpPolicy', 'CartPole-v1')
    env = gym.make('CartPole-v1')
    vec_env = DummyVecEnv([lambda: gym.make('CartPole-v1'), lambda: gym.make('CartPole-v1')])

    obs = env.reset()
    action = model.predict(obs)
    assert action.shape == ()
    assert env.action_space.contains(action)

    vec_env_obs = vec_env.reset()
    action = model.predict(vec_env_obs)
    assert action.shape[0] == vec_env_obs.shape[0]
