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
@pytest.mark.parametrize("env_id", ['Pendulum-v0', 'CartPole-v1'])
def test_predict(model_class, env_id):
    if env_id == 'CartPole-v1' and model_class not in [PPO, A2C]:
        return

    # test detection of different shapes by the predict method
    model = model_class('MlpPolicy', env_id)
    env = gym.make(env_id)
    vec_env = DummyVecEnv([lambda: gym.make(env_id), lambda: gym.make(env_id)])

    obs = env.reset()
    action = model.predict(obs)
    assert action.shape == env.action_space.shape
    assert env.action_space.contains(action)

    vec_env_obs = vec_env.reset()
    action = model.predict(vec_env_obs)
    assert action.shape[0] == vec_env_obs.shape[0]
