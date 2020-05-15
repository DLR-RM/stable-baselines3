import numpy as np
import pytest

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.identity_env import IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


MODEL_LIST = [A2C, PPO]
DIM = 3


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_identity_multidiscrete(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space
    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiDiscrete(DIM)])

    model = model_class("MlpPolicy", env, gamma=0.5, seed=1)
    model.learn(total_timesteps=3000)
    obs = env.reset()

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_identity_multibinary(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multibinary action space
    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiBinary(DIM)])

    model = model_class("MlpPolicy", env, gamma=0.5, seed=1)
    model.learn(total_timesteps=3000)
    obs = env.reset()

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)
