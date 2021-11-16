import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.envs import IdentityEnv, IdentityEnvBox, IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

DIM = 4


@pytest.mark.parametrize("model_class", [A2C, PPO, DQN])
@pytest.mark.parametrize("env", [IdentityEnv(DIM), IdentityEnvMultiDiscrete(DIM), IdentityEnvMultiBinary(DIM)])
def test_discrete(model_class, env):
    env_ = DummyVecEnv([lambda: env])
    kwargs = {}
    n_steps = 3000
    if model_class == DQN:
        kwargs = dict(learning_starts=0)
        n_steps = 4000
        # DQN only support discrete actions
        if isinstance(env, (IdentityEnvMultiDiscrete, IdentityEnvMultiBinary)):
            return
    elif model_class == A2C:
        # slightly higher budget
        n_steps = 3500

    model = model_class("MlpPolicy", env_, gamma=0.4, seed=1, **kwargs).learn(n_steps)

    evaluate_policy(model, env_, n_eval_episodes=20, reward_threshold=90, warn=False)
    obs = env.reset()

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)


@pytest.mark.parametrize("model_class", [A2C, PPO, SAC, DDPG, TD3])
def test_continuous(model_class):
    env = IdentityEnvBox(eps=0.5)

    n_steps = {A2C: 3500, PPO: 3000, SAC: 700, TD3: 500, DDPG: 500}[model_class]

    kwargs = dict(policy_kwargs=dict(net_arch=[64, 64]), seed=0, gamma=0.95)
    if model_class in [TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise

    model = model_class("MlpPolicy", env, **kwargs).learn(n_steps)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)
