import gym
import numpy as np
import pytest

from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.multi_input_envs import NineRoomMultiObsEnv, SimpleMultiObsEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


@pytest.mark.slow
@pytest.mark.parametrize("model_class", [PPO])  # , SAC, TD3, DQN])
def test_dict_spaces(model_class):
    """
    Additional tests for PPO/SAC/TD3/DQN to check observation space support
    for Dictionary spaces using MultiInputPolicy.
    """
    make_env = lambda: SimpleMultiObsEnv(random_start=True)
    env = DummyVecEnv([make_env])
    # env = VecFrameStack(env, n_stack=2)

    model = model_class(
        "MultiInputPolicy",
        env,
        gamma=0.5,
        seed=1,
        policy_kwargs=dict(net_arch=[64]),
    )
    model.learn(total_timesteps=500)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)
