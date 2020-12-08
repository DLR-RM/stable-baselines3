import pytest

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.multi_input_envs import SimpleMultiObsEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


@pytest.mark.parametrize("model_class", [PPO, A2C, DQN, SAC, TD3])
def test_dict_spaces(model_class):
    """
    Additional tests for PPO/A2C/SAC/TD3/DQN to check observation space support
    for Dictionary spaces using MultiInputPolicy.
    """
    use_discrete_actions = model_class not in [SAC, TD3]
    env = DummyVecEnv([lambda: SimpleMultiObsEnv(random_start=True, discrete_actions=use_discrete_actions)])
    env = VecFrameStack(env, n_stack=2)
    kwargs = {}
    n_steps = 250

    if model_class == DQN:
        kwargs = dict(learning_starts=0)

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)
