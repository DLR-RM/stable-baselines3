import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.envs.multi_input_envs import SimpleMultiObsEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


@pytest.mark.parametrize("model_class", [PPO, A2C, DQN, DDPG, SAC, TD3])
def test_dict_spaces(model_class):
    """
    Additional tests for PPO/A2C/SAC/DDPG/TD3/DQN to check observation space support
    for Dictionary spaces using MultiInputPolicy.
    """
    use_discrete_actions = model_class not in [SAC, TD3, DDPG]
    env = DummyVecEnv([lambda: SimpleMultiObsEnv(random_start=True, discrete_actions=use_discrete_actions)])
    env = VecFrameStack(env, n_stack=2)
    kwargs = {}
    n_steps = 250

    if model_class in {A2C, PPO}:
        kwargs = dict(n_steps=100)
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features
        kwargs = dict(
            buffer_size=250,
            policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32)),
        )
        if model_class == DQN:
            kwargs["learning_starts"] = 0

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)
