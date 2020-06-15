import gym
import pytest
from stable_baselines3 import create_her, SAC, TD3
import highway_env


@pytest.mark.parametrize('model_class', [SAC, TD3])
def test_her(model_class):
    HER = create_her(model_class)
    model = HER('MlpPolicy', 'parking-v0', n_sampled_goal=4,
                goal_selection_strategy='future', policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=100, verbose=1, create_eval_env=True)
    model.learn(total_timesteps=10, eval_freq=100)
    model.save('her_highway')

    env = gym.make('parking-v0')
    model = HER.load('her_highway', env=env)
    obs = env.reset()
    episode_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done or info.get('is_success', False):
            print("Reward:", episode_reward, "Success?", info.get('is_success', False))
            episode_reward = 0.0
            obs = env.reset()
