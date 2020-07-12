import gym
import pytest
from stable_baselines3 import create_her, SAC, TD3
import highway_env


@pytest.mark.parametrize('model_class', [SAC, TD3])
def test_her_model(model_class):
    HER = create_her(model_class)
    model = HER('MlpPolicy', 'parking-v0', n_sampled_goal=4,
                goal_selection_strategy='future', policy_kwargs=dict(net_arch=[256, 256, 256]),
                learning_starts=100, verbose=1, create_eval_env=True, gamma=0.95, learning_rate=1e-3,
                add_her_while_sampling=False)
    model.learn(total_timesteps=1000, eval_freq=100)
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


@pytest.mark.parametrize('add_her_while_sampling', [False, True])
@pytest.mark.parametrize('goal_selection_strategy', ['future', 'episode', 'final', 'random'])
def test_her_sampling(add_her_while_sampling, goal_selection_strategy):
    HER = create_her(SAC)
    model = HER('MlpPolicy', 'parking-v0', n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy, policy_kwargs=dict(net_arch=[256, 256, 256]),
                learning_starts=100, verbose=1, create_eval_env=True, gamma=0.95, learning_rate=1e-3,
                add_her_while_sampling=add_her_while_sampling)
    model.learn(total_timesteps=1000, eval_freq=100)
