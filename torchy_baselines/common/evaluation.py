import numpy as np


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=False):
    """
    Runs policy for n episodes and returns average reward
    """
    episode_rewards, n_steps = [], 0
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            n_steps += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards), n_steps
