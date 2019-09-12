import numpy as np


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=False):
    """
    Runs policy for n episodes and returns average reward
    """
    mean_reward, n_steps = 0.0, 0
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            mean_reward += reward
            n_steps += 1
            if render:
                env.render()

    mean_reward /= n_eval_episodes

    return mean_reward, n_steps
