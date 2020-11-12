from typing import Callable, List, Optional, Tuple, Union
import warnings

import gym
import numpy as np

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import base_class


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    If environment has not been wrapped with ``Monitor`` wrapper, reward and
    episode lengths are counted as it appears with ``env.step`` calls. However
    if the environment contains wrappers that modify rewards or episode lengths
    (e.g. reward scaling, early episode reset), these will affect the evaluation
    results as well.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()

        # Remove VecEnv stacking (if any)
        if isinstance(env, VecEnv):
            info = info[0]

        if "episode" in info.keys():
            # Monitor wrapper includes "episode" key in info if environment
            # has been wrapped with it. Use those rewards instead.
            episode_rewards.append(info["episode"]["r"])
            episode_lengths.append(info["episode"]["l"])
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            warnings.warn(
                "Evaluation environment does not provide 'episode' environment (not wrapped with ``Monitor`` wrapper?). "
                "This may result in reporting modified episode lengths and results, depending on the other wrappers. "
                "Consider wrapping environment first with ``Monitor`` wrapper.",
                UserWarning
            )
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
