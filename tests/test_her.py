import numpy as np
import pytest
import torch as th

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her import HER, GoalSelectionStrategy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import MlpPolicy, TD3Policy


@pytest.mark.parametrize(
    "model_class, policy, sde_support", [(SAC, SACPolicy, True), (TD3, TD3Policy, False), (DDPG, MlpPolicy, False)]
)
@pytest.mark.parametrize("online_sampling", [True, False])
def test_her(model_class, policy, sde_support, online_sampling):
    """
    Test Hindsight Experience Replay.
    """

    env = BitFlippingEnv(n_bits=4, continuous=True)
    env = DummyVecEnv([lambda: env])

    # Create action noise
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions,), 0.2 * np.ones((n_actions,)))

    model = HER(
        policy,
        env,
        model_class,
        n_goals=5,
        goal_strategy="future",
        online_sampling=online_sampling,
        action_noise=action_noise,
        verbose=1,
        tau=0.05,
        batch_size=128,
        learning_rate=0.001,
        policy_kwargs=dict(net_arch=[256]),
        buffer_size=int(1e6),
        gamma=0.98,
        gradient_steps=40,
        sde_support=sde_support,
    )

    model.learn(total_timesteps=500, callback=None)

    # Evaluate the agent
    n_eval_episodes = 5
    n_episodes = 0
    episode_rewards = []
    episode_reward = 0.0

    eval_env = BitFlippingEnv(n_bits=4, continuous=True)

    observation = eval_env.reset()

    while n_episodes < n_eval_episodes:

        obs = np.concatenate([observation["observation"], observation["desired_goal"]])

        with th.no_grad():
            obs_ = th.FloatTensor(np.array(obs).reshape(1, -1)).to(model.model.device)
            action = model.model.policy.predict(obs_)[0][0]

        observation, reward, done, _ = eval_env.step(action)

        # Render the env
        # eval_env.render()

        episode_reward += reward

        if done:
            n_episodes += 1
            observation = eval_env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

    eval_env.close()


@pytest.mark.parametrize(
    "goal_strategy",
    [
        "final",
        "episode",
        "future",
        "random",
        GoalSelectionStrategy.FUTURE,
        GoalSelectionStrategy.RANDOM,
        GoalSelectionStrategy.EPISODE,
        GoalSelectionStrategy.FINAL,
    ],
)
@pytest.mark.parametrize("online_sampling", [True, False])
def test_goal_strategy(goal_strategy, online_sampling):
    """
    Test different goal strategies.
    """
    env = BitFlippingEnv(continuous=True)
    env = DummyVecEnv([lambda: env])

    model = HER(SACPolicy, env, SAC, goal_strategy=goal_strategy, online_sampling=online_sampling)
    model.learn(total_timesteps=200, callback=None)
