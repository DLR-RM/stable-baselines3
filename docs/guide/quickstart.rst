.. _quickstart:

===============
Getting Started
===============

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run SAC on a Pendulum environment:

.. code-block:: python

  import gym

  from torchy_baselines.sac.policies import MlpPolicy
  from torchy_baselines.common.vec_env import DummyVecEnv
  from torchy_baselines import SAC

  env = gym.make('Pendulum-v0')

  model = SAC(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=10000)

  obs = env.reset()
  for i in range(1000):
      action = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Or just train a model with a one liner if
`the environment is registered in Gym <https://github.com/openai/gym/wiki/Environments>`_ and if
the policy is registered:

.. code-block:: python

    from torchy_baselines import SAC

    model = SAC('MlpPolicy', 'Pendulum-v0').learn(10000)
