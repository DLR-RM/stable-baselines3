.. _quickstart:

===============
Getting Started
===============

.. note::

  Stable-Baselines3 (SB3) uses :ref:`vectorized environments (VecEnv) <vec_env>` internally.
  Please read the associated section to learn more about its features and differences compared to a single Gym environment.


Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run A2C on a CartPole environment:

.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import A2C

  env = gym.make("CartPole-v1", render_mode="rgb_array")

  model = A2C("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10_000)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      vec_env.render("human")
      # VecEnv resets automatically
      # if done:
      #   obs = vec_env.reset()

.. note::

	You can find explanations about the logger output and names in the :ref:`Logger <logger>` section.


Or just train a model with a one line if
`the environment is registered in Gymnasium <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs>`_ and if
the policy is registered:

.. code-block:: python

    from stable_baselines3 import A2C

    model = A2C("MlpPolicy", "CartPole-v1").learn(10000)
