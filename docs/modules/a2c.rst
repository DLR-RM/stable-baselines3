.. _a2c:

.. automodule:: stable_baselines3.a2c


A2C
====

A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
It uses multiple workers to avoid the use of a replay buffer.


Notes
-----

-  Original paper:  https://arxiv.org/abs/1602.01783
-  OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/


Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ❌
Box           ✔️      ✔️
MultiDiscrete ❌     ❌
MultiBinary   ❌      ❌
============= ====== ===========


Example
-------

Train a A2C agent on ``CartPole-v1`` using 4 environments.

.. code-block:: python

  import gym

  from stable_baselines3 import A2C
  from stable_baselines3.a2c import MlpPolicy
  from stable_baselines3.common.cmd_utils import make_vec_env

  # Parallel environments
  env = make_vec_env('CartPole-v1', n_envs=4)

  model = A2C(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("a2c_cartpole")

  del model # remove to demonstrate saving and loading

  model = A2C.load("a2c_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: A2C
  :members:
  :inherited-members:
