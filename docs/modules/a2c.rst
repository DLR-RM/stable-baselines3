.. _a2c:

.. automodule:: torchy_baselines.a2c


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

Train a A2C agent on `CartPole-v1` using 4 processes.

.. code-block:: python

  import gym

  from torchy_baselines.common.policies import MlpPolicy
  from torchy_baselines.common import make_vec_env
  from torchy_baselines import A2C

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
