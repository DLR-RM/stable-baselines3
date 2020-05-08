.. _ppo2:

.. automodule:: stable_baselines3.ppo

PPO
===

The `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far form the old policy.
For that, ppo uses clipping to avoid too large update.


.. note::

  PPO contains several modifications from the original algorithm not documented
  by OpenAI: advantages are normalized and value function can be also clipped .


Notes
-----

- Original paper: https://arxiv.org/abs/1707.06347
- Clear explanation of PPO on Arxiv Insights channel: https://www.youtube.com/watch?v=5P7I-xPq8u8
- OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
- Spinning Up guide: https://spinningup.openai.com/en/latest/algorithms/ppo.html


Can I use?
----------

-  Recurrent policies: ❌
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

Train a PPO agent on ``Pendulum-v0`` using 4 environments.

.. code-block:: python

  import gym

  from stable_baselines3 import A2C
  from stable_baselines3.ppo import MlpPolicy
  from stable_baselines3.common.cmd_utils import make_vec_env

  # Parallel environments
  env = make_vec_env('CartPole-v1', n_envs=4)

  model = PPO(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("ppo_cartpole")

  del model # remove to demonstrate saving and loading

  model = PPO.load("ppo_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: PPO
  :members:
  :inherited-members:
