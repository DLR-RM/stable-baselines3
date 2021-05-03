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
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
Dict          ❌     ✔️
============= ====== ===========

Example
-------

Train a PPO agent on ``Pendulum-v0`` using 4 environments.

.. code-block:: python

  import gym

  from stable_baselines3 import PPO
  from stable_baselines3.common.env_util import make_vec_env

  # Parallel environments
  env = make_vec_env("CartPole-v1", n_envs=4)

  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("ppo_cartpole")

  del model # remove to demonstrate saving and loading

  model = PPO.load("ppo_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Results
-------

Atari Games
^^^^^^^^^^^

The complete learning curves are available in the `associated PR #110 <https://github.com/DLR-RM/stable-baselines3/pull/110>`_.


PyBullet Environments
^^^^^^^^^^^^^^^^^^^^^

Results on the PyBullet benchmark (2M steps) using 6 seeds.
The complete learning curves are available in the `associated issue #48 <https://github.com/DLR-RM/stable-baselines3/issues/48>`_.


.. note::

  Hyperparameters from the `gSDE paper <https://arxiv.org/abs/2005.05719>`_ were used (as they are tuned for PyBullet envs).


*Gaussian* means that the unstructured Gaussian noise is used for exploration,
*gSDE* (generalized State-Dependent Exploration) is used otherwise.

+--------------+--------------+--------------+--------------+-------------+
| Environments | A2C          | A2C          | PPO          | PPO         |
+==============+==============+==============+==============+=============+
|              | Gaussian     | gSDE         | Gaussian     | gSDE        |
+--------------+--------------+--------------+--------------+-------------+
| HalfCheetah  | 2003 +/- 54  | 2032 +/- 122 | 1976 +/- 479 | 2826 +/- 45 |
+--------------+--------------+--------------+--------------+-------------+
| Ant          | 2286 +/- 72  | 2443 +/- 89  | 2364 +/- 120 | 2782 +/- 76 |
+--------------+--------------+--------------+--------------+-------------+
| Hopper       | 1627 +/- 158 | 1561 +/- 220 | 1567 +/- 339 | 2512 +/- 21 |
+--------------+--------------+--------------+--------------+-------------+
| Walker2D     | 577 +/- 65   | 839 +/- 56   | 1230 +/- 147 | 2019 +/- 64 |
+--------------+--------------+--------------+--------------+-------------+


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the `rl-zoo repo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo ppo --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results (here for PyBullet envs only):

.. code-block:: bash

  python scripts/all_plots.py -a ppo -e HalfCheetah Ant Hopper Walker2D -f logs/ -o logs/ppo_results
  python scripts/plot_from_file.py -i logs/ppo_results.pkl -latex -l PPO


Parameters
----------

.. autoclass:: PPO
  :members:
  :inherited-members:


PPO Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.common.policies.ActorCriticPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: stable_baselines3.common.policies.ActorCriticCnnPolicy
  :members:
  :noindex:

.. autoclass:: MultiInputPolicy
  :members:

.. autoclass:: stable_baselines3.common.policies.MultiInputActorCriticPolicy
  :members:
  :noindex:
