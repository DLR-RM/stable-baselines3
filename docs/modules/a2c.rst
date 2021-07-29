.. _a2c:

.. automodule:: stable_baselines3.a2c


A2C
====

A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
It uses multiple workers to avoid the use of a replay buffer.


.. warning::

  If you find training unstable or want to match performance of stable-baselines A2C, consider using
  ``RMSpropTFLike`` optimizer from ``stable_baselines3.common.sb2_compat.rmsprop_tf_like``.
  You can change optimizer with ``A2C(policy_kwargs=dict(optimizer_class=RMSpropTFLike, eps=1e-5))``.
  Read more `here <https://github.com/DLR-RM/stable-baselines3/pull/110#issuecomment-663255241>`_.


Notes
-----

-  Original paper:  https://arxiv.org/abs/1602.01783
-  OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/


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

Train a A2C agent on ``CartPole-v1`` using 4 environments.

.. code-block:: python

  import gym

  from stable_baselines3 import A2C
  from stable_baselines3.common.env_util import make_vec_env

  # Parallel environments
  env = make_vec_env("CartPole-v1", n_envs=4)

  model = A2C("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("a2c_cartpole")

  del model # remove to demonstrate saving and loading

  model = A2C.load("a2c_cartpole")

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

  python train.py --algo a2c --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results (here for PyBullet envs only):

.. code-block:: bash

  python scripts/all_plots.py -a a2c -e HalfCheetah Ant Hopper Walker2D -f logs/ -o logs/a2c_results
  python scripts/plot_from_file.py -i logs/a2c_results.pkl -latex -l A2C


Parameters
----------

.. autoclass:: A2C
  :members:
  :inherited-members:


A2C Policies
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
