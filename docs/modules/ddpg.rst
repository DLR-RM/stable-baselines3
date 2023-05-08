.. _ddpg:

.. automodule:: stable_baselines3.ddpg


DDPG
====

`Deep Deterministic Policy Gradient (DDPG) <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_ combines the
trick for DQN with the deterministic policy gradient, to obtain an algorithm for continuous actions.


.. note::

  As ``DDPG`` can be seen as a special case of its successor :ref:`TD3 <td3>`,
  they share the same policies and same implementation.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy


Notes
-----

- Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
- DDPG Paper: https://arxiv.org/abs/1509.02971
- OpenAI Spinning Guide for DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html



Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
Dict          ❌     ✔️
============= ====== ===========


Example
-------

This example is only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized hyperparameters can be found in RL Zoo `repository <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

.. code-block:: python

  import gymnasium as gym
  import numpy as np

  from stable_baselines3 import DDPG
  from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

  env = gym.make("Pendulum-v1", render_mode="rgb_array")

  # The noise objects for DDPG
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

  model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
  model.learn(total_timesteps=10000, log_interval=10)
  model.save("ddpg_pendulum")
  vec_env = model.get_env()

  del model # remove to demonstrate saving and loading

  model = DDPG.load("ddpg_pendulum")

  obs = vec_env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = vec_env.step(action)
      env.render("human")

Results
-------

PyBullet Environments
^^^^^^^^^^^^^^^^^^^^^

Results on the PyBullet benchmark (1M steps) using 6 seeds.
The complete learning curves are available in the `associated issue #48 <https://github.com/DLR-RM/stable-baselines3/issues/48>`_.


.. note::

  Hyperparameters of :ref:`TD3 <td3>` from the `gSDE paper <https://arxiv.org/abs/2005.05719>`_ were used for ``DDPG``.


*Gaussian* means that the unstructured Gaussian noise is used for exploration,
*gSDE* (generalized State-Dependent Exploration) is used otherwise.

+--------------+--------------+--------------+--------------+
| Environments | DDPG         | TD3          | SAC          |
+==============+==============+==============+==============+
|              | Gaussian     | Gaussian     | gSDE         |
+--------------+--------------+--------------+--------------+
| HalfCheetah  | 2272 +/- 69  | 2774 +/- 35  | 2984 +/- 202 |
+--------------+--------------+--------------+--------------+
| Ant          | 1651 +/- 407 | 3305 +/- 43  | 3102 +/- 37  |
+--------------+--------------+--------------+--------------+
| Hopper       | 1201 +/- 211 | 2429 +/- 126 | 2262 +/- 1   |
+--------------+--------------+--------------+--------------+
| Walker2D     | 882 +/- 186  | 2063 +/- 185 | 2136 +/- 67  |
+--------------+--------------+--------------+--------------+



How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the `rl-zoo repo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo ddpg --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a ddpg -e HalfCheetah Ant Hopper Walker2D -f logs/ -o logs/ddpg_results
  python scripts/plot_from_file.py -i logs/ddpg_results.pkl -latex -l DDPG



Parameters
----------

.. autoclass:: DDPG
  :members:
  :inherited-members:

.. _ddpg_policies:

DDPG Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.td3.policies.TD3Policy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:
  :noindex:

.. autoclass:: MultiInputPolicy
  :members:
  :noindex:
