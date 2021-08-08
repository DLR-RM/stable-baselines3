.. _td3:

.. automodule:: stable_baselines3.td3


TD3
===

`Twin Delayed DDPG (TD3) <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ Addressing Function Approximation Error in Actor-Critic Methods.

TD3 is a direct successor of :ref:`DDPG <ddpg>` and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing.
We recommend reading `OpenAI Spinning guide on TD3 <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ to learn more about those.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy


Notes
-----

- Original paper: https://arxiv.org/pdf/1802.09477.pdf
- OpenAI Spinning Guide for TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
- Original Implementation: https://github.com/sfujim/TD3

.. note::

    The default policies for TD3 differ a bit from others MlpPolicy: it uses ReLU instead of tanh activation,
    to match the original paper


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
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

  import gym
  import numpy as np

  from stable_baselines3 import TD3
  from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

  env = gym.make("Pendulum-v0")

  # The noise objects for TD3
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

  model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
  model.learn(total_timesteps=10000, log_interval=10)
  model.save("td3_pendulum")
  env = model.get_env()

  del model # remove to demonstrate saving and loading

  model = TD3.load("td3_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Results
-------

PyBullet Environments
^^^^^^^^^^^^^^^^^^^^^

Results on the PyBullet benchmark (1M steps) using 3 seeds.
The complete learning curves are available in the `associated issue #48 <https://github.com/DLR-RM/stable-baselines3/issues/48>`_.


.. note::

  Hyperparameters from the `gSDE paper <https://arxiv.org/abs/2005.05719>`_ were used (as they are tuned for PyBullet envs).


*Gaussian* means that the unstructured Gaussian noise is used for exploration,
*gSDE* (generalized State-Dependent Exploration) is used otherwise.

+--------------+--------------+--------------+--------------+
| Environments | SAC          | SAC          | TD3          |
+==============+==============+==============+==============+
|              | Gaussian     | gSDE         | Gaussian     |
+--------------+--------------+--------------+--------------+
| HalfCheetah  | 2757 +/- 53  | 2984 +/- 202 | 2774 +/- 35  |
+--------------+--------------+--------------+--------------+
| Ant          | 3146 +/- 35  | 3102 +/- 37  | 3305 +/- 43  |
+--------------+--------------+--------------+--------------+
| Hopper       | 2422 +/- 168 | 2262 +/- 1   | 2429 +/- 126 |
+--------------+--------------+--------------+--------------+
| Walker2D     | 2184 +/- 54  | 2136 +/- 67  | 2063 +/- 185 |
+--------------+--------------+--------------+--------------+


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the `rl-zoo repo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo td3 --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a td3 -e HalfCheetah Ant Hopper Walker2D -f logs/ -o logs/td3_results
  python scripts/plot_from_file.py -i logs/td3_results.pkl -latex -l TD3


Parameters
----------

.. autoclass:: TD3
  :members:
  :inherited-members:

.. _td3_policies:

TD3 Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.td3.policies.TD3Policy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: MultiInputPolicy
  :members:
