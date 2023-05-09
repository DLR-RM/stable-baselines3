.. _sac:

.. automodule:: stable_baselines3.sac


SAC
===

`Soft Actor Critic (SAC) <https://spinningup.openai.com/en/latest/algorithms/sac.html>`_ Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

SAC is the successor of `Soft Q-Learning SQL <https://arxiv.org/abs/1702.08165>`_ and incorporates the double Q-learning trick from TD3.
A key feature of SAC, and a major difference with common RL algorithms, is that it is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/1801.01290
- OpenAI Spinning Guide for SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
- Original Implementation: https://github.com/haarnoja/sac
- Blog post on using SAC with real robots: https://bair.berkeley.edu/blog/2018/12/14/sac/

.. note::
    In our implementation, we use an entropy coefficient (as in OpenAI Spinning or Facebook Horizon),
    which is the equivalent to the inverse of reward scale in the original SAC paper.
    The main reason is that it avoids having too high errors when updating the Q functions.


.. note::

    The default policies for SAC differ a bit from others MlpPolicy: it uses ReLU instead of tanh activation,
    to match the original paper


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

  from stable_baselines3 import SAC

  env = gym.make("Pendulum-v1", render_mode="human")

  model = SAC("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("sac_pendulum")

  del model # remove to demonstrate saving and loading

  model = SAC.load("sac_pendulum")

  obs, info = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      if terminated or truncated:
          obs, info = env.reset()


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

  python train.py --algo sac --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a sac -e HalfCheetah Ant Hopper Walker2D -f logs/ -o logs/sac_results
  python scripts/plot_from_file.py -i logs/sac_results.pkl -latex -l SAC


Parameters
----------

.. autoclass:: SAC
  :members:
  :inherited-members:

.. _sac_policies:

SAC Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.sac.policies.SACPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: MultiInputPolicy
  :members:
