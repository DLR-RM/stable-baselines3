.. _sac:

.. automodule:: stable_baselines3.sac


SAC
===

`Soft Actor Critic (SAC) <https://spinningup.openai.com/en/latest/algorithms/sac.html>`_ Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

SAC is the successor of `Soft Q-Learning SQL <https://arxiv.org/abs/1702.08165>`_ and incorporates the double Q-learning trick from TD3.
A key feature of SAC, and a major difference with common RL algorithms, is that it is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy.


.. warning::

  The SAC model does not support ``stable_baselines3.ppo.policies`` because it uses double q-values
  and value estimation, as a result it must use its own policy models (see :ref:`sac_policies`).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy


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
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ❌
Box           ✔️       ✔️
MultiDiscrete ❌      ❌
MultiBinary   ❌      ❌
============= ====== ===========


Example
-------

.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines3 import SAC
  from stable_baselines3.sac import MlpPolicy

  env = gym.make('Pendulum-v0')

  model = SAC(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("sac_pendulum")

  del model # remove to demonstrate saving and loading

  model = SAC.load("sac_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()

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

.. .. autoclass:: CnnPolicy
..   :members:
..   :inherited-members:
