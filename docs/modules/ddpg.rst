.. _ddpg:

.. automodule:: stable_baselines3.ddpg


DDPG
====

`Deep Deterministic Policy Gradient (DDPG) <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_ combines the
trick for DQN with the deterministic policy gradient, to obtain an algorithm for continuous actions.


.. note::

  As ``DDPG`` can be seen as a special case of its successor :ref:`TD3 <td3>`,
  they share the same policies.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy


Notes
-----

- Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
- DDPG Paper: https://arxiv.org/abs/1509.02971
- OpenAI Spinning Guide for DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html



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
============= ====== ===========


Example
-------

.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines3 import DDPG
  from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

  env = gym.make('Pendulum-v0')

  # The noise objects for DDPG
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

  model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
  model.learn(total_timesteps=10000, log_interval=10)
  model.save("ddpg_pendulum")
  env = model.get_env()

  del model # remove to demonstrate saving and loading

  model = DDPG.load("ddpg_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Results
-------

As ``DDPG`` is currently treated as a special case of :ref:`TD3 <td3>`,
this implementation can be trusted as TD3 results are macthing the one from the original implementation.


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
