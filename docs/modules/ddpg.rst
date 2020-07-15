.. _ddpg:

.. automodule:: stable_baselines3.ddpg


DDPG
====

`Deep Deterministic Policy Gradient (DDPG) <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_ combines the
trick for DQN with the deterministic policy gradient, to obtain an algorithm for continuous actions.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy


Notes
-----

- Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
- DDPG Paper: https://arxiv.org/abs/1509.02971
- OpenAI Spinning Guide for DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

.. note::

    The default policy for DDPG uses a ReLU activation, to match the original paper, whereas most other algorithms' MlpPolicy uses a tanh activation.
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


.. .. autoclass:: CnnPolicy
..   :members:
..   :inherited-members:
