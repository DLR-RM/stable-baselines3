.. _dqn:

.. automodule:: stable_baselines3.dqn


DQN
===

`Deep Q Network (DQN) <https://arxiv.org/abs/1312.5602>`_

.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/1312.5602
- Further reference: https://www.nature.com/articles/nature14236

.. note::
    This implementation provides only vanilla Deep Q-Learning and has no extensions such as Double-DQN, Dueling-DQN and Prioritized Experience Replay.


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔      ✔
Box           ❌      ✔
MultiDiscrete ❌      ✔
MultiBinary   ❌      ✔
============= ====== ===========


Example
-------

.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines3 import DQN
  from stable_baselines3.dqn import MlpPolicy

  env = gym.make('Pendulum-v0')

  model = DQN(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("dqn_pendulum")

  del model # remove to demonstrate saving and loading

  model = DQN.load("dqn_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()

Parameters
----------

.. autoclass:: DQN
  :members:
  :inherited-members:

.. _dqn_policies:

DQN Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.dqn.policies.DQNPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:
