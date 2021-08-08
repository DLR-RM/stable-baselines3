.. _dqn:

.. automodule:: stable_baselines3.dqn


DQN
===

`Deep Q Network (DQN) <https://arxiv.org/abs/1312.5602>`_ builds on `Fitted Q-Iteration (FQI) <http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf>`_
and make use of different tricks to stabilize the learning with neural networks: it uses a replay buffer, a target network and gradient clipping.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy


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
Discrete      ✔️      ✔️
Box           ❌      ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
Dict          ❌      ✔️️
============= ====== ===========


Example
-------

This example is only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized hyperparameters can be found in RL Zoo `repository <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

.. code-block:: python

  import gym

  from stable_baselines3 import DQN

  env = gym.make("CartPole-v0")

  model = DQN("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("dqn_cartpole")

  del model # remove to demonstrate saving and loading

  model = DQN.load("dqn_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()


Results
-------

Atari Games
^^^^^^^^^^^

The complete learning curves are available in the `associated PR #110 <https://github.com/DLR-RM/stable-baselines3/pull/110>`_.


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the `rl-zoo repo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


Run the benchmark (replace ``$ENV_ID`` by the env id, for instance ``BreakoutNoFrameskip-v4``):

.. code-block:: bash

  python train.py --algo dqn --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a dqn -e Pong Breakout -f logs/ -o logs/dqn_results
  python scripts/plot_from_file.py -i logs/dqn_results.pkl -latex -l DQN


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

.. autoclass:: MultiInputPolicy
  :members:
