.. _sbx:

==========================
Stable Baselines Jax (SBX)
==========================

`Stable Baselines Jax (SBX) <https://github.com/araffin/sbx>`_ is a proof of concept version of Stable-Baselines3 in Jax.

It provides a minimal number of features compared to SB3 but can be much faster (up to 20x times!): https://twitter.com/araffin2/status/1590714558628253698

Implemented algorithms:

- Soft Actor-Critic (SAC) and SAC-N
- Truncated Quantile Critics (TQC)
- Dropout Q-Functions for Doubly Efficient Reinforcement Learning (DroQ)
- Proximal Policy Optimization (PPO)
- Deep Q Network (DQN)
- Twin Delayed DDPG (TD3)
- Deep Deterministic Policy Gradient (DDPG)
- Batch Normalization in Deep Reinforcement Learning (CrossQ)


As SBX follows SB3 API, it is also compatible with the `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_.
For that you will need to create two files:

``train_sbx.py``:

.. code-block:: python

  import rl_zoo3
  import rl_zoo3.train
  from rl_zoo3.train import train
  from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

  rl_zoo3.ALGOS["ddpg"] = DDPG
  rl_zoo3.ALGOS["dqn"] = DQN
  # See SBX readme to use DroQ configuration
  # rl_zoo3.ALGOS["droq"] = DroQ
  rl_zoo3.ALGOS["sac"] = SAC
  rl_zoo3.ALGOS["ppo"] = PPO
  rl_zoo3.ALGOS["td3"] = TD3
  rl_zoo3.ALGOS["tqc"] = TQC
  rl_zoo3.ALGOS["crossq"] = CrossQ
  rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
  rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS


  if __name__ == "__main__":
      train()

Then you can call ``python train_sbx.py --algo sac --env Pendulum-v1`` and use the RL Zoo CLI.


``enjoy_sbx.py``:

.. code-block:: python

  import rl_zoo3
  import rl_zoo3.enjoy
  from rl_zoo3.enjoy import enjoy
  from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

  rl_zoo3.ALGOS["ddpg"] = DDPG
  rl_zoo3.ALGOS["dqn"] = DQN
  # See SBX readme to use DroQ configuration
  # rl_zoo3.ALGOS["droq"] = DroQ
  rl_zoo3.ALGOS["sac"] = SAC
  rl_zoo3.ALGOS["ppo"] = PPO
  rl_zoo3.ALGOS["td3"] = TD3
  rl_zoo3.ALGOS["tqc"] = TQC
  rl_zoo3.ALGOS["crossq"] = CrossQ
  rl_zoo3.enjoy.ALGOS = rl_zoo3.ALGOS
  rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS


  if __name__ == "__main__":
      enjoy()
