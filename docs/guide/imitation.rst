.. _imitation:

Imitation Learning
==================

The `imitation <https://github.com/HumanCompatibleAI/imitation>`__ library implements
imitation learning algorithms on top of Stable-Baselines3, including:

  - Behavioral Cloning
  - `DAgger <https://arxiv.org/abs/1011.0686>`_ with synthetic examples
  - `Adversarial Inverse Reinforcement Learning <https://arxiv.org/abs/1710.11248>`_ (AIRL)
  - `Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_  (GAIL)


It also provides `CLI scripts <#cli-quickstart>`_ for training and saving
demonstrations from RL experts, and for training imitation learners on these demonstrations.


Installation
------------

Installation requires Python 3.7+:

::

  pip install imitation


CLI Quickstart
---------------------

::

  # Train PPO agent on cartpole and collect expert demonstrations
  python -m imitation.scripts.expert_demos with fast cartpole log_dir=quickstart

  # Train GAIL from demonstrations
  python -m imitation.scripts.train_adversarial with fast gail cartpole rollout_path=quickstart/rollouts/final.pkl

  # Train AIRL from demonstrations
  python -m imitation.scripts.train_adversarial with fast airl cartpole rollout_path=quickstart/rollouts/final.pkl


.. note::

    You can remove the ``fast`` option to run training to completion. For more CLI options
    and information on reading Tensorboard plots, see the
    `README <https://github.com/HumanCompatibleAI/imitation#cli-quickstart>`_.


Python Interface Quickstart
---------------------------

This `example script <https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py>`_
uses the Python API to train BC, GAIL, and AIRL models on CartPole data.
