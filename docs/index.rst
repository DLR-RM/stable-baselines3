.. Stable Baselines3 documentation master file, created by
   sphinx-quickstart on Thu Sep 26 11:06:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations
========================================================================

`Stable Baselines3 (SB3) <https://github.com/DLR-RM/stable-baselines3>`_ is a set of reliable implementations of reinforcement learning algorithms in PyTorch.
It is the next major version of `Stable Baselines <https://github.com/hill-a/stable-baselines>`_.


Github repository: https://github.com/DLR-RM/stable-baselines3

RL Baselines3 Zoo (training framework for SB3): https://github.com/DLR-RM/rl-baselines3-zoo

RL Baselines3 Zoo provides a collection of pre-trained agents, scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

SB3 Contrib (experimental RL code, latest algorithms): https://github.com/Stable-Baselines-Team/stable-baselines3-contrib


Main Features
--------------

- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- Tests, high code coverage and type hints
- Clean code
- Tensorboard support
- **The performance of each algorithm was tested** (see *Results* section in their respective page)


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/rl_tips
   guide/rl
   guide/algos
   guide/examples
   guide/vec_envs
   guide/custom_env
   guide/custom_policy
   guide/callbacks
   guide/tensorboard
   guide/rl_zoo
   guide/sb3_contrib
   guide/imitation
   guide/migration
   guide/checking_nan
   guide/developer
   guide/save_format
   guide/export


.. toctree::
  :maxdepth: 1
  :caption: RL Algorithms

  modules/base
  modules/a2c
  modules/ddpg
  modules/dqn
  modules/her
  modules/ppo
  modules/sac
  modules/td3

.. toctree::
  :maxdepth: 1
  :caption: Common

  common/atari_wrappers
  common/env_util
  common/envs
  common/distributions
  common/evaluation
  common/env_checker
  common/monitor
  common/logger
  common/noise
  common/utils

.. toctree::
  :maxdepth: 1
  :caption: Misc

  misc/changelog
  misc/projects


Citing Stable Baselines3
------------------------
To cite this project in publications:

.. code-block:: bibtex

    @misc{stable-baselines3,
      author = {Raffin, Antonin and Hill, Ashley and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Dormann, Noah},
      title = {Stable Baselines3},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/DLR-RM/stable-baselines3}},
    }

Contributing
------------

To any interested in making the rl baselines better, there are still some improvements
that need to be done.
You can check issues in the `repo <https://github.com/DLR-RM/stable-baselines3/issues>`_.

If you want to contribute, please read `CONTRIBUTING.md <https://github.com/DLR-RM/stable-baselines3/blob/master/CONTRIBUTING.md>`_ first.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
