.. Stable Baselines3 documentation master file, created by
   sphinx-quickstart on Thu Sep 26 11:06:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Stable Baselines3 docs! - RL Baselines Made Easy
===========================================================

`Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_ is a set of improved implementations of reinforcement learning algorithms in PyTorch.
It is the next major version of `Stable Baselines <https://github.com/hill-a/stable-baselines>`_.


Github repository: https://github.com/DLR-RM/stable-baselines3

RL Baselines3 Zoo (collection of pre-trained agents): https://github.com/DLR-RM/rl-baselines3-zoo

RL Baselines3 Zoo also offers a simple interface to train, evaluate agents and do hyperparameter tuning.


Main Features
--------------

- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- Tests, high code coverage and type hints
- Clean code



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
   guide/rl_zoo
   guide/migration
   guide/checking_nan
   guide/developer


.. toctree::
  :maxdepth: 1
  :caption: RL Algorithms

  modules/base
  modules/a2c
  modules/ppo
  modules/sac
  modules/td3

.. toctree::
  :maxdepth: 1
  :caption: Common

  common/atari_wrappers
  common/cmd_utils
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

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
