.. _rl_zoo:

==================
RL Baselines3 Zoo
==================

`RL Baselines3 Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ is a training framework for Reinforcement Learning (RL).

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

Installation
------------

1. Clone the repository:

::

  git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


.. note::

	You can remove the ``--recursive`` option if you don't want to download the trained agents


.. note::

  If you only need the training/plotting scripts and additional callbacks/wrappers from the RL Zoo, you can also install it via pip: ``pip install rl_zoo3``


2. Install dependencies
::

   apt-get install swig cmake ffmpeg
   pip install -r requirements.txt


Train an Agent
--------------

The hyperparameters for each environment are defined in
``hyperparameters/algo_name.yml``.

If the environment exists in this file, then you can train an agent
using:

::

 python train.py --algo algo_name --env env_id

For example (with evaluation and checkpoints):

::

 python train.py --algo ppo --env CartPole-v1 --eval-freq 10000 --save-freq 50000


Continue training (here, load pretrained agent for Breakout and continue
training for 5000 steps):

::

 python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i trained_agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000


Enjoy a Trained Agent
---------------------

If the trained agent exists, then you can see it in action using:

::

  python enjoy.py --algo algo_name --env env_id

For example, enjoy A2C on Breakout during 5000 timesteps:

::

  python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000


Hyperparameter Optimization
---------------------------

We use `Optuna <https://optuna.org/>`_ for optimizing the hyperparameters.


Tune the hyperparameters for PPO, using a random sampler and median pruner, 2 parallels jobs,
with a budget of 1000 trials and a maximum of 50000 steps:

::

  python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 \
    --sampler random --pruner median


Colab Notebook: Try it Online!
------------------------------

You can train agents online using Google `colab notebook <https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb>`_.


.. note::

	You can find more information about the rl baselines3 zoo in the repo `README <https://github.com/DLR-RM/rl-baselines3-zoo>`_. For instance, how to record a video of a trained agent.
