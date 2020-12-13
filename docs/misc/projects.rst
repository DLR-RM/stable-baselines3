.. _projects:

Projects
=========

This is a list of projects using stable-baselines3.
Please tell us, if you want your project to appear on this page ;)


.. RL Racing Robot
.. --------------------------
.. Implementation of reinforcement learning approach to make a donkey car learn to race.
.. Uses SAC on autoencoder features
..
.. | Author: Antonin Raffin  (@araffin)
.. | Github repo: https://github.com/araffin/RL-Racing-Robot


Generalized State Dependent Exploration for Deep Reinforcement Learning in Robotics
-----------------------------------------------------------------------------------

An exploration method to train RL agent directly on real robots.
It was the starting point of Stable-Baselines3.

| Author: Antonin Raffin, Freek Stulp
| Github: https://github.com/DLR-RM/stable-baselines3/tree/sde
| Paper: https://arxiv.org/abs/2005.05719

Reacher
-------
A solution to the second project of the Udacity deep reinforcement learning course.
It is an example of:

- wrapping single and multi-agent Unity environments to make them usable in Stable-Baselines3
- creating experimentation scripts which train and run A2C, PPO, TD3 and SAC models (a better choice for this one is https://github.com/DLR-RM/rl-baselines3-zoo)
- generating several pre-trained models which solve the reacher environment

| Author: Marios Koulakis
| Github: https://github.com/koulakis/reacher-deep-reinforcement-learning

SUMO-RL
-------
A simple interface to instantiate RL environments with SUMO for Traffic Signal Control.

- Supports Multiagent RL
- Compatibility with gym.Env and popular RL libraries such as stable-baselines3 and RLlib
- Easy customisation: state and reward definitions are easily modifiable

| Author: Lucas Alegre
| Github: https://github.com/LucasAlegre/sumo-rl