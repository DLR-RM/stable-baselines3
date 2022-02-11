.. _projects:

Projects
=========

This is a list of projects using stable-baselines3.
Please tell us, if you want your project to appear on this page ;)

DriverGym
---------

An open-source Gym-compatible environment specifically tailored for developing RL algorithms for autonomous driving. DriverGym provides access to more than 1000 hours of expert logged data and also supports reactive and data-driven agent behavior. The performance of an RL policy can be easily validated using an extensive and flexible closed-loop evaluation protocol. We also provide behavior cloning baselines using supervised learning and RL, trained in DriverGym.

| Authors: Parth Kothari, Christian Perone, Luca Bergamini, Alexandre Alahi, Peter Ondruska
| Github: https://github.com/lyft/l5kit
| Paper: https://arxiv.org/abs/2111.06889 


RL Reach
--------

A platform for running reproducible reinforcement learning experiments for customisable robotic reaching tasks. This self-contained and straightforward toolbox allows its users to quickly investigate and identify optimal training configurations.

| Authors: Pierre Aumjaud, David McAuliffe, Francisco Javier Rodríguez Lera, Philip Cardiff
| Github: https://github.com/PierreExeter/rl_reach
| Paper: https://arxiv.org/abs/2102.04916


Generalized State Dependent Exploration for Deep Reinforcement Learning in Robotics
-----------------------------------------------------------------------------------

An exploration method to train RL agent directly on real robots.
It was the starting point of Stable-Baselines3.

| Author: Antonin Raffin, Freek Stulp
| Github: https://github.com/DLR-RM/stable-baselines3/tree/sde
| Paper: https://arxiv.org/abs/2005.05719


Furuta Pendulum Robot
---------------------

Everything you need to build and train a rotary inverted pendulum, also know as a furuta pendulum! This makes use of gSDE listed above.   
The Github repository contains code, CAD files and a bill of materials for you to build the robot. You can watch `a video overview of the project here <https://www.youtube.com/watch?v=Y6FVBbqjR40>`_.

| Authors: Armand du Parc Locmaria, Pierre Fabre
| Github: https://github.com/Armandpl/furuta


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

gym-pybullet-drones
-------------------
PyBullet Gym environments for single and multi-agent reinforcement learning of quadcopter control.

- Physics-based simulation for the development and test of quadcopter control.
- Compatibility with ``gym.Env``, RLlib's MultiAgentEnv.
- Learning and testing script templates for stable-baselines3 and RLlib.

| Author: Jacopo Panerati
| Github: https://github.com/utiasDSL/gym-pybullet-drones/
| Paper: https://arxiv.org/abs/2103.02142

SuperSuit
---------

SuperSuit contains easy to use wrappers for Gym (and multi-agent PettingZoo) environments to do all forms of common preprocessing (frame stacking, converting graphical observations to greyscale, max-and-skip for Atari, etc.). It also notably includes:

-Wrappers that apply lambda functions to observations, actions, or rewards with a single line of code.
-All wrappers can be used natively on vector environments, wrappers exist to Gym environments to vectorized environments and concatenate multiple vector environments together
-A wrapper is included that allows for using regular single agent RL libraries (e.g. stable baselines) to learn simple multi-agent PettingZoo environments, explained in this tutorial:

| Author: Justin Terry
| GitHub: https://github.com/PettingZoo-Team/SuperSuit
| Tutorial on multi-agent support in stable baselines: https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b

Rocket League Gym
-----------------

A fully custom python API and C++ DLL to treat the popular game Rocket League like an OpenAI Gym environment.

- Dramatically increases the rate at which the game runs.
- Supports full configuration of initial states, observations, rewards, and terminal states.
- Supports multiple simultaneous game clients.
- Supports multi-agent training and self-play.
- Provides custom wrappers for easy use with stable-baselines3.

| Authors: Lucas Emery, Matthew Allen
| GitHub: https://github.com/lucas-emery/rocket-league-gym
| Website: https://rlgym.github.io

gym-electric-motor
-------------------

An OpenAI gym environment for the simulation and control of electric drive trains.
Think of Matlab/Simulink for electric motors, inverters, and load profiles, but non-graphical and open-source in Python.

`gym-electric-motor` offers a rich interface for customization, including
- plug-and-play of different control algorithms ranging from classical controllers (like field-oriented control) up to any RL agent you can find,
- reward shaping,
- load profiling,
- finite-set or continuous-set control,
- one-phase and three-phase motors such as induction machines and permanent magnet synchronous motors, among others.

SB3 is used as an example in one of many tutorials showcasing the easy usage of `gym-electric-motor`.

| Author: `Paderborn University, LEA department <https://github.com/upb-lea>`_
| GitHub: https://github.com/upb-lea/gym-electric-motor
| SB3 Tutorial: `Colab Link <https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/stable_baselines3_dqn_disc_pmsm_example.ipynb>`_
| Paper: `JOSS <https://joss.theoj.org/papers/10.21105/joss.02498>`_, `TNNLS <https://ieeexplore.ieee.org/document/9241851>`_, `ArXiv <https://arxiv.org/abs/1910.09434>`_

policy-distillation-baselines
------------------------------
A PyTorch implementation of Policy Distillation for control, which has well-trained teachers via Stable Baselines3.

- `policy-distillation-baselines` provides some good examples for policy distillation in various environment and using reliable algorithms.
- All well-trained models and algorithms are compatible with Stable Baselines3.

| Authors: Junyeob Baek
| GitHub: https://github.com/CUN-bjy/policy-distillation-baselines
| Demo: `link <https://github.com/CUN-bjy/policy-distillation-baselines/issues/3#issuecomment-817730173>`_

highway-env
-------------------

A minimalist environment for decision-making in Autonomous Driving.

Driving policies can be trained in different scenarios, and several notebooks using SB3 are provided as examples.

| Author: `Edouard Leurent <https://edouardleurent.com>`_
| GitHub: https://github.com/eleurent/highway-env
| Examples: `Colab Links <https://github.com/eleurent/highway-env/tree/master/scripts#using-stable-baselines3>`_

tactile-gym
-------------------

Suite of RL environments focussed on using a simulated tactile sensor as the primary source of observations. Sim-to-Real results across 4 out of 5 proposed envs.

| Author: Alex Church
| GitHub: https://github.com/ac-93/tactile_gym
| Paper: https://arxiv.org/abs/2106.08796
| Website: `tactile-gym website <https://sites.google.com/my.bristol.ac.uk/tactile-gym-sim2real/home>`_
