(projects)=

# Projects

This is a list of projects using stable-baselines3.
Please tell us, if you want your project to appear on this page ;)

## DriverGym

An open-source Gym-compatible environment specifically tailored for developing RL algorithms for autonomous driving. DriverGym provides access to more than 1000 hours of expert logged data and also supports reactive and data-driven agent behavior. The performance of an RL policy can be easily validated using an extensive and flexible closed-loop evaluation protocol. We also provide behavior cloning baselines using supervised learning and RL, trained in DriverGym.

Authors: Parth Kothari, Christian Perone, Luca Bergamini, Alexandre Alahi, Peter Ondruska

Github: 

<https://github.com/lyft/l5kit>

Paper: 

<https://arxiv.org/abs/2111.06889>

## RL Reach

A platform for running reproducible reinforcement learning experiments for customizable robotic reaching tasks. This self-contained and straightforward toolbox allows its users to quickly investigate and identify optimal training configurations.

Authors: Pierre Aumjaud, David McAuliffe, Francisco Javier Rodríguez Lera, Philip Cardiff

Github: 

<https://github.com/PierreExeter/rl_reach>

Paper: 

<https://arxiv.org/abs/2102.04916>

## Generalized State Dependent Exploration for Deep Reinforcement Learning in Robotics

An exploration method to train RL agent directly on real robots.
It was the starting point of Stable-Baselines3.

Author: Antonin Raffin, Freek Stulp

Github: 

<https://github.com/DLR-RM/stable-baselines3/tree/sde>

Paper: 

<https://arxiv.org/abs/2005.05719>

## Furuta Pendulum Robot

Everything you need to build and train a rotary inverted pendulum, also known as a furuta pendulum! This makes use of gSDE listed above.
The Github repository contains code, CAD files and a bill of materials for you to build the robot. You can watch [a video overview of the project here](https://www.youtube.com/watch?v=Y6FVBbqjR40).

Authors: Armand du Parc Locmaria, Pierre Fabre

Github: 

<https://github.com/Armandpl/furuta>

## Reacher

A solution to the second project of the Udacity deep reinforcement learning course.
It is an example of:

- wrapping single and multi-agent Unity environments to make them usable in Stable-Baselines3
- creating experimentation scripts which train and run A2C, PPO, TD3 and SAC models (a better choice for this one is <https://github.com/DLR-RM/rl-baselines3-zoo>)
- generating several pre-trained models which solve the reacher environment

Author: Marios Koulakis

Github: 

<https://github.com/koulakis/reacher-deep-reinforcement-learning>

## SUMO-RL

A simple interface to instantiate RL environments with SUMO for Traffic Signal Control.

- Supports Multiagent RL
- Compatibility with gym.Env and popular RL libraries such as stable-baselines3 and RLlib
- Easy customization: state and reward definitions are easily modifiable

Author: Lucas Alegre

Github: 

<https://github.com/LucasAlegre/sumo-rl>

## gym-pybullet-drones

PyBullet Gym environments for single and multi-agent reinforcement learning of quadcopter control.

- Physics-based simulation for the development and test of quadcopter control.
- Compatibility with `gym.Env`, RLlib's MultiAgentEnv.
- Learning and testing script templates for stable-baselines3 and RLlib.

Author: Jacopo Panerati

Github: 

<https://github.com/utiasDSL/gym-pybullet-drones/>

Paper: 

<https://arxiv.org/abs/2103.02142>

## SuperSuit

SuperSuit contains easy to use wrappers for Gym (and multi-agent PettingZoo) environments to do all forms of common preprocessing (frame stacking, converting graphical observations to greyscale, max-and-skip for Atari, etc.). It also notably includes:

-Wrappers that apply lambda functions to observations, actions, or rewards with a single line of code.
-All wrappers can be used natively on vector environments, wrappers exist to Gym environments to vectorized environments and concatenate multiple vector environments together
-A wrapper is included that allows for using regular single agent RL libraries (e.g. stable baselines) to learn simple multi-agent PettingZoo environments, explained in this tutorial:

Author: Justin Terry

GitHub: 

<https://github.com/PettingZoo-Team/SuperSuit>

Tutorial on multi-agent support in stable baselines: 

<https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b>

## Rocket League Gym

A fully custom python API and C++ DLL to treat the popular game Rocket League like an OpenAI Gym environment.

- Dramatically increases the rate at which the game runs.
- Supports full configuration of initial states, observations, rewards, and terminal states.
- Supports multiple simultaneous game clients.
- Supports multi-agent training and self-play.
- Provides custom wrappers for easy use with stable-baselines3.

Authors: Lucas Emery, Matthew Allen

GitHub: 

<https://github.com/lucas-emery/rocket-league-gym>

Website: 

<https://rlgym.github.io>

## gym-electric-motor

An OpenAI gym environment for the simulation and control of electric drive trains.
Think of Matlab/Simulink for electric motors, inverters, and load profiles, but non-graphical and open-source in Python.

`gym-electric-motor` offers a rich interface for customization, including
\- plug-and-play of different control algorithms ranging from classical controllers (like field-oriented control) up to any RL agent you can find,
\- reward shaping,
\- load profiling,
\- finite-set or continuous-set control,
\- one-phase and three-phase motors such as induction machines and permanent magnet synchronous motors, among others.

SB3 is used as an example in one of many tutorials showcasing the easy usage of `gym-electric-motor`.

Author: 

[Paderborn University, LEA department](https://github.com/upb-lea)

GitHub: 

<https://github.com/upb-lea/gym-electric-motor>

SB3 Tutorial: 

[Colab Link](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/stable_baselines3_dqn_disc_pmsm_example.ipynb)

Paper: 

[JOSS](https://joss.theoj.org/papers/10.21105/joss.02498)

, 

[TNNLS](https://ieeexplore.ieee.org/document/9241851)

, 

[ArXiv](https://arxiv.org/abs/1910.09434)

## policy-distillation-baselines

A PyTorch implementation of Policy Distillation for control, which has well-trained teachers via Stable Baselines3.

- `policy-distillation-baselines` provides some good examples for policy distillation in various environment and using reliable algorithms.
- All well-trained models and algorithms are compatible with Stable Baselines3.

Authors: Junyeob Baek

GitHub: 

<https://github.com/CUN-bjy/policy-distillation-baselines>

Demo: 

[link](https://github.com/CUN-bjy/policy-distillation-baselines/issues/3#issuecomment-817730173)

## highway-env

A minimalist environment for decision-making in Autonomous Driving.

Driving policies can be trained in different scenarios, and several notebooks using SB3 are provided as examples.

Author: 

[Edouard Leurent](https://edouardleurent.com)

GitHub: 

<https://github.com/eleurent/highway-env>

Examples: 

[Colab Links](https://github.com/eleurent/highway-env/tree/master/scripts#using-stable-baselines3)

## tactile-gym

Suite of RL environments focused on using a simulated tactile sensor as the primary source of observations. Sim-to-Real results across 4 out of 5 proposed envs.

Author: Alex Church

GitHub: 

<https://github.com/ac-93/tactile_gym>

Paper: 

<https://arxiv.org/abs/2106.08796>

Website: 

[tactile-gym website](https://sites.google.com/my.bristol.ac.uk/tactile-gym-sim2real/home)

## RLeXplore

RLeXplore is a set of implementations of intrinsic reward driven-exploration approaches in reinforcement learning using PyTorch, which can be deployed in arbitrary algorithms in a plug-and-play manner. In particular, RLeXplore is designed to be well compatible with Stable-Baselines3, providing more stable exploration benchmarks.

- Support arbitrary RL algorithms;
- Highly modular and high expansibility;
- Keep up with the latest research progress.

Author: Mingqi Yuan

GitHub: 

<https://github.com/yuanmingqi/rl-exploration-baselines>

## UAV_Navigation_DRL_AirSim

A platform for training UAV navigation policies in complex unknown environments.

- Based on AirSim and SB3.
- An Open AI Gym env is created including kinematic models for both multirotor and fixed-wing UAVs.
- Some UE4 environments are provided to train and test the navigation policy.

Try to train your own autonomous flight policy and even transfer it to real UAVs! Have fun ^\_^!

Author: Lei He

Github: 

<https://github.com/heleidsn/UAV_Navigation_DRL_AirSim>

## Pink Noise Exploration

A simple library for pink noise exploration with deterministic (DDPG / TD3) and stochastic (SAC) off-policy algorithms. Pink noise has been shown to work better than uncorrelated Gaussian noise (the default choice) and Ornstein-Uhlenbeck noise on a range of continuous control benchmark tasks. This library is designed to work with Stable Baselines3.

Authors: Onno Eberhard, Jakob Hollenstein, Cristina Pinneri, Georg Martius

Github: 

<https://github.com/martius-lab/pink-noise-rl>

Paper: 

<https://openreview.net/forum?id=hQ9V5QN27eS>

 (Oral at ICLR 2023)

## mobile-env

An open, minimalist Gymnasium environment for autonomous coordination in wireless mobile networks.
It allows simulating various scenarios with moving users in a cellular network with multiple base stations.

- Written in pure Python, easy to modify and extend, and can be installed directly via PyPI.
- Implements the standard Gymnasium interface such that it can be used with all common frameworks for reinforcement learning.
- There are examples for both single-agent and multi-agent RL using either `stable-baselines3` or Ray RLlib.

Authors: Stefan Schneider, Stefan Werner

Github: 

<https://github.com/stefanbschneider/mobile-env>

Paper: 

<https://ris.uni-paderborn.de/download/30236/30237>

 (2022 IEEE/IFIP Network Operations and Management Symposium (NOMS))

## DeepNetSlice

A Deep Reinforcement Learning Open-Source Toolkit for Network Slice Placement (NSP).

NSP is the problem of deciding which physical servers in a network should host the virtual network functions (VNFs) that make up a network slice, as well as managing the mapping of the virtual links between the VNFs onto the physical infrastructure.
It is a complex optimization problem, as it involves considering the requirements of the network slice and the available resources on the physical network.
The goal is generally to maximize the utilization of the physical resources while ensuring that the network slices meet their performance requirements.

The toolkit includes a customizable simulation environments, as well as some ready-to-use demos for training
intelligent agents to perform network slice placement.

Author: Alex Pasquali

Github: 

<https://github.com/AlexPasqua/DeepNetSlice>

Paper: 

<https://ieeexplore.ieee.org/document/10625023>

Associated Master's Thesis: 

<https://etd.adm.unipi.it/theses/available/etd-01182023-110038/unrestricted/Tesi_magistrale_Pasquali_Alex.pdf>

## PokemonRedExperiments

Playing Pokemon Red with Reinforcement Learning.

Author: Peter Whidden

Github: 

<https://github.com/PWhiddy/PokemonRedExperiments>

Video: 

<https://www.youtube.com/watch?v=DcYLT37ImBY>

## Evolving Reservoirs for Meta Reinforcement Learning

Meta-RL framework to optimize reservoir-like neural structures (special kind of RNNs), and integrate them to RL agents to improve their training.
It enables solving environments involving partial observability or locomotion (e.g MuJoCo), and optimizing reservoirs that can generalize to unseen tasks.

Authors: Corentin Léger, Gautier Hamon, Eleni Nisioti, Xavier Hinaut, Clément Moulin-Frier

Github: 

<https://github.com/corentinlger/ER-MRL>

Paper: 

<https://arxiv.org/abs/2312.06695>

## FootstepNet Envs

These environments are dedicated to train efficient agents that can plan and forecast bipedal robot footsteps in order to go to a target location possibly avoiding obstacles. They are designed to be used with Reinforcement Learning (RL) algorithms.

Real world experiments were conducted during RoboCup competitions on the Sigmaban robot, a small-sized humanoid designed by the *Rhoban Team*.

Authors: Clément Gaspard, Grégoire Passault, Mélodie Daniel, Olivier Ly

Github: 

<https://github.com/Rhoban/footstepnet_envs>

Paper: 

<https://arxiv.org/abs/2403.12589>

## FRASA: Fall Recovery And Stand up agent

A Deep Reinforcement Learning agent for a humanoid robot that learns to recover from falls and stand up.

The agent is trained using the MuJoCo physics engine. Real world experiments are conducted on the
Sigmaban humanoid robot, a small-sized humanoid designed by the *Rhoban Team* to compete in the RoboCup Kidsize League.
The results, detailed in the paper and the video, show that the agent is able to recover from
various external disturbances and stand up in a few seconds.

Authors: Marc Duclusaud, Clément Gaspard, Grégoire Passault, Mélodie Daniel, Olivier Ly

Github: 

<https://github.com/Rhoban/frasa>

Paper: 

<https://arxiv.org/abs/2410.08655>

Video: 

<https://www.youtube.com/watch?v=NL65XW0O0mk>

## sb3-extra-buffers: RAM expansions are overrated, just compress your observations!

Reduce the memory consumption of memory buffers in Reinforcement Learning while adding minimal overhead.

Tired of reading a cool RL paper and realizing that the author is storing a **MILLION** observations in their replay buffers? Yeah me too.
This project has implemented several compressed buffer classes that replace Stable Baselines3's standard buffers like ReplayBuffer and
RolloutBuffer. With as simple as 2-5 lines of extra code and **negligible overhead**, memory usage can be reduced by more than **95%**!
Benchmark results and documentations are on Github, feel free to submit feature requests / ask how to use these buffers through issues.

Authors: Hugo Huang

Github: 

<https://github.com/Trenza1ore/sb3-extra-buffers>

Relevant project for training RL agents that play Doom with Semantic Segmentation: 

<https://github.com/Trenza1ore/SegDoom>

## sb3-plus: Multi-Output Policy Support for Stable-Baselines3

An extension to Stable-Baselines3 that implements support for multi-output policies and dictionary action spaces.

This project provides PPO with dict action space support, enabling independent action spaces which is particularly useful
for environments requiring multiple types of actions (e.g., discrete and continuous actions). This addresses the
multi-output policy feature requested in the community and provides a practical solution for complex action scenarios.

Author: Adyson Maia

Github: 

<https://github.com/adysonmaia/sb3-plus>
