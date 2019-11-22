<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master)

# Torchy Baselines

PyTorch version of [Stable Baselines](https://github.com/hill-a/stable-baselines), a set of improved implementations of reinforcement learning algorithms.

## Implemented Algorithms

- A2C
- CEM-RL (with TD3)
- PPO
- SAC
- TD3


## Roadmap

TODO:
- save/load
- better predict
- complete logger
- SDE: reduce the number of parameters (only n_features instead of n_features x n_actions) for A2C
(done for TD3)
- SDE: learn the feature extractor?
- Refactor: buffer with numpy array instead of pytorch
- Refactor: remove duplicated code for evaluation
- plotting? -> zoo

Later:
- get_parameters / set_parameters
- CNN policies + normalization
- tensorboard support
- DQN
- TRPO
- ACER
- DDPG
- HER -> use stable-baselines because does not depends on tf?
