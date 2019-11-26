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
- Refactor: buffer with numpy array instead of pytorch
- Refactor: remove duplicated code for evaluation

- plotting? -> zoo

Later:
- get_parameters / set_parameters
- SDE: use [affine transform](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Affine)
  to scale the noise after a tanh transform?
- CNN policies + normalization
- tensorboard support
- DQN
- TRPO
- ACER
- DDPG
- HER -> use stable-baselines because does not depends on tf?
