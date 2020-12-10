<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![pipeline status](https://gitlab.com/araffin/stable-baselines3/badges/master/pipeline.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**WARNING: Stable Baselines3 is currently in a beta version, breaking changes may occur before 1.0 is released**


# Stable Baselines3

Stable Baselines3 is a set of improved implementations of reinforcement learning algorithms in PyTorch. It is the next major version of [Stable Baselines](https://github.com/hill-a/stable-baselines).

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).


These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

**Note: despite its simplicity of use, Stable Baselines3 (SB3) assumes you have some knowledge about Reinforcement Learning (RL).** You should not utilize this library without some practice. To that extent, we provide good resources in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/rl.html) to get started with RL.

## Main Features

**The performance of each algorithm was tested** (see *Results* section in their respective page),
you can take a look at the issues [#48](https://github.com/DLR-RM/stable-baselines3/issues/48) and [#49](https://github.com/DLR-RM/stable-baselines3/issues/49) for more details.


| **Features**                | **Stable-Baselines3** |
| --------------------------- | ----------------------|
| State of the art RL methods | :heavy_check_mark: |
| Documentation               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |
| Custom policies             | :heavy_check_mark: |
| Common interface            | :heavy_check_mark: |
| Ipython / Notebook friendly | :heavy_check_mark: |
| Tensorboard support         | :heavy_check_mark: |
| PEP8 code style             | :heavy_check_mark: |
| Custom callback             | :heavy_check_mark: |
| High code coverage          | :heavy_check_mark: |
| Type hints                  | :heavy_check_mark: |


### Planned features (v1.1+)

Please take a look at the [Roadmap](https://github.com/DLR-RM/stable-baselines3/issues/1) and [Milestones](https://github.com/DLR-RM/stable-baselines3/milestones).

## Migration guide: from Stable-Baselines (SB2) to Stable-Baselines3 (SB3)

A migration guide from SB2 to SB3 can be found in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/migration.html).

## Documentation

Documentation is available online: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)


## RL Baselines3 Zoo: A Collection of Trained RL Agents

[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo). is a collection of pre-trained Reinforcement Learning agents using Stable-Baselines3.

It also provides basic scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

Github repo: https://github.com/DLR-RM/rl-baselines3-zoo

Documentation: https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html

## Installation

**Note:** Stable-Baselines3 supports PyTorch 1.4+.

### Prerequisites
Stable Baselines3 requires python 3.6+.

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html#prerequisites).


### Install using pip
Install the Stable Baselines3 package:
```
pip install stable-baselines3[extra]
```

This includes an optional dependencies like Tensorboard, OpenCV or `atari-py` to train on atari games. If you do not need those, you can use:
```
pip install stable-baselines3
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more details and alternatives (from source, using docker).


## Example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO on a cartpole environment:
```python
import gym

from stable_baselines3 import PPO

env = gym.make('CartPole-v1')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines3 import PPO

model = PPO('MlpPolicy', 'CartPole-v1').learn(10000)
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more examples.


## Try it online with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:

- [Full Tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [All Notebooks](https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3)
- [Getting Started](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb)
- [Training, Saving, Loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb)
- [Multiprocessing](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb)
- [Monitor Training and Plotting](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb)
- [Atari Games](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb)
- [RL Baselines Zoo](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb)
- [PyBullet](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb)


## Implemented Algorithms

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:                |
| DDPG  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| DQN   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| HER   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| PPO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| SAC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TD3   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |


Actions `gym.spaces`:
 * `Box`: A N-dimensional box that containes every point in the action space.
 * `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.
 * `MultiDiscrete`: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.



## Testing the installation
All unit tests in stable baselines3 can be run using `pytest` runner:
```
pip install pytest pytest-cov
make pytest
```

You can also do a static type check using `pytype`:
```
pip install pytype
make type
```

Codestyle check with `flake8`:
```
pip install flake8
make lint
```

## Projects Using Stable-Baselines3

We try to maintain a list of project using stable-baselines3 in the [documentation](https://stable-baselines3.readthedocs.io/en/master/misc/projects.html),
please tell us when if you want your project to appear on this page ;)

## Citing the Project

To cite this repository in publications:

```
@misc{stable-baselines3,
  author = {Raffin, Antonin and Hill, Ashley and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Dormann, Noah},
  title = {Stable Baselines3},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/stable-baselines3}},
}
```

## Maintainers

Stable-Baselines3 is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave) and [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli).

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.


## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.

## Acknowledgments

The initial work to develop Stable Baselines3 was partially funded by the project *Reduced Complexity Models* from the *Helmholtz-Gemeinschaft Deutscher Forschungszentren*.

The original version, Stable Baselines, was created in the [robotics lab U2IS](http://u2is.ensta-paristech.fr/index.php?lang=en) ([INRIA Flowers](https://flowers.inria.fr/) team) at [ENSTA ParisTech](http://www.ensta-paristech.fr/en).


Logo credits: [L.M. Tenkes](https://www.instagram.com/lucillehue/)
