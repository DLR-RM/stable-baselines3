<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master)

# Torchy Baselines

PyTorch version of [Stable Baselines](https://github.com/hill-a/stable-baselines), a set of improved implementations of reinforcement learning algorithms.

NOTE: Python 3.6 is required!

## Implemented Algorithms

- A2C
- CEM-RL (with TD3)
- PPO
- SAC
- TD3

- SDE support for A2C, PPO, SAC and TD3.


## Roadmap

- cf github Roadmap


## Run the Tests

```
pip install -e .[tests]
make pytest
```

## Type check

```
pip install -e .[tests]
make type
```

## Build the Documentation

```
pip install -e .[docs]
make doc
```

Spell check for the documentation:

```
make spelling
```


## Citing the Project

To cite this repository in publications:

```
@misc{torchy-baselines,
  author = {Raffin, Antonin and Hill, Ashley and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Dormann, Noah},
  title = {Torchy Baselines},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/araffin/torchy-baselines}},
}
```
