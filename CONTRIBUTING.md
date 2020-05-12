## Contributing to Stable-Baselines3

If you are interested in contributing to Stable-Baselines, your contributions will fall
into two categories:
1. You want to propose a new Feature and implement it
    - Create an issue about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/DLR-RM/stable-baselines3/issues
    - Pick an issue or feature and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/DLR-RM/stable-baselines3


If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Developing Stable-Baselines3

To develop Stable-Baselines3 on your machine, here are some tips:

1. Clone a copy of Stable-Baselines3 from source:

```bash
git clone https://github.com/DLR-RM/stable-baselines3
cd stable-baselines3/
```

2. Install Stable-Baselines3 in develop mode, with support for building the docs and running tests:

```bash
pip install -e .[docs,tests,extra]
```

## Codestyle

We follow the [PEP8 codestyle](https://www.python.org/dev/peps/pep-0008/). Please order the imports as follows:

1. built-in
2. packages
3. current module

with one space between each,  that gives for instance:
```python
import os
import warnings

import numpy as np

from stable_baselines3 import PPO
```

In general, we recommend using pycharm to format everything in an efficient way.

Please document each function/method and [type](https://google.github.io/pytype/user_guide.html) them using the following template:

```python

def my_function(arg1: type1, arg2: type2) -> returntype:
    """
    Short description of the function.

    :param arg1: (type1) describe what is arg1
    :param arg2: (type2) describe what is arg2
    :return: (returntype) describe what is returned
    """
    ...
    return my_variable
```

## Pull Request (PR)

Before proposing a PR, please open an issue, where the feature will be discussed. This prevent from duplicated PR to be proposed and also ease the code review process.

Each PR need to be reviewed and accepted by at least one of the maintainers (@hill-a, @araffin, @erniejunior, @AdamGleave or @Miffyli).
A PR must pass the Continuous Integration tests (travis + codacy) to be merged with the master branch.

Note: in rare cases, we can create exception for codacy failure.

## Test

All new features must add tests in the `tests/` folder ensuring that everything works fine.
We use [pytest](https://pytest.org/).
Also, when a bug fix is proposed, tests should be added to avoid regression.

To run tests with `pytest`:

```
make pytest
```

Type checking with `pytype`:

```
make type
```

Codestyle check with `flake8`:

```
make lint
```

Build the documentation:

```
make doc
```

Check documentation spelling (you need to install `sphinxcontrib.spelling` package for that):

```
make spelling
```


## Changelog and Documentation

Please do not forget to update the changelog (`docs/misc/changelog.rst`) and add documentation if needed.
A README is present in the `docs/` folder for instructions on how to build the documentation.


Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.
