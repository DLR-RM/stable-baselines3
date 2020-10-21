import os

from setuptools import find_packages, setup

with open(os.path.join("stable_baselines3", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable Baselines3

Stable Baselines3 is a set of improved implementations of reinforcement learning algorithms in PyTorch. It is the next major version of [Stable Baselines](https://github.com/hill-a/stable-baselines).

These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.


## Links

Repository:
https://github.com/DLR-RM/stable-baselines3

Medium article:
https://medium.com/@araffin/df87c4b2fc82

Documentation:
https://stable-baselines3.readthedocs.io/en/master/

RL Baselines3 Zoo:
https://github.com/DLR-RM/rl-baselines3-zoo

## Quick example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms using Gym.

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
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines3 import PPO

model = PPO('MlpPolicy', 'CartPole-v1').learn(10000)
```

"""  # noqa:E501


setup(
    name="stable_baselines3",
    packages=[package for package in find_packages() if package.startswith("stable_baselines3")],
    package_data={"stable_baselines3": ["py.typed", "version.txt"]},
    install_requires=[
        "gym>=0.17",
        "numpy",
        "torch>=1.4.0",
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
        "extra": [
            # For render
            "opencv-python",
            # For atari games,
            "atari_py~=0.2.0",
            "pillow",
            # Tensorboard support
            "tensorboard",
            # Checking memory taken by replay buffer
            "psutil",
        ],
    },
    description="Pytorch version of Stable Baselines, implementations of reinforcement learning algorithms.",
    author="Antonin Raffin",
    url="https://github.com/DLR-RM/stable-baselines3",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
