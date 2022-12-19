import os

from setuptools import find_packages, setup

with open(os.path.join("stable_baselines3", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable Baselines3

Stable Baselines3 is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It is the next major version of [Stable Baselines](https://github.com/hill-a/stable-baselines).

These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.


## Links

Repository:
https://github.com/DLR-RM/stable-baselines3

Blog post:
https://araffin.github.io/post/sb3/

Documentation:
https://stable-baselines3.readthedocs.io/en/master/

RL Baselines3 Zoo:
https://github.com/DLR-RM/rl-baselines3-zoo

SB3 Contrib:
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

## Quick example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms using Gym.

Here is a quick example of how to train and run PPO on a cartpole environment:

```python
import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

```

Or just train a model with a one liner if [the environment is registered in Gym](https://www.gymlibrary.ml/content/environment_creation/) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)
```

"""  # noqa:E501


setup(
    name="stable_baselines3",
    packages=[package for package in find_packages() if package.startswith("stable_baselines3")],
    package_data={"stable_baselines3": ["py.typed", "version.txt"]},
    install_requires=[
        "gym==0.21",  # Fixed version due to breaking changes in 0.22
        "numpy",
        "torch>=1.11",
        'typing_extensions>=4.0,<5; python_version < "3.8.0"',
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
        # gym and flake8 not compatible with importlib-metadata>5.0
        "importlib-metadata~=4.13",
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
            "mypy",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
            # For toy text Gym envs
            "scipy>=1.4.1",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
        "extra": [
            # For render
            "opencv-python",
            # For atari games,
            "ale-py==0.7.4",
            "autorom[accept-rom-license]~=0.4.2",
            "pillow",
            # Tensorboard support
            "tensorboard>=2.9.1",
            # Checking memory taken by replay buffer
            "psutil",
            # For progress bar callback
            "tqdm",
            "rich",
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
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
