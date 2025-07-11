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
import gymnasium

from stable_baselines3 import PPO

env = gymnasium.make("CartPole-v1", render_mode="human")

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

Or just train a model with a one liner if [the environment is registered in Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

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
        "gymnasium>=0.29.1,<1.3.0",
        "numpy>=1.20,<3.0",
        "torch>=2.3,<3.0",
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
            "mypy",
            # Lint code and sort imports (flake8 and isort replacement)
            "ruff>=0.3.1",
            # Reformat
            "black>=25.1.0,<26",
        ],
        "docs": [
            "sphinx>=5,<9",
            "sphinx-autobuild",
            "sphinx-rtd-theme>=1.3.0",
            # For spelling
            "sphinxcontrib.spelling",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
        "extra": [
            # For render
            "opencv-python",
            "pygame",
            # Tensorboard support
            "tensorboard>=2.9.1",
            # Checking memory taken by replay buffer
            "psutil",
            # For progress bar callback
            "tqdm",
            "rich",
            # For atari games,
            "ale-py>=0.9.0",
            "pillow",
        ],
    },
    description="Pytorch version of Stable Baselines, implementations of reinforcement learning algorithms.",
    author="Antonin Raffin",
    url="https://github.com/DLR-RM/stable-baselines3",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gymnasium gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.9",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/DLR-RM/stable-baselines3",
        "Documentation": "https://stable-baselines3.readthedocs.io/",
        "Changelog": "https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html",
        "SB3-Contrib": "https://github.com/Stable-Baselines-Team/stable-baselines3-contrib",
        "RL-Zoo": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "SBX": "https://github.com/araffin/sbx",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
