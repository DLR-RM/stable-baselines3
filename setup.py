import sys
import subprocess
from setuptools import setup, find_packages


setup(name='torchy_baselines',
      packages=[package for package in find_packages()
                if package.startswith('torchy_baselines')],
      install_requires=[
          'gym[classic_control]>=0.11',
          'numpy',
          'torch>=1.4.0',
          'cloudpickle',
          # For reading logs
          'pandas',
          # Plotting learning curves
          'matplotlib'
      ],
      extras_require={
        'tests': [
            'pytest',
            'pytest-cov',
            'pytest-env',
            'pytest-xdist',
            'pytype',
        ],
        'docs': [
            'sphinx',
            'sphinx-autobuild',
            'sphinx-rtd-theme',
            # For spelling
            'sphinxcontrib.spelling',
            # Type hints support
            'sphinx-autodoc-typehints'
        ],
        'extra': [
            # For render
            'opencv-python',
        ]
      },
      description='Pytorch version of Stable Baselines, implementations of reinforcement learning algorithms.',
      author='Antonin Raffin',
      url='',
      author_email='antonin.raffin@dlr.de',
      keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
               "gym openai stable baselines toolbox python data-science",
      license="MIT",
      long_description="",
      long_description_content_type='text/markdown',
      version="0.2.2",
      )

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
