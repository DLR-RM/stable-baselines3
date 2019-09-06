import sys
import subprocess
from setuptools import setup, find_packages


setup(name='torchy_baselines',
      packages=[package for package in find_packages()
                if package.startswith('torchy_baselines')],
      install_requires=[
          'gym[classic_control]>=0.10.9',
          'numpy',
          'torch>=1.2.0' # torch>=1.2.0+cpu
      ],
      extras_require={
        'tests': [
            'pytest',
            'pytest-cov',
            'pytest-env',
            'pytest-xdist',
        ],
        'docs': [
            'sphinx',
            'sphinx-autobuild',
            'sphinx-rtd-theme'
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
      version="0.0.1",
      )

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
