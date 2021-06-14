.. _logger:

Logger
======

To overwrite the default logger, you can pass one to the algorithm.
Available formats are ``["stdout", "csv", "log", "tensorboard", "json"]``.


.. warning::

  When passing a custom logger object,
  this will overwrite ``tensorboard_log`` and ``verbose`` settings
  passed to the constructor.


.. code-block:: python

  from stable_baselines3 import A2C
  from stable_baselines3.common.logger import configure

  tmp_path = "/tmp/sb3_log/"
  # set up logger
  new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

  model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
  # Set new logger
  model.set_logger(new_logger)
  model.learn(10000)

.. automodule:: stable_baselines3.common.logger
  :members:
