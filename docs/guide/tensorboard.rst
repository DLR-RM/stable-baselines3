.. _tensorboard:

Tensorboard Integration
=======================

Basic Usage
------------

To use Tensorboard with the rl baselines, you simply need to define a log location for the RL agent:

.. code-block:: python

    import gym

    from stable_baselines3 import A2C

    model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10000)


Or after loading an existing model (by default the log path is not saved):

.. code-block:: python

    import gym

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import A2C

    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = A2C.load("./a2c_cartpole.zip", env=env, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10000)


You can also define custom logging name when training (by default it is the algorithm name)

.. code-block:: python

    import gym

    from stable_baselines3 import A2C

    model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10000, tb_log_name="first_run")
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    model.learn(total_timesteps=10000, tb_log_name="second_run", reset_num_timesteps=False)
    model.learn(total_timesteps=10000, tb_log_name="third_run", reset_num_timesteps=False)


Once the learn function is called, you can monitor the RL agent during or after the training, with the following bash command:

.. code-block:: bash

  tensorboard --logdir ./a2c_cartpole_tensorboard/

you can also add past logging folders:

.. code-block:: bash

  tensorboard --logdir ./a2c_cartpole_tensorboard/;./ppo2_cartpole_tensorboard/

It will display information such as the episode reward, the model losses and other parameter unique to some models.

.. image:: ../_static/img/Tensorboard_example.png
  :width: 600
  :alt: plotting

Logging More Values
-------------------

Using a callback, you can easily log more values with TensorBoard.
Here is a simple example on how to log both additional tensor or arbitrary scalar value:

.. code-block:: python
  
    import numpy as np

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common import logger

    model = SAC("MlpPolicy", "Pendulum-v0", tensorboard_log="/tmp/sac/", verbose=1)


    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Log scalar value (here a random variable)
            value = np.random.random()
            logger.record('random_value', value)
            return True


    model.learn(50000, callback=TensorboardCallback())

