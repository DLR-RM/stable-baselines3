.. _tensorboard:

Tensorboard Integration
=======================

Basic Usage
------------

To use Tensorboard with stable baselines3, you simply need to pass the location of the log folder to the RL agent:

.. code-block:: python

    from stable_baselines3 import A2C

    model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10000)


You can also define custom logging name when training (by default it is the algorithm name)

.. code-block:: python

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

It will display information such as the episode reward (when using a ``Monitor`` wrapper), the model losses and other parameter unique to some models.

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
            self.logger.record('random_value', value)
            return True


    model.learn(50000, callback=TensorboardCallback())

Logging Videos
--------------

TensorBoard supports periodic logging of video data, which helps evaluating agents at various stages during training.

.. warning::
    To support video logging `moviepy <https://zulko.github.io/moviepy/>`_ must be installed otherwise, TensorBoard ignores the video and logs a warning.

Here is an example of how to render an episode and log the resulting video to TensorBoard at regular intervals:

.. code-block:: python

    import gym
    import torch as th

    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.logger import Video
    from typing import Any, Dict


    class VideoRecorderCallback(BaseCallback):
        def __init__(self, eval_env: gym.Env, check_freq: int):
            """
            Records a video of an agent's trajectory traversing `eval_env` and logs it to TensorBoard

            :param eval_env: A gym environment from which the trajectory is recorded
            :param check_freq: Render the agent's trajectory every eval_freq call of the callback.
            """
            super().__init__()
            self._eval_env = eval_env
            self._check_freq = check_freq

        def _on_step(self) -> bool:
            if self.n_calls % self._check_freq == 0:
                screens = []

                def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                    """
                    Renders the environment in its current state, recording the screen in the captured `screens` list

                    :param _locals: A dictionary containing all local variables of the callback's scope
                    :param _globals: A dictionary containing all global variables of the callback's scope
                    """
                    screen = self._eval_env.render(mode="rgb_array")
                    screens.append(screen.transpose(2, 0, 1))

                evaluate_policy(self.model, self._eval_env, callback=grab_screens)
                self.logger.record("trajectory/video", Video(th.ByteTensor([screens]), fps=40))
            return True


    model = A2C("MlpPolicy", "CartPole-v1", tensorboard_log=f"runs/")
    video_recorder = VideoRecorderCallback(gym.make("CartPole-v1"), 1000)
    model.learn(total_timesteps=int(1e4), callback=video_recorder)
