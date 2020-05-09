.. _examples:

Examples
========

Try it online with Colab Notebooks!
-----------------------------------

All the following examples can be executed online using Google colab |colab|
notebooks:

-  `Full Tutorial <https://github.com/araffin/rl-tutorial-jnrr19>`_
-  `All Notebooks <https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3>`_
-  `Getting Started`_
-  `Training, Saving, Loading`_
-  `RL Baselines zoo`_


.. -  `Multiprocessing`_
.. -  `Monitor Training and Plotting`_
.. -  `Atari Games`_
.. -  `Breakout`_ (trained agent included)
.. -  `Hindsight Experience Replay`_

.. _Getting Started: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
.. _Training, Saving, Loading: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb
.. _Multiprocessing: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb
.. _Monitor Training and Plotting: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
.. _Atari Games: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
.. _Breakout: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/breakout.ipynb
.. _Hindsight Experience Replay: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_her.ipynb
.. _RL Baselines zoo: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb

.. |colab| image:: ../_static/img/colab.svg

Basic Usage: Training, Saving, Loading
--------------------------------------

In the following example, we will train, save and load a A2C model on the Lunar Lander environment.

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb


.. figure:: https://cdn-images-1.medium.com/max/960/1*f4VZPKOI0PYNWiwt0la0Rg.gif

  Lunar Lander Environment


.. note::
  LunarLander requires the python package ``box2d``.
  You can install it using ``apt install swig`` and then ``pip install box2d box2d-kengz``

.. .. note::
..   ``load`` function re-creates model from scratch on each call, which can be slow.
..   If you need to e.g. evaluate same model with multiple different sets of parameters, consider
..   using ``load_parameters`` instead.

.. code-block:: python

  import gym

  from stable_baselines3 import A2C
  from stable_baselines3.common.evaluation import evaluate_policy


  # Create environment
  env = gym.make('LunarLander-v2')

  # Instantiate the agent
  model = A2C('MlpPolicy', env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=int(2e5))
  # Save the agent
  model.save("a2c_lunar")
  del model  # delete trained model to demonstrate loading

  # Load the trained agent
  model = A2C.load("a2c_lunar")

  # Evaluate the agent
  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

  # Enjoy trained agent
  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Multiprocessing: Unleashing the Power of Vectorized Environments
----------------------------------------------------------------
..
.. .. image:: ../_static/img/colab-badge.svg
..    :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb

.. figure:: https://cdn-images-1.medium.com/max/960/1*h4WTQNVIsvMXJTCpXm_TAw.gif

  CartPole Environment


.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines3 import PPO
  from stable_baselines3.ppo import MlpPolicy
  from stable_baselines3.common.vec_env import SubprocVecEnv
  from stable_baselines3.common.cmd_util import make_vec_env
  from stable_baselines3.common.utils import set_random_seed

  def make_env(env_id, rank, seed=0):
      """
      Utility function for multiprocessed env.

      :param env_id: (str) the environment ID
      :param num_env: (int) the number of environments you wish to have in subprocesses
      :param seed: (int) the inital seed for RNG
      :param rank: (int) index of the subprocess
      """
      def _init():
          env = gym.make(env_id)
          env.seed(seed + rank)
          return env
      set_random_seed(seed)
      return _init

  if __name__ == '__main__':
      env_id = "CartPole-v1"
      num_cpu = 4  # Number of processes to use
      # Create the vectorized environment
      env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

      # Stable Baselines provides you with make_vec_env() helper
      # which does exactly the previous steps for you:
      # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

      model = PPO('MlpPolicy', env, verbose=1)
      model.learn(total_timesteps=25000)

      obs = env.reset()
      for _ in range(1000):
          action, _states = model.predict(obs)
          obs, rewards, dones, info = env.step(action)
          env.render()



Using Callback: Monitoring Training
-----------------------------------

.. note::

	We recommend reading the `Callback section <callbacks.html>`_

You can define a custom callback function that will be called inside the agent.
This could be useful when you want to monitor training, for instance display live
learning curves in Tensorboard (or in Visdom) or save the best agent.
If your callback returns False, training is aborted early.

.. .. image:: ../_static/img/colab-badge.svg
..    :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
..
.. .. figure:: ../_static/img/learning_curve.png
..
..   Learning curve of TD3 on LunarLanderContinuous environment

.. code-block:: python

  import os

  import gym
  import numpy as np
  import matplotlib.pyplot as plt

  from stable_baselines3 import TD3
  from stable_baselines3.td3 import MlpPolicy
  from stable_baselines3.common import results_plotter
  from stable_baselines3.common.monitor import Monitor
  from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
  from stable_baselines3.common.noise import NormalActionNoise
  from stable_baselines3.common.callbacks import BaseCallback


  class SaveOnBestTrainingRewardCallback(BaseCallback):
      """
      Callback for saving a model (the check is done every ``check_freq`` steps)
      based on the training reward (in practice, we recommend using ``EvalCallback``).

      :param check_freq: (int)
      :param log_dir: (str) Path to the folder where the model will be saved.
        It must contains the file created by the ``Monitor`` wrapper.
      :param verbose: (int)
      """
      def __init__(self, check_freq: int, log_dir: str, verbose=1):
          super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
          self.check_freq = check_freq
          self.log_dir = log_dir
          self.save_path = os.path.join(log_dir, 'best_model')
          self.best_mean_reward = -np.inf

      def _init_callback(self) -> None:
          # Create folder if needed
          if self.save_path is not None:
              os.makedirs(self.save_path, exist_ok=True)

      def _on_step(self) -> bool:
          if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                  print("Num timesteps: {}".format(self.num_timesteps))
                  print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                      print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

          return True

  # Create log dir
  log_dir = "tmp/"
  os.makedirs(log_dir, exist_ok=True)

  # Create and wrap the environment
  env = gym.make('LunarLanderContinuous-v2')
  env = Monitor(env, log_dir)

  # Add some action noise for exploration
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
  # Because we use parameter noise, we should use a MlpPolicy with layer normalization
  model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=0)
  # Create the callback: check every 1000 steps
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
  # Train the agent
  timesteps = 1e5
  model.learn(total_timesteps=int(timesteps), callback=callback)

  plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
  plt.show()


Atari Games
-----------

.. figure:: ../_static/img/breakout.gif

  Trained A2C agent on Breakout

.. figure:: https://cdn-images-1.medium.com/max/960/1*UHYJE7lF8IDZS_U5SsAFUQ.gif

 Pong Environment


Training a RL agent on Atari games is straightforward thanks to ``make_atari_env`` helper function.
It will do `all the preprocessing <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>`_
and multiprocessing for you.

.. .. image:: ../_static/img/colab-badge.svg
..    :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
..

.. code-block:: python

  from stable_baselines3.common.cmd_util import make_atari_env
  from stable_baselines3.common.vec_env import VecFrameStack
  from stable_baselines3 import A2C

  # There already exists an environment generator
  # that will make and wrap atari environments correctly.
  # Here we are also multi-worker training (n_envs=4 => 4 environments)
  env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
  # Frame-stacking with 4 frames
  env = VecFrameStack(env, n_stack=4)

  model = A2C('CnnPolicy', env, verbose=1)
  model.learn(total_timesteps=25000)

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


PyBullet: Normalizing input features
------------------------------------

Normalizing input features may be essential to successful training of an RL agent
(by default, images are scaled but not other types of input),
for instance when training on `PyBullet <https://github.com/bulletphysics/bullet3/>`_ environments. For that, a wrapper exists and
will compute a running average and standard deviation of input features (it can do the same for rewards).


.. note::

	you need to install pybullet with ``pip install pybullet``


.. code-block:: python

  import gym
  import pybullet_envs

  from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
  from stable_baselines3 import PPO

  env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
  # Automatically normalize the input features and reward
  env = VecNormalize(env, norm_obs=True, norm_reward=True,
                     clip_obs=10.)

  model = PPO('MlpPolicy', env)
  model.learn(total_timesteps=2000)

  # Don't forget to save the VecNormalize statistics when saving the agent
  log_dir = "/tmp/"
  model.save(log_dir + "ppo_halfcheetah")
  stats_path = os.path.join(log_dir, "vec_normalize.pkl")
  env.save(stats_path)

  # To demonstrate loading
  del model, env

  # Load the agent
  model = PPO.load(log_dir + "ppo_halfcheetah")

  # Load the saved statistics
  env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
  env = VecNormalize.load(stats_path, env)
  #  do not update them at test time
  env.training = False
  # reward normalization is not needed at test time
  env.norm_reward = False


Record a Video
--------------

Record a mp4 video (here using a random agent).

.. note::

  It requires ``ffmpeg`` or ``avconv`` to be installed on the machine.

.. code-block:: python

  import gym
  from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

  env_id = 'CartPole-v1'
  video_folder = 'logs/videos/'
  video_length = 100

  env = DummyVecEnv([lambda: gym.make(env_id)])

  obs = env.reset()

  # Record the video starting at the first step
  env = VecVideoRecorder(env, video_folder,
                         record_video_trigger=lambda x: x == 0, video_length=video_length,
                         name_prefix="random-agent-{}".format(env_id))

  env.reset()
  for _ in range(video_length + 1):
    action = [env.action_space.sample()]
    obs, _, _, _ = env.step(action)
  # Save the video
  env.close()


Bonus: Make a GIF of a Trained Agent
------------------------------------

.. note::
  For Atari games, you need to use a screen recorder such as `Kazam <https://launchpad.net/kazam>`_.
  And then convert the video using `ffmpeg <https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality>`_

.. code-block:: python

  import imageio
  import numpy as np

  from stable_baselines3 import A2C

  model = A2C("MlpPolicy", "LunarLander-v2").learn(100000)

  images = []
  obs = model.env.reset()
  img = model.env.render(mode='rgb_array')
  for i in range(350):
      images.append(img)
      action, _ = model.predict(obs)
      obs, _, _ ,_ = model.env.step(action)
      img = model.env.render(mode='rgb_array')

  imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
