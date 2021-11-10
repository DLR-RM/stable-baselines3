.. _examples:

Examples
========

.. note::

        These examples are only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized               hyperparameters can be found in the RL Zoo `repository <https://github.com/DLR-RM/rl-baselines3-zoo>`_.


Try it online with Colab Notebooks!
-----------------------------------

All the following examples can be executed online using Google colab |colab|
notebooks:

-  `Full Tutorial <https://github.com/araffin/rl-tutorial-jnrr19/tree/sb3>`_
-  `All Notebooks <https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3>`_
-  `Getting Started`_
-  `Training, Saving, Loading`_
-  `Multiprocessing`_
-  `Monitor Training and Plotting`_
-  `Atari Games`_
-  `RL Baselines zoo`_
-  `PyBullet`_
-  `Hindsight Experience Replay`_
-  `Advanced Saving and Loading`_

.. _Getting Started: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
.. _Training, Saving, Loading: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb
.. _Multiprocessing: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb
.. _Monitor Training and Plotting: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
.. _Atari Games: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
.. _Hindsight Experience Replay: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_her.ipynb
.. _RL Baselines zoo: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb
.. _PyBullet: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
.. _Advanced Saving and Loading: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/advanced_saving_loading.ipynb

.. |colab| image:: ../_static/img/colab.svg

Basic Usage: Training, Saving, Loading
--------------------------------------

In the following example, we will train, save and load a DQN model on the Lunar Lander environment.

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

  from stable_baselines3 import DQN
  from stable_baselines3.common.evaluation import evaluate_policy


  # Create environment
  env = gym.make('LunarLander-v2')

  # Instantiate the agent
  model = DQN('MlpPolicy', env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=int(2e5))
  # Save the agent
  model.save("dqn_lunar")
  del model  # delete trained model to demonstrate loading

  # Load the trained agent
  # NOTE: if you have loading issue, you can pass `print_system_info=True`
  # to compare the system on which the model was trained vs the current one
  # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
  model = DQN.load("dqn_lunar", env=env)

  # Evaluate the agent
  # NOTE: If you use wrappers with your environment that modify rewards,
  #       this will be reflected here. To evaluate with original rewards,
  #       wrap environment in a "Monitor" wrapper before other wrappers.
  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

  # Enjoy trained agent
  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs, deterministic=True)
      obs, rewards, dones, info = env.step(action)
      env.render()


Multiprocessing: Unleashing the Power of Vectorized Environments
----------------------------------------------------------------

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb

.. figure:: https://cdn-images-1.medium.com/max/960/1*h4WTQNVIsvMXJTCpXm_TAw.gif

  CartPole Environment


.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines3 import PPO
  from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
  from stable_baselines3.common.env_util import make_vec_env
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
      # which does exactly the previous steps for you.
      # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
      # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

      model = PPO('MlpPolicy', env, verbose=1)
      model.learn(total_timesteps=25_000)

      obs = env.reset()
      for _ in range(1000):
          action, _states = model.predict(obs)
          obs, rewards, dones, info = env.step(action)
          env.render()


Dict Observations
-----------------

You can use environments with dictionary observation spaces. This is useful in the case where one can't directly
concatenate observations such as an image from a camera combined with a vector of servo sensor data (e.g., rotation angles).
Stable Baselines3 provides ``SimpleMultiObsEnv`` as an example of this kind of of setting.
The environment is a simple grid world but the observations for each cell come in the form of dictionaries.
These dictionaries are randomly initilaized on the creation of the environment and contain a vector observation and an image observation.

.. code-block:: python

  from stable_baselines3 import PPO
  from stable_baselines3.common.envs import SimpleMultiObsEnv


  # Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
  env = SimpleMultiObsEnv(random_start=False)

  model = PPO("MultiInputPolicy", env, verbose=1)
  model.learn(total_timesteps=100_000)


Using Callback: Monitoring Training
-----------------------------------

.. note::

	We recommend reading the `Callback section <callbacks.html>`_

You can define a custom callback function that will be called inside the agent.
This could be useful when you want to monitor training, for instance display live
learning curves in Tensorboard (or in Visdom) or save the best agent.
If your callback returns False, training is aborted early.

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb


.. code-block:: python

  import os

  import gym
  import numpy as np
  import matplotlib.pyplot as plt

  from stable_baselines3 import TD3
  from stable_baselines3.common import results_plotter
  from stable_baselines3.common.monitor import Monitor
  from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
  from stable_baselines3.common.noise import NormalActionNoise
  from stable_baselines3.common.callbacks import BaseCallback


  class SaveOnBestTrainingRewardCallback(BaseCallback):
      """
      Callback for saving a model (the check is done every ``check_freq`` steps)
      based on the training reward (in practice, we recommend using ``EvalCallback``).

      :param check_freq:
      :param log_dir: Path to the folder where the model will be saved.
        It must contains the file created by the ``Monitor`` wrapper.
      :param verbose: Verbosity level.
      """
      def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
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
                  print(f"Num timesteps: {self.num_timesteps}")
                  print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                      print(f"Saving new best model to {self.save_path}")
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
  model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)
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

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
..

.. code-block:: python

  from stable_baselines3.common.env_util import make_atari_env
  from stable_baselines3.common.vec_env import VecFrameStack
  from stable_baselines3 import A2C

  # There already exists an environment generator
  # that will make and wrap atari environments correctly.
  # Here we are also multi-worker training (n_envs=4 => 4 environments)
  env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
  # Frame-stacking with 4 frames
  env = VecFrameStack(env, n_stack=4)

  model = A2C('CnnPolicy', env, verbose=1)
  model.learn(total_timesteps=25_000)

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


PyBullet: Normalizing input features
------------------------------------

Normalizing input features may be essential to successful training of an RL agent
(by default, images are scaled but not other types of input),
for instance when training on `PyBullet <https://github.com/bulletphysics/bullet3/>`__ environments. For that, a wrapper exists and
will compute a running average and standard deviation of input features (it can do the same for rewards).


.. note::

	you need to install pybullet with ``pip install pybullet``


.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb


.. code-block:: python

  import os
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

  # Load the saved statistics
  env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
  env = VecNormalize.load(stats_path, env)
  #  do not update them at test time
  env.training = False
  # reward normalization is not needed at test time
  env.norm_reward = False

  # Load the agent
  model = PPO.load(log_dir + "ppo_halfcheetah", env=env)


Hindsight Experience Replay (HER)
---------------------------------

For this example, we are using `Highway-Env <https://github.com/eleurent/highway-env>`_ by `@eleurent <https://github.com/eleurent>`_.


.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_her.ipynb


.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/parking-env.gif

   The highway-parking-v0 environment.

The parking env is a goal-conditioned continuous control task, in which the vehicle must park in a given space with the appropriate heading.

.. note::

  The hyperparameters in the following example were optimized for that environment.


.. code-block:: python

  import gym
  import highway_env
  import numpy as np

  from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
  from stable_baselines3.common.noise import NormalActionNoise

  env = gym.make("parking-v0")

  # Create 4 artificial transitions per real transition
  n_sampled_goal = 4

  # SAC hyperparams:
  model = SAC(
      "MultiInputPolicy",
      env,
      replay_buffer_class=HerReplayBuffer,
      replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy="future",
        # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
        # we have to manually specify the max number of steps per episode
        max_episode_length=100,
        online_sampling=True,
      ),
      verbose=1,
      buffer_size=int(1e6),
      learning_rate=1e-3,
      gamma=0.95,
      batch_size=256,
      policy_kwargs=dict(net_arch=[256, 256, 256]),
  )

  model.learn(int(2e5))
  model.save("her_sac_highway")

  # Load saved model
  # Because it needs access to `env.compute_reward()`
  # HER must be loaded with the env
  model = SAC.load("her_sac_highway", env=env)

  obs = env.reset()

  # Evaluate the agent
  episode_reward = 0
  for _ in range(100):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      episode_reward += reward
      if done or info.get("is_success", False):
          print("Reward:", episode_reward, "Success?", info.get("is_success", False))
          episode_reward = 0.0
          obs = env.reset()


Learning Rate Schedule
----------------------

All algorithms allow you to pass a learning rate schedule that takes as input the current progress remaining (from 1 to 0).
``PPO``'s ``clip_range``` parameter also accepts such schedule.

The `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ already includes
linear and constant schedules.


.. code-block:: python

  from typing import Callable

  from stable_baselines3 import PPO


  def linear_schedule(initial_value: float) -> Callable[[float], float]:
      """
      Linear learning rate schedule.

      :param initial_value: Initial learning rate.
      :return: schedule that computes
        current learning rate depending on remaining progress
      """
      def func(progress_remaining: float) -> float:
          """
          Progress will decrease from 1 (beginning) to 0.

          :param progress_remaining:
          :return: current learning rate
          """
          return progress_remaining * initial_value

      return func

  # Initial learning rate of 0.001
  model = PPO("MlpPolicy", "CartPole-v1", learning_rate=linear_schedule(0.001), verbose=1)
  model.learn(total_timesteps=20_000)
  # By default, `reset_num_timesteps` is True, in which case the learning rate schedule resets.
  # progress_remaining = 1.0 - (num_timesteps / total_timesteps)
  model.learn(total_timesteps=10_000, reset_num_timesteps=True)


Advanced Saving and Loading
---------------------------------

In this example, we show how to use some advanced features of Stable-Baselines3 (SB3):
how to easily create a test environment to evaluate an agent periodically,
use a policy independently from a model (and how to save it, load it) and save/load a replay buffer.

By default, the replay buffer is not saved when calling ``model.save()``, in order to save space on the disk (a replay buffer can be up to several GB when using images).
However, SB3 provides a ``save_replay_buffer()`` and ``load_replay_buffer()`` method to save it separately.


Stable-Baselines3 automatic creation of an environment for evaluation.
For that, you only need to specify ``create_eval_env=True`` when passing the Gym ID of the environment while creating the agent.
Behind the scene, SB3 uses an :ref:`EvalCallback <callbacks>`.


.. note::

	For training model after loading it, we recommend loading the replay buffer to ensure stable learning (for off-policy algorithms).
	You also need to pass ``reset_num_timesteps=True`` to ``learn`` function which initializes the environment
	and agent for training if a new environment was created since saving the model.


.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/advanced_saving_loading.ipynb


.. code-block:: python

  from stable_baselines3 import SAC
  from stable_baselines3.common.evaluation import evaluate_policy
  from stable_baselines3.sac.policies import MlpPolicy

  # Create the model, the training environment
  # and the test environment (for evaluation)
  model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1,
              learning_rate=1e-3, create_eval_env=True)

  # Evaluate the model every 1000 steps on 5 test episodes
  # and save the evaluation to the "logs/" folder
  model.learn(6000, eval_freq=1000, n_eval_episodes=5, eval_log_path="./logs/")

  # save the model
  model.save("sac_pendulum")

  # the saved model does not contain the replay buffer
  loaded_model = SAC.load("sac_pendulum")
  print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

  # now save the replay buffer too
  model.save_replay_buffer("sac_replay_buffer")

  # load it into the loaded_model
  loaded_model.load_replay_buffer("sac_replay_buffer")

  # now the loaded replay is not empty anymore
  print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

  # Save the policy independently from the model
  # Note: if you don't save the complete model with `model.save()`
  # you cannot continue training afterward
  policy = model.policy
  policy.save("sac_policy_pendulum")

  # Retrieve the environment
  env = model.get_env()

  # Evaluate the policy
  mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

  # Load the policy independently from the model
  saved_policy = MlpPolicy.load("sac_policy_pendulum")

  # Evaluate the loaded policy
  mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)

  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



Accessing and modifying model parameters
----------------------------------------

You can access model's parameters via ``load_parameters`` and ``get_parameters`` functions,
or via ``model.policy.state_dict()`` (and ``load_state_dict()``),
which use dictionaries that map variable names to PyTorch tensors.

These functions are useful when you need to e.g. evaluate large set of models with same network structure,
visualize different layers of the network or modify parameters manually.

Policies also offers a simple way to save/load weights as a NumPy vector, using ``parameters_to_vector()``
and ``load_from_vector()`` method.

Following example demonstrates reading parameters, modifying some of them and loading them to model
by implementing `evolution strategy (es) <http://blog.otoro.net/2017/10/29/visual-evolution-strategies/>`_
for solving the ``CartPole-v1`` environment. The initial guess for parameters is obtained by running
A2C policy gradient updates on the model.

.. code-block:: python

  from typing import Dict

  import gym
  import numpy as np
  import torch as th

  from stable_baselines3 import A2C
  from stable_baselines3.common.evaluation import evaluate_policy


  def mutate(params: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
      """Mutate parameters by adding normal noise to them"""
      return dict((name, param + th.randn_like(param)) for name, param in params.items())


  # Create policy with a small network
  model = A2C(
      "MlpPolicy",
      "CartPole-v1",
      ent_coef=0.0,
      policy_kwargs={"net_arch": [32]},
      seed=0,
      learning_rate=0.05,
  )

  # Use traditional actor-critic policy gradient updates to
  # find good initial parameters
  model.learn(total_timesteps=10_000)

  # Include only variables with "policy", "action" (policy) or "shared_net" (shared layers)
  # in their name: only these ones affect the action.
  # NOTE: you can retrieve those parameters using model.get_parameters() too
  mean_params = dict(
      (key, value)
      for key, value in model.policy.state_dict().items()
      if ("policy" in key or "shared_net" in key or "action" in key)
  )

  # population size of 50 invdiduals
  pop_size = 50
  # Keep top 10%
  n_elite = pop_size // 10
  # Retrieve the environment
  env = model.get_env()

  for iteration in range(10):
      # Create population of candidates and evaluate them
      population = []
      for population_i in range(pop_size):
          candidate = mutate(mean_params)
          # Load new policy parameters to agent.
          # Tell function that it should only update parameters
          # we give it (policy parameters)
          model.policy.load_state_dict(candidate, strict=False)
          # Evaluate the candidate
          fitness, _ = evaluate_policy(model, env)
          population.append((candidate, fitness))
      # Take top 10% and use average over their parameters as next mean parameter
      top_candidates = sorted(population, key=lambda x: x[1], reverse=True)[:n_elite]
      mean_params = dict(
          (
              name,
              th.stack([candidate[0][name] for candidate in top_candidates]).mean(dim=0),
          )
          for name in mean_params.keys()
      )
      mean_fitness = sum(top_candidate[1] for top_candidate in top_candidates) / n_elite
      print(f"Iteration {iteration + 1:<3} Mean top fitness: {mean_fitness:.2f}")
      print(f"Best fitness: {top_candidates[0][1]:.2f}")


SB3 and ProcgenEnv
------------------

Some environments like `Procgen <https://github.com/openai/procgen>`_ already produce a vectorized
environment (see discussion in `issue #314 <https://github.com/DLR-RM/stable-baselines3/issues/314>`_). In order to use it with SB3, you must wrap it in a ``VecMonitor`` wrapper which will also allow
to keep track of the agent progress.

.. code-block:: python

  from procgen import ProcgenEnv

  from stable_baselines3 import PPO
  from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor

  # ProcgenEnv is already vectorized
  venv = ProcgenEnv(num_envs=2, env_name='starpilot')

  # To use only part of the observation:
  # venv = VecExtractDictObs(venv, "rgb")

  # Wrap with a VecMonitor to collect stats and avoid errors
  venv = VecMonitor(venv=venv)

  model = PPO("MultiInputPolicy", venv, verbose=1)
  model.learn(10_000)


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
                         name_prefix=f"random-agent-{env_id}")

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

  model = A2C("MlpPolicy", "LunarLander-v2").learn(100_000)

  images = []
  obs = model.env.reset()
  img = model.env.render(mode='rgb_array')
  for i in range(350):
      images.append(img)
      action, _ = model.predict(obs)
      obs, _, _ ,_ = model.env.step(action)
      img = model.env.render(mode='rgb_array')

  imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
