.. _integrations:

============
Integrations
============

Weights & Biases
================

Weights & Biases provides a callback for experiment tracking that allows to visualize and share results.

The full documentation is available here: https://docs.wandb.ai/guides/integrations/other/stable-baselines-3

.. code-block:: python

  import gym
  import wandb
  from wandb.integration.sb3 import WandbCallback

  from stable_baselines3 import PPO

  config = {
      "policy_type": "MlpPolicy",
      "total_timesteps": 25000,
      "env_name": "CartPole-v1",
  }
  run = wandb.init(
      project="sb3",
      config=config,
      sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
      # monitor_gym=True,  # auto-upload the videos of agents playing the game
      # save_code=True,  # optional
  )

  model = PPO(config["policy_type"], config["env_name"], verbose=1, tensorboard_log=f"runs/{run.id}")
  model.learn(
      total_timesteps=config["total_timesteps"],
      callback=WandbCallback(
          model_save_path=f"models/{run.id}",
          verbose=2,
      ),
  )
  run.finish()


Hugging Face 🤗
===============
The Hugging Face Hub 🤗 is a central place where anyone can share and explore models. It allows you to host your saved models 💾.

You can see the list of stable-baselines3 saved models here: https://huggingface.co/models?library=stable-baselines3
Most of them are available via the RL Zoo.

Official pre-trained models are saved in the SB3 organization on the hub: https://huggingface.co/sb3

We wrote a tutorial on how to use 🤗 Hub and Stable-Baselines3
`here <https://colab.research.google.com/github/huggingface/huggingface_sb3/blob/main/notebooks/sb3_huggingface.ipynb>`_.


Installation
-------------

.. code-block:: bash

 pip install huggingface_sb3


.. note::

 If you use the `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_, pushing/loading models from the hub are already integrated:

 .. code-block:: bash

     # Download model and save it into the logs/ folder
     python -m rl_zoo3.load_from_hub --algo a2c --env LunarLander-v2 -orga sb3 -f logs/
     # Test the agent
     python -m rl_zoo3.enjoy --algo a2c --env LunarLander-v2  -f logs/
     # Push model, config and hyperparameters to the hub
     python -m rl_zoo3.push_to_hub --algo a2c --env LunarLander-v2 -f logs/ -orga sb3 -m "Initial commit"



Download a model from the Hub
-----------------------------
You need to copy the repo-id that contains your saved model.
For instance ``sb3/demo-hf-CartPole-v1``:

.. code-block:: python

  import gym

  from huggingface_sb3 import load_from_hub
  from stable_baselines3 import PPO
  from stable_baselines3.common.evaluation import evaluate_policy

  # Retrieve the model from the hub
  ## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
  ## filename = name of the model zip file from the repository
  checkpoint = load_from_hub(
      repo_id="sb3/demo-hf-CartPole-v1",
      filename="ppo-CartPole-v1.zip",
  )
  model = PPO.load(checkpoint)

  # Evaluate the agent and watch it
  eval_env = gym.make("CartPole-v1")
  mean_reward, std_reward = evaluate_policy(
      model, eval_env, render=True, n_eval_episodes=5, deterministic=True, warn=False
  )
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

You need to define two parameters:

- ``repo-id``: the name of the Hugging Face repo you want to download.
- ``filename``: the file you want to download.


Upload a model to the Hub
-------------------------

You can easily upload your models using two different functions:

1. ``package_to_hub()``: save the model, evaluate it, generate a model card and record a replay video of your agent before pushing the complete repo to the Hub.

2. ``push_to_hub()``: simply push a file to the Hub.


First, you need to be logged in to Hugging Face to upload a model:

- If you're using Colab/Jupyter Notebooks:

.. code-block:: python

 from huggingface_hub import notebook_login
 notebook_login()


- Otherwise:

.. code-block:: bash

 huggingface-cli login


Then, in this example, we train a PPO agent to play CartPole-v1 and push it to a new repo ``sb3/demo-hf-CartPole-v1``

With ``package_to_hub()``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  from stable_baselines3 import PPO
  from stable_baselines3.common.env_util import make_vec_env

  from huggingface_sb3 import package_to_hub

  # Create the environment
  env_id = "CartPole-v1"
  env = make_vec_env(env_id, n_envs=1)

  # Create the evaluation environment
  eval_env = make_vec_env(env_id, n_envs=1)

  # Instantiate the agent
  model = PPO("MlpPolicy", env, verbose=1)

  # Train the agent
  model.learn(total_timesteps=int(5000))

  # This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
  package_to_hub(model=model,
               model_name="ppo-CartPole-v1",
               model_architecture="PPO",
               env_id=env_id,
               eval_env=eval_env,
               repo_id="sb3/demo-hf-CartPole-v1",
               commit_message="Test commit")

You need to define seven parameters:

- ``model``: your trained model.
- ``model_architecture``: name of the architecture of your model (DQN, PPO, A2C, SAC…).
- ``env_id``: name of the environment.
- ``eval_env``: environment used to evaluate the agent.
- ``repo-id``: the name of the Hugging Face repo you want to create or update. It’s <your huggingface username>/<the repo name>.
- ``commit-message``.
- ``filename``: the file you want to push to the Hub.

With ``push_to_hub()``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python


  from stable_baselines3 import PPO
  from stable_baselines3.common.env_util import make_vec_env

  from huggingface_sb3 import push_to_hub

  # Create the environment
  env_id = "CartPole-v1"
  env = make_vec_env(env_id, n_envs=1)

  # Instantiate the agent
  model = PPO("MlpPolicy", env, verbose=1)

  # Train the agent
  model.learn(total_timesteps=int(5000))

  # Save the model
  model.save("ppo-CartPole-v1")

  # Push this saved model .zip file to the hf repo
  # If this repo does not exists it will be created
  ## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
  ## filename: the name of the file == "name" inside model.save("ppo-CartPole-v1")
  push_to_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
    commit_message="Added CartPole-v1 model trained with PPO",
  )

You need to define three parameters:

- ``repo-id``: the name of the Hugging Face repo you want to create or update. It’s <your huggingface username>/<the repo name>.
- ``filename``: the file you want to push to the Hub.
- ``commit-message``.

MLFLow
======

If you want to use `MLFLow <https://github.com/mlflow/mlflow>`_ to track your SB3 experiments,
you can adapt the following code which defines a custom logger output:

.. code-block:: python

  import sys
  from typing import Any, Dict, Tuple, Union

  import mlflow
  import numpy as np

  from stable_baselines3 import SAC
  from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger


  class MLflowOutputFormat(KVWriter):
      """
      Dumps key/value pairs into MLflow's numeric format.
      """

      def write(
          self,
          key_values: Dict[str, Any],
          key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
          step: int = 0,
      ) -> None:

          for (key, value), (_, excluded) in zip(
              sorted(key_values.items()), sorted(key_excluded.items())
          ):

              if excluded is not None and "mlflow" in excluded:
                  continue

              if isinstance(value, np.ScalarType):
                  if not isinstance(value, str):
                      mlflow.log_metric(key, value, step)


  loggers = Logger(
      folder=None,
      output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
  )

  with mlflow.start_run():
      model = SAC("MlpPolicy", "Pendulum-v1", verbose=2)
      # Set custom logger
      model.set_logger(loggers)
      model.learn(total_timesteps=10000, log_interval=1)
