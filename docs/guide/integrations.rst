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

You can see the list of stable-baselines3 saved models here: https://huggingface.co/models?other=stable-baselines3

Official pre-trained models are saved in the SB3 organization on the hub: https://huggingface.co/sb3

We wrote a tutorial on how to use 🤗 Hub and Stable-Baselines3 here: https://colab.research.google.com/drive/1GI0WpThwRHbl-Fu2RHfczq6dci5GBDVE#scrollTo=q4cz-w9MdO7T

Installation
-------------

.. code-block:: bash

 pip install huggingface_hub
 pip install huggingface_sb3


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
      filename="ppo-CartPole-v1",
  )
  model = PPO.load(checkpoint)

  # Evaluate the agent and watch it
  eval_env = gym.make("CartPole-v1")
  mean_reward, std_reward = evaluate_policy(
      model, eval_env, render=True, n_eval_episodes=5, deterministic=True, warn=False
  )
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



Upload a model to the Hub
-------------------------

First, you need to be logged in to Hugging Face to upload a model:

- If you're using Colab/Jupyter Notebooks:

.. code-block:: python

 from huggingface_hub import notebook_login
 notebook_login()


- Otheriwse:

.. code-block:: bash

 huggingface-cli login

Then, in this example, we train a PPO agent to play CartPole-v1 and push it to a new repo ``sb3/demo-hf-CartPole-v1``

.. code-block:: python

  from huggingface_sb3 import push_to_hub
  from stable_baselines3 import PPO

  # Define a PPO model with MLP policy network
  model = PPO("MlpPolicy", "CartPole-v1", verbose=1)

  # Train it for 10000 timesteps
  model.learn(total_timesteps=10_000)

  # Save the model
  model.save("ppo-CartPole-v1")

  # Push this saved model to the hf repo
  # If this repo does not exists it will be created
  ## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
  ## filename: the name of the file == "name" inside model.save("ppo-CartPole-v1")
  push_to_hub(
      repo_id="sb3/demo-hf-CartPole-v1",
      filename="ppo-CartPole-v1",
      commit_message="Added Cartpole-v1 model trained with PPO",
  )
