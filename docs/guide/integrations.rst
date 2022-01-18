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


Hugging Face
============

To be added.
