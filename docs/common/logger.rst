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


Explanation of logger output
----------------------------

You can find below short explanations of the values logged in Stable-Baselines3 (SB3).
Depending on the algorithm used and of the wrappers/callbacks applied, SB3 only logs a subset of those keys during training.

Below you can find an example of the logger output when training a PPO agent:

.. code-block:: bash

  -----------------------------------------
  | eval/                   |             |
  |    mean_ep_length       | 200         |
  |    mean_reward          | -157        |
  | rollout/                |             |
  |    ep_len_mean          | 200         |
  |    ep_rew_mean          | -227        |
  | time/                   |             |
  |    fps                  | 972         |
  |    iterations           | 19          |
  |    time_elapsed         | 80          |
  |    total_timesteps      | 77824       |
  | train/                  |             |
  |    approx_kl            | 0.037781604 |
  |    clip_fraction        | 0.243       |
  |    clip_range           | 0.2         |
  |    entropy_loss         | -1.06       |
  |    explained_variance   | 0.999       |
  |    learning_rate        | 0.001       |
  |    loss                 | 0.245       |
  |    n_updates            | 180         |
  |    policy_gradient_loss | -0.00398    |
  |    std                  | 0.205       |
  |    value_loss           | 0.226       |
  -----------------------------------------


eval/
^^^^^
All ``eval/`` values are computed by the ``EvalCallback``.

- ``mean_ep_length``: Mean episode length
- ``mean_reward``: Mean episodic reward (during evaluation)
- ``success_rate``: Mean success rate during evaluation (1.0 means 100% success), the environment info dict must contain an ``is_success`` key to compute that value

rollout/
^^^^^^^^
- ``ep_len_mean``: Mean episode length (averaged over 100 episodes)
- ``ep_rew_mean``: Mean episodic training reward (averaged over 100 episodes), a ``Monitor`` wrapper is required to compute that value (automatically added by `make_vec_env`).
- ``exploration_rate``: Current value of the exploration rate when using DQN, it corresponds to the fraction of actions taken randomly (epsilon of the "epsilon-greedy" exploration)
- ``success_rate``: Mean success rate during training (averaged over 100 episodes), you must pass an extra argument to the ``Monitor`` wrapper to log that value (``info_keywords=("is_success",)``) and provide ``info["is_success"]=True/False`` on the final step of the episode

time/
^^^^^
- ``episodes``: Total number of episodes
- ``fps``: Number of frames per seconds (includes time taken by gradient update)
- ``iterations``: Number of iterations (data collection + policy update for A2C/PPO)
- ``time_elapsed``: Time in seconds since the beginning of training
- ``total_timesteps``: Total number of timesteps (steps in the environments)

train/
^^^^^^
- ``actor_loss``: Current value for the actor loss for off-policy algorithms
- ``approx_kl``: approximate mean KL divergence between old and new policy (for PPO), it is an estimation of how much changes happened in the update
- ``clip_fraction``: mean fraction of surrogate loss that was clipped (above ``clip_range`` threshold) for PPO.
- ``clip_range``: Current value of the clipping factor for the surrogate loss of PPO
- ``critic_loss``: Current value for the critic function loss for off-policy algorithms, usually error between value function output and TD(0), temporal difference estimate
- ``ent_coef``: Current value of the entropy coefficient (when using SAC)
- ``ent_coef_loss``: Current value of the entropy coefficient loss (when using SAC)
- ``entropy_loss``: Mean value of the entropy loss (negative of the average policy entropy)
- ``explained_variance``: Fraction of the return variance explained by the value function, see https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
  (ev=0 => might as well have predicted zero, ev=1 => perfect prediction, ev<0 => worse than just predicting zero)
- ``learning_rate``: Current learning rate value
- ``loss``: Current total loss value
- ``n_updates``: Number of gradient updates applied so far
- ``policy_gradient_loss``: Current value of the policy gradient loss (its value does not have much meaning)
- ``value_loss``: Current value for the value function loss for on-policy algorithms, usually error between value function output and Monte-Carle estimate (or TD(lambda) estimate)
- ``std``: Current standard deviation of the noise when using generalized State-Dependent Exploration (gSDE)


.. automodule:: stable_baselines3.common.logger
  :members:
