.. _plotting:

========
Plotting
========

Stable Baselines3 provides utilities for plotting training results to monitor and visualize your agent"s learning progress.
The main plotting functionality is provided by the ``results_plotter`` module, which can load monitor files created during training and generate various plots.


Basic Plotting: Single Training Run
===================================

The simplest way to plot training results is to use the ``plot_results`` function after training an agent.
This function reads monitor files created by the ``Monitor`` wrapper and plots the episode rewards over time.

.. code-block:: python

    import os
    import gymnasium as gym
    import matplotlib.pyplot as plt

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import plot_results
    from stable_baselines3.common import results_plotter

    # Create log directory
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment with Monitor
    env = gym.make("CartPole-v1")
    env = Monitor(env, log_dir)

    # Train the agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20_000)

    # Plot the results
    plot_results([log_dir], 20_000, results_plotter.X_TIMESTEPS, "PPO CartPole")
    plt.show()


Different Plotting Modes
========================

The plotting functions support three different x-axis modes:

- ``X_TIMESTEPS``: Plot rewards vs. timesteps (default)
- ``X_EPISODES``: Plot rewards vs. episode number
- ``X_WALLTIME``: Plot rewards vs. wall-clock time in hours

.. code-block:: python

    import matplotlib.pyplot as plt
    from stable_baselines3.common import results_plotter

    # Plot by timesteps (shows sample efficiency)
    # plot_results([log_dir], None, results_plotter.X_TIMESTEPS, "Rewards vs Timesteps")
    # By Episodes
    plot_results([log_dir], None, results_plotter.X_EPISODES, "Rewards vs Episodes")
    # plot_results([log_dir], None, results_plotter.X_WALLTIME, "Rewards vs Time")

    plt.tight_layout()
    plt.show()


Advanced Plotting with Manual Data Processing
=============================================

For more control over the plotting, you can use the underlying functions to process the data manually:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3.common.monitor import load_results
    from stable_baselines3.common.results_plotter import ts2xy, window_func

    # Load the results
    df = load_results(log_dir)

    # Convert dataframe (x=timesteps, y=episodic return)
    x, y = ts2xy(df, "timesteps")

    # Plot raw data
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, s=2, alpha=0.6)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Raw Episode Rewards")

    # Plot smoothed data with custom window
    plt.subplot(2, 1, 2)
    if len(x) >= 50:  # Only smooth if we have enough data
        x_smooth, y_smooth = window_func(x, y, 50, np.mean)
        plt.plot(x_smooth, y_smooth, linewidth=2)
        plt.xlabel("Timesteps")
        plt.ylabel("Average Episode Reward (50-episode window)"")
        plt.title("Smoothed Episode Rewards")

    plt.tight_layout()
    plt.show()


Monitor File Format
===================

The ``Monitor`` wrapper saves training data in CSV format with the following columns:

- ``r``: Episode reward
- ``l``: Episode length (number of steps)
- ``t``: Timestamp (wall-clock time when episode ended)

Additional columns may be present if you log custom metrics in the environment"s info dict.

.. note::

    The plotting functions automatically handle multiple monitor files from the same directory,
    which occurs when using vectorized environments. The episodes are loaded and sorted by timestamp
    to maintain proper chronological order.


Advanced Plotting with RL Baselines3 Zoo
========================================

For more advanced plotting capabilities, including:

- Comparing results across different environments
- Publication-ready plots with confidence intervals
- Evaluation plots with error bars

We recommend using the plotting scripts from `RL Baselines3 Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

- `plot_train.py <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/plots/plot_train.py>`_: For training plots
- `all_plots.py <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/plots/all_plots.py>`_: For comprehensive evaluation plots
- `plot_from_file.py <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/plots/plot_from_file.py>`_: For plotting from saved results

These scripts provide production-ready plotting with many additional features not available in the basic SB3 plotting utilities.


Real-Time Monitoring
====================

For real-time monitoring during training, consider using the plotting functions within callbacks
(see the `Callbacks guide <callbacks.html>`_) or integrating with tools like `Tensorboard <tensorboard.html>`_  Weights & Biases
(see the `Integrations guide <integrations.html>`_).
