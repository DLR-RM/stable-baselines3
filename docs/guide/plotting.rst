.. _plotting:

========
Plotting
========

Stable Baselines3 provides utilities for plotting training results to monitor and visualize your agent's learning progress.

.. note::

    **Recommended Approach**: For production-ready plotting with advanced features, we recommend using the 
    `RL Baselines3 Zoo plotting scripts <https://rl-baselines3-zoo.readthedocs.io/en/master/guide/plot.html>`_ 
    which provide comprehensive plotting capabilities with confidence intervals, statistical analysis, and publication-ready visualizations.


Recommended Approach: RL Baselines3 Zoo Plotting
==============================================

**For production use, we strongly recommend using the plotting utilities from RL Baselines3 Zoo**, which provide 
comprehensive plotting capabilities far beyond the basic utilities included in Stable Baselines3.

The RL Zoo plotting tools offer:

- **Statistical analysis** with confidence intervals and error bars
- **Multi-algorithm comparisons** across different environments  
- **Publication-ready plots** with customizable styling
- **Advanced evaluation plots** with proper statistical testing
- **Automated hyperparameter comparison** visualizations

Installation and Usage
----------------------

First, install RL Baselines3 Zoo:

.. code-block:: bash

    pip install rl_zoo3[plots]

Basic Training Plot Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Plot training results for a single experiment
    python -m rl_zoo3.plots.plot_train --algo ppo --env CartPole-v1 --exp-folder logs/

    # Compare multiple algorithms on the same environment
    python -m rl_zoo3.plots.plot_train --algo ppo a2c --env CartPole-v1 --exp-folder logs/

    # Plot with custom styling and save to file
    python -m rl_zoo3.plots.plot_train --algo ppo --env CartPole-v1 --exp-folder logs/ \
           --figsize 10 8 --format pdf --output-file ppo_cartpole_results.pdf

Evaluation and Comparison Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Generate comprehensive evaluation plots with confidence intervals
    python -m rl_zoo3.plots.all_plots --algo ppo a2c sac --env-folder logs/ --output-folder eval_plots/

    # Create comparison plots from benchmark results
    python -m rl_zoo3.plots.plot_from_file --input benchmark_results.pkl --output comparison_plot.pdf

Advanced Features
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Plot with specific confidence intervals and statistical tests
    python -m rl_zoo3.plots.plot_train --algo ppo --env CartPole-v1 --exp-folder logs/ \
           --confidence-interval 0.95 --smooth 10 --million-timesteps

    # Generate plots for multiple environments simultaneously  
    python -m rl_zoo3.plots.all_plots --env CartPole-v1 LunarLander-v2 --algo ppo \
           --env-folder logs/ --rliable

For complete documentation on the plotting capabilities, see the 
`RL Baselines3 Zoo plotting guide <https://rl-baselines3-zoo.readthedocs.io/en/master/guide/plot.html>`_.


Basic SB3 Plotting (Simple Use Cases)
======================================

The following sections document the basic plotting utilities included directly in Stable Baselines3. 
These are suitable for quick debugging and simple visualizations, but we recommend using RL Zoo3 for any serious analysis.

Simple Plotting Examples: Single Training Run
---------------------------------------------

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


Simple Plotting Modes
---------------------

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


Manual Data Processing with SB3 Utilities
------------------------------------------

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
        plt.ylabel("Average Episode Reward (50-episode window)")
        plt.title("Smoothed Episode Rewards")

    plt.tight_layout()
    plt.show()


Monitor File Format
-------------------

The ``Monitor`` wrapper saves training data in CSV format with the following columns:

- ``r``: Episode reward
- ``l``: Episode length (number of steps)
- ``t``: Timestamp (wall-clock time when episode ended)

Additional columns may be present if you log custom metrics in the environment's info dict.

.. note::

    The plotting functions automatically handle multiple monitor files from the same directory,
    which occurs when using vectorized environments. The episodes are loaded and sorted by timestamp
    to maintain proper chronological order.


Real-Time Monitoring and Integrations
====================================

For real-time monitoring during training, consider using the plotting functions within callbacks
(see the `Callbacks guide <callbacks.html>`_) or integrating with external monitoring tools.

**Weights & Biases Integration**

You can log plots to Weights & Biases for remote monitoring:

.. code-block:: python

    import wandb
    from stable_baselines3.common.monitor import load_results
    
    # Log plots to W&B
    df = load_results(log_dir)
    wandb.log({"episode_reward": wandb.plot.line_series(
        xs=df.index, 
        ys=[df['r']], 
        keys=["reward"],
        title="Episode Rewards",
        xname="Episode"
    )})

**TensorBoard**

Training metrics are automatically logged to TensorBoard when you specify a ``tensorboard_log`` directory
during model creation. The plotting utilities complement TensorBoard by providing publication-ready figures.

.. note::

    For comprehensive real-time monitoring and advanced plotting, we recommend using the RL Baselines3 Zoo 
    plotting tools alongside TensorBoard and Weights & Biases (see the `Integrations guide <integrations.html>`_).